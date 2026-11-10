"""
Bản ghi schema chung (CORPUS_SCHEMA) của VJOL, sinh từ kết quả trích text (interim/vjol_text/) và metadata của tạp chí.

Mỗi bài (một sha256 PDF) một bản ghi. Text là các trang đã làm sạch: bỏ trang bìa / mục lục / tài liệu tham khảo
(vi_corpus.vjol.clean) rồi vi_corpus.common.pdf_text.clean_pages (tiêu đề chạy, số trang, ghép dòng), áp dụng lúc sinh bản ghi
chứ không lưu vào interim. Quyền (rights_status) lấy mặc định của registry ("unknown"): chỉ gắn nhãn, không loại (R-01).

Dùng chung cho VISTA (VS-02): vòng sinh bản ghi (iter_paper_records) nhận một RecordAdapter nói cách đọc metadata và dựng phần đầu của `meta` của từng nguồn;
VJOL_ADAPTER ở cuối file giữ nguyên kết quả cũ, VISTA có adapter riêng ở vi_corpus.vista.records.

J-42 (5): meta mang thêm `repairs` ({quy tắc: số chỗ} đã chạy trên bài, xem repairs_of).

Metadata (J-03): raw/vjol/metadata/vjol_merged.jsonl (hàng trăm MB, mỗi dòng một bài) được đọc theo luồng MỘT lượt, chỉ giữ
dòng của các bài có trong checkpoint, nối qua article_id (khớp tên file <article_id>_<galley_id>.pdf). PDF không có dòng
metadata vẫn thành bản ghi, chỉ thiếu trường. Nhãn ngôn ngữ trong metadata chỉ nằm ở `meta` (khoá journal_language_raw), không thay language id của pipeline.
"""

import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from vi_corpus.common.garbled import GARBLED_REASONS, REASON_NO_TEXT, load_vocab
from vi_corpus.common.pdf_text import clean_pages
from vi_corpus.common.registry import SourceSpec
from vi_corpus.common.schema import make_doc_id, text_sha256
from vi_corpus.common.state import read_json
from vi_corpus.vjol.clean import clean_paper_pages
from vi_corpus.vjol.extract import METADATA_REL, OUT_PATH, PDF_STEM, SYLLABLES_REL, TEXT_DIR  # noqa: F401 — OUT_PATH, METADATA_REL: giữ tên cũ cho nơi khác import

logger = logging.getLogger("vi_corpus")

# Trường metadata giữ lại (bỏ abstract và các trường nặng khác để không phình RAM khi kho lớn).
META_FIELDS = ("journal_slug", "journal_title", "year", "language", "doi", "title", "keywords")
# Trường `language` của metadata là nhãn của tạp chí, gần như luôn "en" cho cả bài tiếng Việt (J-22), nên đổi tên khi ghi vào meta để không ai dùng nhầm.
META_RENAMES = {"language": "journal_language_raw"}
META_OUT = tuple(META_RENAMES.get(k, k) for k in META_FIELDS)
# Năm hợp lệ của một bài báo. 1970 là mốc 0 của Unix (R-25, J-33 D): giá trị mặc định khi thiếu ngày, không phải năm xuất bản, nên coi là thiếu.
YEAR_MIN, YEAR_MAX = 1900, 2100
DEFAULT_YEARS = frozenset({1970})


def repairs_of(stats: dict) -> dict[str, int]:
    """
    Trường `repairs` của meta (J-42 (5)): từng quy tắc làm sạch / sửa chữ đã chạy trên bài và số chỗ (chỉ quy tắc có số > 0), gồm cả `hyphen_ambiguous` (chỗ KHÔNG sửa vì mơ hồ). Để lọc bài theo
    mức bị sửa và truy vết một lỗi về quy tắc đã chạm vào nó. Số đếm theo BÀI, không theo đoạn.
    """
    return {k: int(v) for k, v in stats.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and v}


def normalize_year(value: object) -> int | None:
    """
    Năm xuất bản hợp lệ từ giá trị của metadata hoặc tên thư mục năm; None nếu thiếu, không phải số, ngoài [YEAR_MIN, YEAR_MAX] hoặc là năm mặc định (DEFAULT_YEARS).
    """
    try:
        year = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return None if year in DEFAULT_YEARS or not YEAR_MIN <= year <= YEAR_MAX else year


def load_metadata(path: Path, stems: set[str]) -> dict[str, dict]:
    """
    Đọc metadata theo luồng một lượt, giữ dòng của các bài có trong `stems`.

    Args:
        path:  File jsonl (mỗi dòng một bài, có article_id và galleys[].galley_id).
        stems: Tên file PDF không đuôi, dạng "<article_id>_<galley_id>".

    Returns:
        {stem: {META_OUT..., "galley_match": galley_id có trong galleys[] của bài hay không}}. Bài không có dòng metadata
        vắng mặt. Dòng JSON hỏng bị bỏ (đếm và ghi log). File không tồn tại thì trả về {} (ghi log cảnh báo).
    """
    if not path.is_file():
        logger.warning("Không thấy metadata %s: bản ghi VJOL sẽ thiếu tiêu đề / tạp chí / năm", path)
        return {}
    wanted: dict[str, list[str]] = {}
    for stem in stems:
        m = PDF_STEM.match(stem)
        if m:
            wanted.setdefault(m.group(1), []).append(stem)
    rows: dict[str, dict] = {}
    bad = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(row, dict) and str(row.get("article_id")) in wanted:
                rows[str(row["article_id"])] = row
    if bad:
        logger.warning("Metadata %s: bỏ %d dòng JSON hỏng", path, bad)
    out = {}
    for aid, row in rows.items():
        galleys = {str(g.get("galley_id")) for g in row.get("galleys") or [] if isinstance(g, dict)}
        for stem in wanted[aid]:
            out[stem] = {**{META_RENAMES.get(k, k): row.get(k) for k in META_FIELDS}, "galley_match": stem.split("_")[1] in galleys}
    return out


@dataclass(frozen=True)
class RecordAdapter:
    """
    Phần của vòng sinh bản ghi khác nhau giữa các nguồn bài báo.

    Attributes:
        label:         Tên hiển thị trong log ("VJOL", "VISTA").
        text_dir:      Thư mục checkpoint trích text, tương đối so với data_root.
        metadata_rel:  File metadata mặc định, tương đối so với thư mục raw của nguồn.
        load_metadata: Hàm (đường dẫn file metadata, danh sách checkpoint) -> {sha256: dòng metadata đã chuẩn hoá}; bài không có metadata thì vắng mặt.
                       Dòng metadata phải có khoá "journal_title" (tên tạp chí, dùng cắt tiêu đề chạy) và "year".
        meta_head:     Hàm (checkpoint, dòng metadata hoặc None, năm đã chuẩn hoá hoặc None) -> phần đầu của `meta` (mã bài, trường metadata, cờ có / không metadata);
                       các số đo làm sạch (trang, dòng bị bỏ, repairs) do vòng chung thêm sau.
    """

    label: str
    text_dir: str
    metadata_rel: str
    load_metadata: Callable[[Path, list[dict]], dict[str, dict]]
    meta_head: Callable[[dict, dict | None, int | None], dict]


def _vjol_load_metadata(path: Path, checkpoints: list[dict]) -> dict[str, dict]:
    """Metadata VJOL theo sha256 của checkpoint: nối theo tên file PDF đầu tiên của checkpoint (load_metadata)."""
    by_stem = load_metadata(path, {Path(d["paths"][0]).stem for d in checkpoints})
    return {d["sha256"]: by_stem[Path(d["paths"][0]).stem] for d in checkpoints if Path(d["paths"][0]).stem in by_stem}


def _vjol_meta_head(done: dict, md: dict | None, year: int | None) -> dict:
    """Phần đầu `meta` của bài VJOL: article_id, galley_id, các trường metadata của tạp chí, năm, cờ metadata_found / galley_match."""
    return {
        "article_id": done.get("article_id"), "galley_id": done.get("galley_id"),
        **({k: md[k] for k in META_OUT if k != "year"} if md else {k: None for k in META_OUT if k != "year"}),
        "year": year, "year_dir": done.get("year_dir"), "metadata_found": md is not None,
        "galley_match": bool(md and md["galley_match"]),
    }


VJOL_ADAPTER = RecordAdapter(label="VJOL", text_dir=TEXT_DIR, metadata_rel=METADATA_REL, load_metadata=_vjol_load_metadata, meta_head=_vjol_meta_head)


def in_shard(sha256: str, shard: tuple[int, int] | None) -> bool:
    """True nếu PDF có sha256 này thuộc phần `shard` = (I, N) (băm 8 ký tự hex đầu chia dư N); shard None thì mọi PDF đều thuộc."""
    return shard is None or int(sha256[:8], 16) % shard[1] == shard[0]


def iter_records(spec: SourceSpec, data_root: Path, limit_files: int | None = None, with_meta: bool = False,
                 metadata_path: Path | None = None, hyphen_chooser=None, shard: tuple[int, int] | None = None) -> Iterator[dict]:
    """Bản ghi VJOL: iter_paper_records với VJOL_ADAPTER (xem iter_paper_records về các tham số)."""
    yield from iter_paper_records(spec, data_root, VJOL_ADAPTER, limit_files, with_meta, metadata_path, hyphen_chooser, shard)


def iter_paper_records(spec: SourceSpec, data_root: Path, adapter: RecordAdapter, limit_files: int | None = None, with_meta: bool = False,
                       metadata_path: Path | None = None, hyphen_chooser=None, shard: tuple[int, int] | None = None) -> Iterator[dict]:
    """
    Sinh bản ghi CORPUS_SCHEMA từ checkpoint trích text của một nguồn bài báo (adapter.text_dir), mỗi checkpoint (mỗi PDF duy nhất) một bản ghi.

    Bỏ qua checkpoint lỗi / bị skip / không có chữ. Trang needs_ocr (chưa OCR) thành trang rỗng vì text của chúng là rác; trang OCR bị loại
    (flagged) có text rỗng nên không vào kho. source_path là đường dẫn PDF (nằm trong checkpoint).

    Args:
        spec:          Nguồn trong registry (vjol, vista).
        data_root:     Thư mục gốc dữ liệu.
        adapter:       Cách đọc metadata và dựng phần đầu của `meta` của nguồn.
        limit_files:   Chỉ đọc N checkpoint đầu (chạy thử).
        with_meta:     Thêm khoá "meta" (JSON): phần đầu do adapter.meta_head dựng (mã bài, tạp chí, năm, tiêu đề ...), có/không metadata, số trang lỗi font phải OCR lại (J-10),
                       số trang / dòng bị bỏ (bìa, mục lục, tài liệu tham khảo, tiêu đề chạy).
        metadata_path: File metadata (mặc định <data-root>/<spec.path>/<adapter.metadata_rel>); chỉ đọc khi with_meta.
        hyphen_chooser: Bộ chọn cách đọc cho chỗ "ph-ơng" mơ hồ (vi_corpus.vjol.hyphen_lm.LmChooser, J-57); None thì chỉ dùng bằng chứng dấu thanh cả bài.
        shard:         (I, N): chỉ lấy checkpoint thuộc phần I trong N phần theo sha256 (in_shard); metadata chỉ nối cho phần này. None = lấy hết.

    Cuối lượt ghi log số bài bị bỏ theo lý do (lỗi / bỏ qua / không có chữ), để bài không mất im lặng trước ingest (J-33 B).
    Tập âm tiết (<data-root>/processed/ocr_bakeoff/syllables.json) dùng để khôi phục "ư" bị font vẽ thành "-" (vi_corpus.vjol.hyphen_fix, J-33 A); thiếu file thì bỏ bước đó (ghi cảnh báo).
    """
    checkpoints = []
    dropped = {"error": 0, "skipped": 0, "no_text": 0}
    vocab = load_vocab(data_root / SYLLABLES_REL)
    if vocab is None:
        logger.warning("Không thấy tập âm tiết %s: bỏ bước khôi phục \"ư\" cho trang font gạch nối (J-33 A)", data_root / SYLLABLES_REL)
    for path in sorted((data_root / adapter.text_dir).glob("*.json"))[:limit_files]:
        done = read_json(path)
        if not done or not in_shard(path.stem, shard):  # tên checkpoint là sha256 (batch_extract.checkpoint_path)
            continue
        if done.get("error") or done.get("skipped"):
            dropped["error" if done.get("error") else "skipped"] += 1
            continue
        checkpoints.append(done)
    meta_rows: dict[str, dict] = {}
    if with_meta:
        meta_rows = adapter.load_metadata(metadata_path or data_root / spec.path / adapter.metadata_rel, checkpoints)
    for done in checkpoints:
        pages = ["" if p["method"] == "needs_ocr" else p["text"] for p in done["pages"]]  # giữ chỉ số trang cho heuristic bìa
        md = meta_rows.get(done["sha256"])
        pages, clean_stats = clean_paper_pages(pages, (md or {}).get("journal_title"), vocab, hyphen_chooser)
        page_stats: dict = {}
        text = clean_pages(pages, stats=page_stats)
        if not text:
            dropped["no_text"] += 1
            continue
        rec = {
            "doc_id": make_doc_id(spec.key, done["sha256"], 0),
            "source_key": spec.key,
            "source_path": done["paths"][0],
            "source_sha256": text_sha256(text),
            "text": text,
            "language": None,
            "domain": spec.domain,
            "license": spec.license,
            "owner": spec.owner,
            "rights_status": spec.rights_status,
            "quality_band": None,
            "reason_codes": [],
            "token_count": None,
            "dedup_family_id": None,
        }
        if with_meta:
            year = normalize_year((md or {}).get("year")) or normalize_year(done.get("year_dir"))  # J-33 D: 1970 (mặc định khi thiếu ngày) coi là thiếu
            rec["meta"] = json.dumps({
                **adapter.meta_head(done, md, year), "pages": len(done["pages"]),
                "garbled_pages": sum(p.get("reason") in GARBLED_REASONS for p in done["pages"]),
                "no_text_layer_pages": sum(p.get("reason") == REASON_NO_TEXT for p in done["pages"]),
                "page_lines_removed": page_stats.get("lines_removed", 0), **clean_stats,
                "repairs": repairs_of({**clean_stats, "page_lines_removed": page_stats.get("lines_removed", 0)}),
            }, ensure_ascii=False)
        yield rec
    if any(dropped.values()):
        logger.info("%s: %d checkpoint không thành bản ghi (lỗi trích %d, bỏ qua %d, không có chữ sau khi làm sạch %d)", adapter.label,
                    sum(dropped.values()), dropped["error"], dropped["skipped"], dropped["no_text"])
