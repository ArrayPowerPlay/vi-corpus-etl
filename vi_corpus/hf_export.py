"""
Dựng thư mục dataset để đẩy lên Hugging Face từ đầu ra của `make pipeline-cpu` (VJOL + VISTA chia phần).

Đọc từng phần `<nguồn>_full_s<i>of<N>/{clean,cpt_segments}.parquet` (mỗi lần một phần, không nạp cả nguồn vào RAM), làm phẳng
cột `meta` (JSON) thành cột riêng có kiểu, bỏ cột nội bộ (minhash, chi tiết chấm điểm), thêm `split_bucket` (băm của
dedup_family_id, cùng công thức `knowledge.split_bucket`) rồi ghi parquet zstd chia file. Không ghi `knowledge_units`:
mọi đơn vị của hai nguồn này là `unit_type=cpt` với văn bản y hệt `clean`, nên chỉ nhân đôi dung lượng.

Bố cục đầu ra:
    <out>/data/vjol/train-XXXXX-of-NNNNN.parquet      văn bản sạch theo đoạn (chunk), metadata bài báo VJOL
    <out>/data/vista/train-XXXXX-of-NNNNN.parquet     như trên cho VISTA
    <out>/data/cpt_segments/<nguồn>-XXXXX-of-NNNNN.parquet   đoạn liền mạch cho CPT (cả hai nguồn, cùng schema)
    <out>/README.md                                   dataset card (YAML configs + mô tả + số liệu thật)
    <out>/export_stats.json                           số dòng / token từng file, dùng để kiểm lại sau khi đẩy
"""

import json
import hashlib
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger("hf_export")

LICENSE_TEXT = "chưa rõ (bản quyền thuộc tạp chí / tác giả)"

# Cột chung của config văn bản sạch (nguồn nào cũng có), lấy thẳng từ clean.parquet.
COMMON_CLEAN = [
    ("doc_id", pa.string()), ("parent_doc_id", pa.string()), ("chunk_index", pa.int64()), ("source_key", pa.string()),
    ("text", pa.string()), ("language", pa.string()), ("lang_score", pa.float64()), ("token_count", pa.int64()),
    ("char_count", pa.int64()), ("word_count", pa.int64()), ("quality_band", pa.string()), ("quality_score", pa.float64()),
    ("reason_codes", pa.list_(pa.string())), ("dedup_family_id", pa.string()), ("split_bucket", pa.int64()),
    ("domain", pa.string()), ("rights_status", pa.string()), ("license", pa.string()), ("source_path", pa.string()),
    ("source_sha256", pa.string()),
]
# Cột làm phẳng từ meta, theo nguồn: (tên cột, kiểu, khoá trong meta).
META_COLS = {
    "vjol": [
        ("title", pa.string(), "title"), ("journal_title", pa.string(), "journal_title"),
        ("journal_slug", pa.string(), "journal_slug"), ("year", pa.int64(), "year"), ("doi", pa.string(), "doi"),
        ("keywords", pa.list_(pa.string()), "keywords"), ("article_id", pa.string(), "article_id"),
        ("galley_id", pa.string(), "galley_id"), ("metadata_found", pa.bool_(), "metadata_found"),
        ("pdf_pages", pa.int64(), "pages"), ("garbled_pages", pa.int64(), "garbled_pages"),
        ("no_text_layer_pages", pa.int64(), "no_text_layer_pages"),
    ],
    "vista": [
        ("title", pa.string(), "title"), ("title_en", pa.string(), "title_en"), ("authors", pa.list_(pa.string()), "authors"),
        ("journal_title", pa.string(), "journal_title"), ("issn", pa.string(), "issn"), ("issue_number", pa.string(), "issue_number"),
        ("page_range", pa.string(), "page_range"), ("year", pa.int64(), "year"), ("keywords", pa.list_(pa.string()), "keywords"),
        ("research_field_code", pa.string(), "research_field_code"), ("research_field_name", pa.string(), "research_field_name"),
        ("document_type", pa.string(), "document_type"), ("publication_id", pa.string(), "publication_id"),
        ("publication_ids", pa.list_(pa.string()), "publication_ids"), ("shared_pdf", pa.bool_(), "shared_pdf"),
        ("metadata_found", pa.bool_(), "metadata_found"), ("pdf_pages", pa.int64(), "pages"),
        ("garbled_pages", pa.int64(), "garbled_pages"), ("no_text_layer_pages", pa.int64(), "no_text_layer_pages"),
    ],
}
# Cột của config cpt_segments (cùng schema cho mọi nguồn); `year` và `journal_title` lấy từ meta_json.
CPT_COLS = [
    ("segment_id", pa.string()), ("parent_doc_id", pa.string()), ("source_key", pa.string()), ("text", pa.string()),
    ("token_count", pa.int64()), ("title", pa.string()), ("journal_title", pa.string()), ("year", pa.int64()),
    ("language", pa.string()), ("domain", pa.string()), ("rights_status", pa.string()), ("license", pa.string()),
    ("dedup_family_id", pa.string()), ("split_bucket", pa.int64()), ("doc_ids", pa.list_(pa.string())),
    ("quality_bands", pa.list_(pa.string())), ("chunk_start", pa.int64()), ("chunk_end", pa.int64()),
    ("chunk_token_counts", pa.list_(pa.int64())), ("chunk_char_offsets", pa.list_(pa.int64())),
    ("doc_start", pa.bool_()), ("doc_end", pa.bool_()), ("doc_complete", pa.bool_()), ("break_reason", pa.string()),
    ("source_path", pa.string()), ("source_sha256", pa.string()),
]


def split_bucket(family: str) -> int:
    """Số 0..99 ổn định từ dedup_family_id (cùng công thức `vi_corpus.pipeline.knowledge.split_bucket`)."""
    return int(hashlib.sha1(family.encode("utf-8")).hexdigest()[:8], 16) % 100


def clean_schema(source: str) -> pa.schema:
    """Schema của config văn bản sạch cho một nguồn: cột chung rồi cột metadata riêng của nguồn."""
    return pa.schema(COMMON_CLEAN + [(n, t) for n, t, _ in META_COLS[source]])


def cpt_schema() -> pa.schema:
    """Schema của config cpt_segments."""
    return pa.schema(CPT_COLS)


def _coerce(value, typ: pa.DataType):
    """Ép giá trị từ JSON về kiểu cột; không ép được thì trả None (không bịa số thay thế)."""
    if value is None:
        return None
    try:
        if pa.types.is_string(typ):
            return str(value)
        if pa.types.is_integer(typ):
            return int(value)
        if pa.types.is_boolean(typ):
            return bool(value)
        if pa.types.is_list(typ):
            if not isinstance(value, (list, tuple)):
                return None
            return [str(x) for x in value if x is not None]
    except (TypeError, ValueError):
        return None
    return value


def flatten_clean(rows: list[dict], source: str) -> pa.Table:
    """Đổi các dòng clean.parquet (dict) thành bảng theo `clean_schema(source)`, làm phẳng meta và thêm split_bucket."""
    schema = clean_schema(source)
    cols: dict[str, list] = {f.name: [] for f in schema}
    for r in rows:
        meta = json.loads(r["meta"]) if r.get("meta") else {}
        for name, typ in COMMON_CLEAN:
            if name == "split_bucket":
                cols[name].append(split_bucket(r["dedup_family_id"]))
            else:
                cols[name].append(_coerce(r.get(name), typ) if name != "reason_codes" else list(r.get(name) or []))
        for name, typ, key in META_COLS[source]:
            cols[name].append(_coerce(meta.get(key), typ))
    return pa.table(cols, schema=schema)


def flatten_cpt(rows: list[dict]) -> pa.Table:
    """Đổi các dòng cpt_segments.parquet thành bảng theo `cpt_schema()`, lấy năm và tên tạp chí từ meta_json."""
    schema = cpt_schema()
    cols: dict[str, list] = {f.name: [] for f in schema}
    for r in rows:
        meta = json.loads(r["meta_json"]) if r.get("meta_json") else {}
        for name, typ in CPT_COLS:
            if name == "year":
                cols[name].append(_coerce(meta.get("year"), typ))
            elif name == "journal_title":
                cols[name].append(_coerce(meta.get("journal_title"), typ))
            elif pa.types.is_list(typ) and not pa.types.is_string(typ.value_type):
                cols[name].append(list(r.get(name) or []))
            else:
                cols[name].append(r.get(name))
    return pa.table(cols, schema=schema)


def shard_dirs(runs: Path, source: str, shards: int) -> list[Path]:
    """Thư mục của mọi phần đã chạy xong của một nguồn; báo lỗi nếu thiếu phần hoặc thiếu file kết quả."""
    dirs = [runs / f"{source}_full_s{i}of{shards}" for i in range(shards)]
    missing = [d.name for d in dirs if not (d / "clean.parquet").exists() or not (d / "cpt_segments.parquet").exists()]
    if missing:
        raise SystemExit(f"{source}: thiếu clean/cpt_segments ở {len(missing)} phần, ví dụ {missing[:3]}")
    return dirs


def write_group(dirs: list[Path], fname: str, out_path: Path, kind: str, source: str, batch_rows: int = 4000) -> dict:
    """Ghi các file `fname` của một nhóm phần vào một file parquet zstd; trả về số dòng, token và byte.

    `kind` = "clean" (làm phẳng theo `source`) hoặc "cpt_segments". Ghi ra `.part` rồi đổi tên nên file dở không lẫn vào kết quả.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.part")
    schema = clean_schema(source) if kind == "clean" else cpt_schema()
    rows = tokens = 0
    with pq.ParquetWriter(tmp, schema, compression="zstd", compression_level=6) as w:
        for d in dirs:
            for batch in pq.ParquetFile(d / fname).iter_batches(batch_size=batch_rows):
                table = flatten_clean(batch.to_pylist(), source) if kind == "clean" else flatten_cpt(batch.to_pylist())
                w.write_table(table)
                rows += table.num_rows
                tokens += sum(x or 0 for x in table.column("token_count").to_pylist())
    tmp.rename(out_path)
    return {"file": str(out_path.relative_to(out_path.parents[2])), "kind": kind, "source": source, "rows": rows,
            "tokens": tokens, "bytes": out_path.stat().st_size}


def plan_tasks(runs: Path, out: Path, source: str, shards: int, files: int, cpt_files: int) -> list[tuple]:
    """Chia các phần của một nguồn thành nhóm, mỗi nhóm một file đầu ra; trả về danh sách đối số cho `write_group`."""
    dirs = shard_dirs(runs, source, shards)
    tasks = []
    for kind, fname, n_files in (("clean", "clean.parquet", files), ("cpt_segments", "cpt_segments.parquet", cpt_files)):
        per = -(-len(dirs) // n_files)
        groups = [dirs[i:i + per] for i in range(0, len(dirs), per)]
        for gi, group in enumerate(groups):
            if kind == "clean":
                path = out / "data" / source / f"train-{gi:05d}-of-{len(groups):05d}.parquet"
            else:
                path = out / "data" / "cpt_segments" / f"{source}-{gi:05d}-of-{len(groups):05d}.parquet"
            tasks.append((group, fname, path, kind, source))
    return tasks


def collect_stats(out: Path) -> dict:
    """Số liệu thật của dữ liệu đã xuất (đọc lại file parquet đã ghi): dòng, token, ngôn ngữ, band, năm, số bài."""
    res: dict = {}
    for source in ("vjol", "vista"):
        files = sorted((out / "data" / source).glob("*.parquet"))
        rows = tokens = 0
        lang: dict = {}
        band: dict = {}
        years: dict = {}
        docs: set = set()
        no_layer_docs: set = set()
        for f in files:
            t = pq.read_table(f, columns=["parent_doc_id", "language", "quality_band", "year", "token_count", "no_text_layer_pages"])
            for doc, lg, b, y, tk, nl in zip(*(t.column(c).to_pylist() for c in t.column_names)):
                rows += 1
                tokens += tk or 0
                lang[lg] = lang.get(lg, 0) + (tk or 0)
                band[b] = band.get(b, 0) + 1
                if y:
                    years[y] = years.get(y, 0) + 1
                docs.add(doc)
                if nl:
                    no_layer_docs.add(doc)
        res[source] = {"files": len(files), "chunks": rows, "tokens": tokens, "docs": len(docs), "tokens_by_language": lang,
                       "chunks_by_band": band, "year_min": min(years) if years else None, "year_max": max(years) if years else None,
                       "docs_with_missing_pages": len(no_layer_docs), "bytes": sum(f.stat().st_size for f in files)}
    cpt_rows = cpt_tokens = 0
    for f in sorted((out / "data" / "cpt_segments").glob("*.parquet")):
        t = pq.read_table(f, columns=["token_count"])
        cpt_rows += t.num_rows
        cpt_tokens += sum(x or 0 for x in t.column("token_count").to_pylist())
    res["cpt_segments"] = {"segments": cpt_rows, "tokens": cpt_tokens}
    return res


def _fmt(n: int) -> str:
    """Số nguyên có dấu chấm ngăn nghìn kiểu Việt Nam."""
    return f"{n:,}".replace(",", ".")


def render_card(stats: dict, repo_id: str) -> str:
    """Dataset card (README.md của repo Hugging Face) bằng tiếng Việt, kèm tóm tắt tiếng Anh; số liệu lấy từ `stats`."""
    v, s, c = stats["vjol"], stats["vista"], stats["cpt_segments"]
    total_tok = v["tokens"] + s["tokens"]

    def langs(d: dict) -> str:
        """Chuỗi token theo ngôn ngữ của một nguồn."""
        return ", ".join(f"{k}: {_fmt(x)}" for k, x in sorted(d["tokens_by_language"].items(), key=lambda kv: -kv[1]))

    return f"""---
pretty_name: Kho ngữ liệu bài báo khoa học tiếng Việt (VJOL + VISTA)
language:
- vi
- en
license: other
task_categories:
- text-generation
size_categories:
- 1M<n<10M
tags:
- vietnamese
- scientific-papers
- pretraining
- continued-pretraining
- ocr
configs:
- config_name: vjol
  data_files: data/vjol/*.parquet
  default: true
- config_name: vista
  data_files: data/vista/*.parquet
- config_name: cpt_segments
  data_files: data/cpt_segments/*.parquet
---

# Kho ngữ liệu bài báo khoa học tiếng Việt (VJOL + VISTA)

> **Bản xem trước (v0, chỉ chạy CPU).** Chưa OCR bằng GPU, chưa bật chấm perplexity và sửa chữ bằng mô hình ngôn ngữ; bản cuối sẽ chạy
> lại từ đầu nên số dòng, id đoạn và số token sẽ thay đổi.

*English summary: ~{total_tok / 1e9:.2f}B Qwen3 tokens of Vietnamese (plus some English) scientific-paper text extracted from PDFs of two
Vietnamese portals (VJOL and VISTA), cleaned, language-filtered, quality-scored and deduplicated, with article metadata.
Rights are NOT cleared: see "Bản quyền và cách dùng". Repository `{repo_id}`.*

Văn bản trích từ PDF bài báo của hai cổng tạp chí khoa học Việt Nam, đã làm sạch, lọc ngôn ngữ, chấm chất lượng và khử trùng lặp,
kèm metadata bài báo. Đây là ngữ liệu thô cho huấn luyện tiếp tục (CPT), chưa gắn nhãn tác vụ.

## Các config

| config | nội dung | số dòng | token (Qwen3) |
|---|---|---|---|
| `vjol` (mặc định) | đoạn văn bản sạch từ VJOL (`vjol.info.vn`) | {_fmt(v['chunks'])} đoạn của {_fmt(v['docs'])} bài | {_fmt(v['tokens'])} |
| `vista` | đoạn văn bản sạch từ VISTA (`sti.vista.gov.vn`) | {_fmt(s['chunks'])} đoạn của {_fmt(s['docs'])} bài | {_fmt(s['tokens'])} |
| `cpt_segments` | các đoạn liền nhau của cùng một bài được nối thành một segment (cả hai nguồn) | {_fmt(c['segments'])} segment | {_fmt(c['tokens'])} |

`cpt_segments` chứa lại đúng văn bản của hai config trên ở dạng nối liền, **không** cộng thêm vào tổng token. Chỉ dùng một trong
hai: `vjol` + `vista` (đoạn ~1.000 token) hoặc `cpt_segments` (bài liền mạch, tiện cho đóng gói chuỗi dài).

- VJOL: ngôn ngữ (token) {langs(v)}; năm xuất bản {v['year_min']}-{v['year_max']}.
- VISTA: ngôn ngữ (token) {langs(s)}; năm xuất bản {s['year_min']}-{s['year_max']}.
- Phân bố band chất lượng (số đoạn): VJOL {json.dumps(v['chunks_by_band'], sort_keys=True)}; VISTA {json.dumps(s['chunks_by_band'], sort_keys=True)}.

```python
from datasets import load_dataset
ds = load_dataset("{repo_id}", "vjol", split="train", streaming=True)
```

## Cột

Chung cho `vjol` và `vista`: `doc_id` (đoạn), `parent_doc_id` (bài), `chunk_index`, `text`, `language` (`vi`/`en`, fastText lid.176
theo đoạn), `lang_score`, `token_count`, `char_count`, `word_count`, `quality_band` (A tốt nhất, rồi B, C; band D đã bị loại),
`quality_score`, `reason_codes` (lý do trừ điểm), `dedup_family_id`, `split_bucket` (0..99, băm của `dedup_family_id`; cắt
train/val/test bằng cách so với ngưỡng thì các bài gần trùng luôn rơi cùng một phía), `domain`, `rights_status`, `license`,
`source_path`, `source_sha256` (PDF gốc), `source_key`.

Metadata bài báo (làm phẳng từ nguồn, có thể rỗng khi không khớp được metadata: xem `metadata_found`):
- `vjol`: `title`, `journal_title`, `journal_slug`, `year`, `doi`, `keywords`, `article_id`, `galley_id`.
- `vista`: `title`, `title_en`, `authors`, `journal_title`, `issn`, `issue_number`, `page_range`, `year`, `keywords`,
  `research_field_code`, `research_field_name`, `document_type`, `publication_id`, `publication_ids` (+ `shared_pdf` = một PDF
  thuộc nhiều bài trong metadata).
- Cả hai: `pdf_pages`, `garbled_pages` (số trang bị phát hiện lỗi font: văn bản của các trang này kém tin cậy hơn, có thể lọc theo cột này), `no_text_layer_pages` (số trang không có lớp chữ).

`cpt_segments`: `segment_id`, `parent_doc_id`, `text`, `token_count`, `title`, `journal_title`, `year`, `doc_ids`, `quality_bands`,
`chunk_start`/`chunk_end`, `chunk_token_counts`, `chunk_char_offsets` (để cắt lại đúng ranh giới đoạn), `doc_start`, `doc_end`,
`doc_complete` (false nếu một đoạn sau bị loại nên segment không phủ hết bài), `break_reason`, cùng các cột nguồn/bản quyền ở trên.

## Cách tạo

PDF của hai cổng chỉ được **xử lý**, không thu thập lại ở đây. Quy trình (mã trong repo `vi-corpus-etl`): trích lớp chữ bằng PyMuPDF,
phát hiện lỗi font (chữ Việt sai dấu / mất chữ) và sửa bằng quy tắc (nối lại chữ bị ngắt bằng gạch nối, v.v.); làm sạch ở mức bài báo (bìa, mục lục, phần Tài liệu tham khảo, tiêu đề
trang lặp, che email/số điện thoại); chuẩn hoá Unicode, xoá dòng lặp; cắt đoạn theo token Qwen3; nhận diện ngôn ngữ; chấm chất lượng;
khử trùng lặp chính xác + gần trùng + bao hàm, hai vòng (VJOL được ưu tiên giữ khi trùng với VISTA). Chạy hoàn toàn bằng CPU, nên
**trang scan không có lớp chữ bị bỏ qua** (chưa OCR bằng GPU): {_fmt(v['docs_with_missing_pages'])} bài VJOL và {_fmt(s['docs_with_missing_pages'])} bài VISTA
có ít nhất một trang như vậy (cột `no_text_layer_pages`); văn bản của các bài đó thiếu các trang này.

## Hạn chế đã biết

- Ngưỡng chất lượng, ngôn ngữ và dedup là bản đầu, chưa tinh chỉnh; band C nên dùng thận trọng.
- Văn bản còn lỗi trích xuất rải rác (công thức, bảng, chữ ghép/dính, lỗi OCR còn sót ở trang đã sửa font).
- Không chia train/val/test ở tầng ngữ liệu; dùng `split_bucket`.
- Bài song ngữ Việt-Anh nằm trong cùng đoạn; `language` là nhãn của cả đoạn.

## Bản quyền và cách dùng

**Bản quyền chưa được làm rõ.** Nội dung bài báo thuộc về tạp chí và tác giả; ngữ liệu này không kèm giấy phép từ họ
(`rights_status = unknown`, `license = "{LICENSE_TEXT}"`). Chỉ dùng cho nghiên cứu; không giả định được phép phân phối lại hay
dùng thương mại. Chủ sở hữu muốn gỡ bài: mở thảo luận (Community) của repo này kèm `article_id` / `publication_id`, bài sẽ được xem xét gỡ ở bản sau.
"""
