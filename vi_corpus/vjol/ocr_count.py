"""
Đếm số file PDF và số trang phải OCR của VJOL và VISTA, nhanh và chạy lại được (đọc checkpoint, không băm, không mở PDF).

Vì sao cần: `extract.py --status` (xem `report()` trong vi_corpus/vjol/extract.py) đọc tuần tự TOÀN BỘ nội dung mọi checkpoint (mỗi file chứa chữ của mọi trang)
nên chậm cả giờ trên đĩa mạng. Module này làm cùng việc đếm theo bốn cách:
  1. Chỉ `stat` từng checkpoint (kích thước + thời điểm sửa); chỉ file mới hoặc đã đổi mới bị đọc và parse. Kết quả rút gọn của từng checkpoint nằm trong bộ nhớ đệm
     `<data-root>/state/ocr_count/<nguồn>.json` nên lần chạy sau (kể cả khi lượt trích còn đang chạy) chỉ đọc phần mới.
  2. File cần đọc được chia cho nhiều tiến trình (parse JSON tốn CPU).
  3. Kiểm kê PDF trong raw bằng nhiều luồng (mỗi thư mục năm một luồng) và đối chiếu với `paths` trong checkpoint, nên biết còn bao nhiêu PDF CHƯA có checkpoint
     (chưa quét) mà không phải băm sha256 cả kho.
  4. Phần chưa quét được ước theo từng năm (số file chưa quét x số trang cần OCR trung bình mỗi file đã quét của năm đó), ghi rõ là ước.

"Trang cần OCR" = trang có `method` là `ocr_vlm` (đã OCR xong) hoặc `needs_ocr` (trang scan hoặc lỗi font mà lượt trích không OCR, còn chờ). Trang `text` / `tcvn3`
lấy thẳng từ lớp chữ nên không tính. Chưa tính trang bị OCR lại lần hai sau guard (khoảng 0,6% ở stbook, O-22): số này chỉ biết sau khi chạy GPU.

Số trang tính theo NỘI DUNG DUY NHẤT (một checkpoint = một sha256), còn số file tính theo đường dẫn, nên hai nội dung giống nhau ở hai đường dẫn là 2 file nhưng 1 lần trang.
"""

import csv
import json
import logging
import os
import re
import subprocess
import sys
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

from vi_corpus.common.registry import SOURCES
from vi_corpus.common.state import write_json_atomic
from vi_corpus.vista.layout import VISTA_LAYOUT
from vi_corpus.vjol.extract import SKIP_DIRS, VJOL_LAYOUT, version_is_current
from vi_corpus.vjol.layout import PaperLayout

logger = logging.getLogger("vi_corpus")

LAYOUTS: dict[str, PaperLayout] = {"vjol": VJOL_LAYOUT, "vista": VISTA_LAYOUT}
CACHE_VERSION = 1  # đổi khi đổi dạng mục rút gọn của bộ nhớ đệm: bộ nhớ đệm cũ bị bỏ
CACHE_DIR = "state/ocr_count"
OUT_DIR = "processed/ocr_count"
OCR_DONE = "ocr_vlm"  # trang đã OCR xong
OCR_PENDING = "needs_ocr"  # trang cần OCR mà lượt trích không OCR (còn chờ)
OTHER_YEAR = "(khác)"  # thư mục không phải năm (vd unknown/ của VISTA) và checkpoint không có year_dir
_YEAR = re.compile(r"^\d{4}$")


def _rel(path: Path, data_root: Path) -> str:
    """Đường dẫn dạng chuỗi giống `paths` trong checkpoint: tương đối so với data_root, tuyệt đối nếu nằm ngoài."""
    return str(path.relative_to(data_root)) if path.is_relative_to(data_root) else str(path)


def _scan_pdfs(top: Path) -> list[Path]:
    """Mọi PDF dưới thư mục `top` (đệ quy bằng os.scandir, bỏ file và thư mục ẩn); mỗi thư mục chỉ mở một lần."""
    found, stack = [], [top]
    while stack:
        with os.scandir(stack.pop()) as it:
            for e in it:
                if e.name.startswith("."):
                    continue
                if e.is_dir():
                    stack.append(Path(e.path))
                elif e.name.lower().endswith(".pdf"):
                    found.append(Path(e.path))
    return found


def list_pdfs(root: Path, data_root: Path, workers: int) -> dict[str, list[str]]:
    """
    Kiểm kê PDF trong `root` theo thư mục cấp 1 (cùng quy tắc với `list_pdfs_by_year`: bỏ thư mục ẩn và metadata/), mỗi thư mục cấp 1 một luồng.

    Returns:
        {nhãn năm: [đường dẫn kiểu checkpoint]}; nhãn là tên thư mục năm, OTHER_YEAR cho thư mục không phải 4 chữ số (unknown/) và PDF nằm thẳng trong root.
    """
    tops, by_label = [], {}
    with os.scandir(root) as it:
        for e in sorted(it, key=lambda x: x.name):
            if e.name.startswith("."):
                continue
            if e.is_dir():
                if e.name not in SKIP_DIRS:
                    tops.append(Path(e.path))
            elif e.name.lower().endswith(".pdf"):
                by_label.setdefault(OTHER_YEAR, []).append(_rel(Path(e.path), data_root))
    with ThreadPoolExecutor(max(1, workers)) as pool:
        for top, pdfs in zip(tops, pool.map(_scan_pdfs, tops)):
            label = top.name if _YEAR.match(top.name) else OTHER_YEAR
            by_label.setdefault(label, []).extend(_rel(p, data_root) for p in pdfs)
    return by_label


def summarize_checkpoint(path_str: str) -> tuple[str, dict | None]:
    """
    Đọc một checkpoint và rút gọn thành mục nhỏ của bộ nhớ đệm (chạy trong tiến trình con).

    Returns:
        (tên file, mục) với mục {"pages", "methods", "vlm", "error", "skipped", "current", "year", "paths"}; mục là None nếu file không đọc được
        (đang ghi dở hoặc JSON hỏng; checkpoint ghi nguyên tử nên chỉ gặp khi file bị cắt).
    """
    path = Path(path_str)
    try:
        with open(path, encoding="utf-8") as f:
            done = json.load(f)
        pages = done["pages"]
    except (OSError, ValueError, KeyError, TypeError):
        return path.name, None
    year = done.get("year_dir")
    return path.name, {
        "pages": len(pages),
        "methods": dict(Counter(p["method"] for p in pages)),
        "vlm": dict(Counter(p.get("status") for p in pages if p["method"] == OCR_DONE)),
        "error": bool(done.get("error")),
        "skipped": bool(done.get("skipped")),
        "current": version_is_current(done.get("extract_version")),
        "year": str(year) if year is not None else OTHER_YEAR,
        "paths": done.get("paths", []),
    }


def _load_cache(path: Path) -> dict[str, list]:
    """Bộ nhớ đệm {tên checkpoint: [kích thước, mtime_ns, mục]}; rỗng nếu chưa có, hỏng hoặc khác phiên bản."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data["entries"] if data.get("version") == CACHE_VERSION else {}
    except (OSError, ValueError, KeyError, AttributeError):
        return {}


def _save_cache(path: Path, entries: dict[str, list]) -> None:
    """Ghi bộ nhớ đệm nguyên tử (file tạm riêng mỗi tiến trình), gọn một dòng vì có hàng trăm nghìn mục."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": CACHE_VERSION, "entries": entries}, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def load_entries(ckpt_dir: Path, cache_path: Path | None, workers: int) -> tuple[dict[str, dict], int, int]:
    """
    Mục rút gọn của mọi checkpoint trong `ckpt_dir`: lấy từ bộ nhớ đệm nếu (kích thước, mtime) không đổi, ngược lại đọc lại bằng nhiều tiến trình.

    Args:
        ckpt_dir:   Thư mục checkpoint (vd data/interim/vjol_text).
        cache_path: File bộ nhớ đệm; None = không dùng (đọc lại tất cả, không ghi).
        workers:    Số tiến trình đọc.

    Returns:
        (mục theo tên file, số mục lấy từ bộ nhớ đệm, số file không đọc được).
    """
    cache = _load_cache(cache_path) if cache_path else {}
    fresh: dict[str, list] = {}
    todo: list[str] = []
    stats: dict[str, tuple[int, int]] = {}
    if ckpt_dir.is_dir():
        with os.scandir(ckpt_dir) as it:
            for e in it:
                if not e.name.endswith(".json"):
                    continue
                st = e.stat()
                stats[e.name] = (st.st_size, st.st_mtime_ns)
                old = cache.get(e.name)
                if old and old[0] == st.st_size and old[1] == st.st_mtime_ns:
                    fresh[e.name] = old
                else:
                    todo.append(e.path)
    hits, unreadable = len(fresh), 0
    if todo:
        with Pool(max(1, workers)) as pool:
            for name, entry in tqdm(pool.imap_unordered(summarize_checkpoint, todo, chunksize=32), total=len(todo), desc="Đọc checkpoint", unit="file"):
                if entry is None:
                    unreadable += 1
                else:
                    fresh[name] = [*stats[name], entry]
    if cache_path and (todo or len(fresh) != len(cache)):
        _save_cache(cache_path, fresh)
    return {name: v[2] for name, v in fresh.items()}, hits, unreadable


def aggregate(entries: dict[str, dict], listed: dict[str, list[str]]) -> dict:
    """
    Gộp mục checkpoint và danh sách PDF thành số liệu theo năm và tổng.

    Một PDF "đã quét" nếu đường dẫn của nó nằm trong `paths` của một checkpoint không lỗi. PDF còn lại là "chưa quét" (chưa có checkpoint, hoặc checkpoint lỗi
    sẽ được thử lại). Phần chưa quét được ước theo từng năm bằng số trang cần OCR trung bình mỗi file đã quét của năm đó (năm chưa có file nào đã quét thì bỏ).

    Returns:
        {"total": {...}, "by_year": {nhãn năm: {...}}}; mỗi dict có files, scanned_files, missing_files, contents, errors, skipped, old_version, pages, methods,
        ocr_done, ocr_pending, ocr_pages (= done + pending), vlm (ok / blank / flagged), est_missing_ocr_pages, est_total_ocr_pages.
    """
    covered: set[str] = set()
    for e in entries.values():
        if not e["error"]:
            covered.update(e["paths"])
    by_year: dict[str, dict] = {}

    def row(label: str) -> dict:
        """Dòng của một năm (tạo khi gặp lần đầu)."""
        return by_year.setdefault(label, {"files": 0, "scanned_files": 0, "missing_files": 0, "contents": 0, "errors": 0, "skipped": 0, "old_version": 0,
                                          "pages": 0, "methods": Counter(), "vlm": Counter()})

    for label, paths in listed.items():
        r = row(label)
        r["files"] = len(paths)
        r["scanned_files"] = sum(p in covered for p in paths)
        r["missing_files"] = r["files"] - r["scanned_files"]
    for e in entries.values():
        r = row(e["year"])
        r["contents"] += 1
        r["errors"] += e["error"]
        r["skipped"] += e["skipped"]
        r["old_version"] += not e["current"]
        if not e["error"]:
            r["pages"] += e["pages"]
            r["methods"].update(e["methods"])
            r["vlm"].update(e["vlm"])
    total = {"files": 0, "scanned_files": 0, "missing_files": 0, "contents": 0, "errors": 0, "skipped": 0, "old_version": 0, "pages": 0,
             "methods": Counter(), "vlm": Counter(), "est_missing_ocr_pages": 0.0}
    for label in sorted(by_year):
        r = by_year[label]
        r["ocr_done"], r["ocr_pending"] = r["methods"][OCR_DONE], r["methods"][OCR_PENDING]
        r["ocr_pages"] = r["ocr_done"] + r["ocr_pending"]
        r["est_missing_ocr_pages"] = r["missing_files"] * r["ocr_pages"] / r["scanned_files"] if r["scanned_files"] else 0.0
        r["est_total_ocr_pages"] = r["ocr_pages"] + r["est_missing_ocr_pages"]
        for k, v in r.items():
            if k in total and k not in ("methods", "vlm"):
                total[k] += v
            elif k in ("methods", "vlm"):
                total[k].update(v)
    total["ocr_done"], total["ocr_pending"] = total["methods"][OCR_DONE], total["methods"][OCR_PENDING]
    total["ocr_pages"] = total["ocr_done"] + total["ocr_pending"]
    total["est_total_ocr_pages"] = total["ocr_pages"] + total["est_missing_ocr_pages"]
    for r in [total, *by_year.values()]:
        r["methods"], r["vlm"] = dict(r["methods"]), dict(r["vlm"])
        r["est_missing_ocr_pages"] = round(r["est_missing_ocr_pages"])
        r["est_total_ocr_pages"] = round(r["est_total_ocr_pages"])
    return {"total": total, "by_year": dict(sorted(by_year.items()))}


def count_source(key: str, data_root: Path, workers: int, use_cache: bool = True, root: Path | None = None) -> dict:
    """
    Đếm một nguồn ("vjol" hoặc "vista") và ghi kết quả vào <data-root>/processed/ocr_count/<key>.json và <key>_by_year.csv.

    Args:
        key:       Mã nguồn trong LAYOUTS.
        data_root: Thư mục gốc dữ liệu.
        workers:   Số luồng kiểm kê và số tiến trình đọc checkpoint.
        use_cache: False = bỏ bộ nhớ đệm (đọc lại mọi checkpoint, không ghi lại).
        root:      Thư mục PDF (mặc định <data-root>/<đường dẫn raw trong registry>).

    Returns:
        Kết quả của `aggregate` kèm "source", "cache_hits", "unreadable".
    """
    layout = LAYOUTS[key]
    root = root or data_root / SOURCES[key].path
    listed = list_pdfs(root, data_root, workers) if root.is_dir() else {}
    if not listed:
        logger.warning("%s: không thấy PDF nào trong %s", layout.label, root)
    entries, hits, unreadable = load_entries(data_root / layout.text_dir, data_root / CACHE_DIR / f"{key}.json" if use_cache else None, workers)
    result = {"source": key, **aggregate(entries, listed), "cache_hits": hits, "unreadable": unreadable}
    out = data_root / OUT_DIR
    write_json_atomic(out / f"{key}.json", result)
    with open(out / f"{key}_by_year.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        cols = ["files", "scanned_files", "missing_files", "pages", "ocr_done", "ocr_pending", "ocr_pages", "est_total_ocr_pages", "errors", "old_version"]
        w.writerow(["year", *cols])
        for label, r in result["by_year"].items():
            w.writerow([label, *(r[c] for c in cols)])
    return result


def format_result(result: dict, gpus: int, pages_per_sec: float) -> str:
    """Bản tóm tắt dễ đọc của `count_source`, kèm ước giờ GPU cho số trang cần OCR (chỉ là ước, tốc độ đo trên stbook)."""
    t = result["total"]
    est_hours = t["est_total_ocr_pages"] / (gpus * pages_per_sec) / 3600
    lines = [
        f"== {result['source']} ==",
        f"File PDF trong raw        : {t['files']:,}",
        f"  đã quét (có checkpoint) : {t['scanned_files']:,}",
        f"  CHƯA quét               : {t['missing_files']:,}",
        f"Nội dung duy nhất (ckpt)  : {t['contents']:,}  (lỗi {t['errors']:,}, bỏ qua {t['skipped']:,}, bản trích cũ {t['old_version']:,})",
        f"Tổng số trang (đã quét)   : {t['pages']:,}",
        f"  theo method             : {t['methods']}",
        f"Trang CẦN OCR (đã quét)   : {t['ocr_pages']:,}  = đã OCR {t['ocr_done']:,} + còn chờ {t['ocr_pending']:,}",
        f"  trạng thái trang đã OCR : {t['vlm']}",
    ]
    if t["missing_files"]:
        lines.append(f"Ước cho {t['missing_files']:,} file chưa quét  : +{t['est_missing_ocr_pages']:,} trang (theo trung bình từng năm)")
        lines.append(f"Ước TỔNG trang cần OCR    : {t['est_total_ocr_pages']:,}  (phần chưa quét chỉ là ước; quét nốt bằng --scan-missing để có số thật)")
    lines.append(f"Ước giờ GPU để OCR hết    : {est_hours:,.1f} giờ trên {gpus} GPU ở {pages_per_sec} trang/giây/GPU (chưa tính ~0,6% trang đọc lại)")
    lines.append(f"(bộ nhớ đệm: {result['cache_hits']:,} checkpoint khỏi đọc lại; không đọc được: {result['unreadable']})")
    return "\n".join(lines)


def scan_missing(key: str, data_root: Path, procs: int, repo_root: Path) -> tuple[int, str]:
    """
    Quét nốt PDF chưa có checkpoint bằng `scripts/<key>/extract.py --no-ocr` (CPU, không GPU), `procs` tiến trình chia việc bằng khoá flock.

    Dùng đúng mã trích text và bộ dò chữ hỏng của lượt thật nên số trang cần OCR là số thật. Checkpoint ghi ra còn nguyên giá trị: trang có lớp chữ tốt được giữ,
    file còn trang `needs_ocr` được lượt OCR thật làm lại. Không chạy khi lượt GPU của cùng nguồn đang chạy (hai bên tranh file).

    Returns:
        (số tiến trình thoát mã khác 0, 15 dòng cuối log của tiến trình đầu tiên thoát mã khác 0 hoặc "" nếu không có). Mã 1 có thể chỉ là có PDF lỗi; mã khác
        1 hoặc không ghi được checkpoint nào là tiến trình chết vì lỗi chung, và đoạn log cho biết lỗi gì.
    """
    script = repo_root / "scripts" / key / "extract.py"
    log_dir = data_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    procs_list = []
    for i in range(procs):
        log_path = log_dir / f"{key}_count_cpu{i}.log"
        log = open(log_path, "w", encoding="utf-8")  # noqa: SIM115 — phải sống đến khi tiến trình con xong
        cmd = [sys.executable, str(script), "--data-root", str(data_root), "--no-ocr", "--no-export", "--worker-name", f"count{i}"]
        procs_list.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log, log_path))
    bad, tail = 0, ""
    for p, log, log_path in procs_list:
        code = p.wait()
        log.close()
        if code != 0:
            bad += 1
            if not tail:
                tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:])
    return bad, tail
