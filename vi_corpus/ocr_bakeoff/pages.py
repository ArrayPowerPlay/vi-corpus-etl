"""
Chọn và chuẩn bị bộ trang cho đợt so sánh engine OCR (F-11).

Bộ A (có đáp án tự động): trang PDF giáo trình có lớp chữ Unicode hợp lệ. Đáp án = lớp chữ, sắp theo thứ tự đọc từ vị trí
các khối chữ (trang hai cột: hết cột trái rồi cột phải, khối vắt qua khe giữa như tiêu đề / chú thích tách trang thành
các dải). Trang được render thành ảnh cao bằng ảnh trang stbook (trung vị bộ B) rồi làm xấu nhẹ: xoay ±1°, mờ và / hoặc
nhiễu, lưu JPEG chất lượng 60-75 (tham số lưu theo trang). Phân tầng (cờ theo trang, một trang có thể nhiều cờ):
    two_column : có khe dọc rỗng (cho phép vài khối vắt qua) ở 30-70% bề ngang, mỗi bên >= 2 khối chữ dài và >= MIN_SIDE_CHARS ký tự, hai cột chồng
                 nhau theo chiều dọc; trang có bảng không tính là hai cột.
    table      : PyMuPDF find_tables thấy bảng >= 2 hàng x 2 cột.
    special    : đáp án có ký tự trong metrics.SPECIAL_CHARS (² ³ “ ” ‘ ’ – — … °).
Trang hợp lệ (gt_problem trả None): đủ chữ, không có font TCVN3, tỷ lệ chữ có dấu tiếng Việt bình thường, độ dài từ trung
bình hợp lý (lớp chữ mất dấu cách thì loại), không có ký tự thay thế / vùng riêng, không có chữ xoay, ảnh chiếm ít diện
tích (ảnh có chữ bên trong sẽ bị engine đọc mà đáp án không có).

Bộ B (ảnh quét thật, không có đáp án): ảnh gốc nhúng trong PDF stbook (không render lại), từ >= 10 sách, gồm các sách đã
rà ở F-06 (1248 hai cột, 1271 bảng / km², 1241, 1425); có thể ép trang cụ thể ("1248:150", số trang tính từ 1).

Kết quả trong <out>/: pages/<page_id>.jpg|png, gt/<page_id>.txt, pages.jsonl (mỗi trang một dòng) và selection.json (số
trang theo tầng, thiếu hụt, tổng số trang cần OCR của kho để quy ra giờ GPU, tham số chọn).
"""

import io
import json
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
import pymupdf
from PIL import Image, ImageFilter

from vi_corpus.common.pdf_text import is_tcvn3
from vi_corpus.ocr_bakeoff.metrics import SPECIAL_CHARS, normalize
from vi_corpus.pipeline.language import vi_diacritic_ratio

logger = logging.getLogger("vi_corpus")

REQUIRED_BOOKS = ("1248", "1271", "1241", "1425")  # sách đã rà ở F-06
DEFAULT_FORCED_B = ("1248:150",)  # trang hai cột song ngữ thấy ở F-02
QUOTAS = {"two_column": 40, "table": 30, "special": 30}
MIN_PAGE_CHARS = 400  # ký tự không phải khoảng trắng tối thiểu của một trang bộ A
DIACRITIC_RANGE = (0.12, 0.5)  # tỷ lệ chữ có dấu / chữ cái của tiếng Việt in bình thường (~0,2-0,3)
MAX_MEAN_WORD_LEN = 7.5  # lớn hơn: lớp chữ mất dấu cách
MAX_IMAGE_AREA = 0.3  # ảnh chiếm hơn tỷ lệ này diện tích trang thì bỏ trang
GUTTER_ZONE = (0.3, 0.7)  # khe giữa hai cột phải nằm trong khoảng bề ngang này
GUTTER_MIN = 0.015  # bề rộng khe tối thiểu (tỷ lệ bề ngang trang)
WIDE_BLOCK = 0.5  # khối rộng hơn nửa trang không thể là khối của một cột: không tính vào biểu đồ phủ
SPAN_SHARE = 0.1  # khe được phép bị tối đa chừng này phần số khối hẹp vắt qua (tiêu đề ngắn căn giữa, chú thích)
MIN_SIDE_CHARS = 150
LONG_LINE_SHARE = 0.6  # khối mỗi bên phải rộng trung bình >= tỷ lệ này bề ngang cột (dòng chữ dài, không phải bảng)
_TCVN3_FONT = re.compile(r"^\.?Vn[A-Z]")  # như vi_corpus.common.pdf_text
_BAD_CHARS = re.compile("[�-]")


@dataclass
class Block:
    """Một khối chữ của trang PDF: toạ độ và văn bản (các dòng nối bằng xuống dòng)."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def cx(self) -> float:
        """Tâm ngang."""
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        """Tâm dọc."""
        return (self.y0 + self.y1) / 2

    @property
    def width(self) -> float:
        """Bề rộng."""
        return self.x1 - self.x0


def text_blocks(page: "pymupdf.Page") -> tuple[list[Block], dict]:
    """
    Các khối chữ của trang và thông tin để kiểm hợp lệ.

    Returns:
        (khối, {"rotated": có dòng chữ xoay, "tcvn3_font": có span font TCVN3}).
    """
    info = {"rotated": False, "tcvn3_font": False}
    blocks = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        lines = []
        for ln in b.get("lines", []):
            if abs(ln["dir"][1]) > 0.01:
                info["rotated"] = True
            info["tcvn3_font"] |= any(_TCVN3_FONT.match(s["font"].split("+")[-1]) for s in ln["spans"])
            lines.append("".join(s["text"] for s in ln["spans"]))
        text = "\n".join(lines).strip()
        if text:
            blocks.append(Block(*b["bbox"], text))
    return blocks, info


def find_gutter(blocks: list[Block], width: float) -> float | None:
    """
    Toạ độ x của khe giữa hai cột, hoặc None nếu trang không phải hai cột chữ (xem two_column ở docstring module).
    """
    narrow = [b for b in blocks if b.width < WIDE_BLOCK * width]
    if len(narrow) < 4:
        return None
    cov = np.zeros(int(width) + 2, dtype=int)
    for b in narrow:
        cov[max(int(b.x0), 0): int(b.x1) + 1] += 1
    allowed = int(SPAN_SHARE * len(narrow))
    lo, hi = int(GUTTER_ZONE[0] * width), int(GUTTER_ZONE[1] * width)
    best, start = (0, 0, 0), None
    for x in range(lo, hi + 1):
        if cov[x] <= allowed:
            start = x if start is None else start
            if x - start + 1 > best[0]:
                best = (x - start + 1, start, x)
        else:
            start = None
    if best[0] < GUTTER_MIN * width:
        return None
    g = (best[1] + best[2]) / 2
    left = [b for b in narrow if b.x1 <= g]
    right = [b for b in narrow if b.x0 >= g]
    if len(left) < 2 or len(right) < 2:
        return None
    if sum(len(b.text) for b in left) < MIN_SIDE_CHARS or sum(len(b.text) for b in right) < MIN_SIDE_CHARS:
        return None
    if min(max(b.y1 for b in left), max(b.y1 for b in right)) <= max(min(b.y0 for b in left), min(b.y0 for b in right)):
        return None  # hai bên không chồng nhau theo chiều dọc
    lw = g - min(b.x0 for b in left)
    rw = max(b.x1 for b in right) - g
    if (sum(b.width for b in left) / len(left) < LONG_LINE_SHARE * lw
            or sum(b.width for b in right) / len(right) < LONG_LINE_SHARE * rw):
        return None  # ô ngắn (bảng, danh sách), không phải dòng chữ dài
    return g


def reading_order(blocks: list[Block], gutter: float | None) -> list[Block]:
    """
    Sắp khối theo thứ tự đọc. Không có khe: trên xuống, trái sang phải. Có khe: khối vắt qua khe chia trang thành các
    dải; trong mỗi dải đọc hết cột trái rồi cột phải.
    """
    if gutter is None:
        return sorted(blocks, key=lambda b: (round(b.y0), b.x0))
    spans = sorted((b for b in blocks if b.x0 < gutter < b.x1), key=lambda b: b.y0)
    bands: dict[int, list[Block]] = defaultdict(list)
    for b in blocks:
        if not (b.x0 < gutter < b.x1):
            bands[sum(s.y0 <= b.cy for s in spans)].append(b)
    out: list[Block] = []
    for i in range(len(spans) + 1):
        grp = bands.get(i, [])
        out += sorted((b for b in grp if b.cx < gutter), key=lambda b: b.y0)
        out += sorted((b for b in grp if b.cx >= gutter), key=lambda b: b.y0)
        if i < len(spans):
            out.append(spans[i])
    return out


def has_table(page: "pymupdf.Page") -> bool:
    """Trang có bảng >= 2 hàng x 2 cột (PyMuPDF find_tables); lỗi khi dò thì coi như không có."""
    try:
        return any(t.row_count >= 2 and t.col_count >= 2 for t in page.find_tables().tables)
    except Exception:  # noqa: BLE001 - find_tables lỗi với vài PDF lạ, không đáng dừng cả lượt chọn
        return False


def image_area_share(page: "pymupdf.Page") -> float:
    """Tỷ lệ diện tích trang bị ảnh che (cộng dồn, tối đa 1)."""
    area = abs(page.rect)
    if not area:
        return 0.0
    total = sum(abs(pymupdf.Rect(info["bbox"]) & page.rect) for info in page.get_image_info())
    return min(1.0, total / area)


def gt_problem(text: str, info: dict) -> str | None:
    """Lý do trang KHÔNG dùng làm đáp án được, hoặc None nếu hợp lệ (xem docstring module)."""
    if info.get("rotated"):
        return "rotated"
    if info.get("tcvn3_font") or is_tcvn3(text):
        return "tcvn3"
    nonspace = sum(not c.isspace() for c in text)
    if nonspace < MIN_PAGE_CHARS:
        return "few_chars"
    if _BAD_CHARS.search(text):
        return "bad_chars"
    ratio = vi_diacritic_ratio(text)
    if not DIACRITIC_RANGE[0] <= ratio <= DIACRITIC_RANGE[1]:
        return "diacritics"
    ws = text.split()
    if sum(len(w) for w in ws) / max(len(ws), 1) > MAX_MEAN_WORD_LEN:
        return "missing_spaces"
    return None


def analyze_page(page: "pymupdf.Page") -> tuple[str | None, str, dict]:
    """
    Đáp án và cờ phân tầng của một trang PDF có lớp chữ.

    Returns:
        (lý do loại hoặc None, đáp án theo thứ tự đọc, cờ {"two_column", "table", "special"}).
    """
    blocks, info = text_blocks(page)
    raw = "\n".join(b.text for b in blocks)
    problem = gt_problem(raw, info)
    if problem is None and image_area_share(page) > MAX_IMAGE_AREA:
        problem = "images"
    if problem:
        return problem, "", {}
    table = has_table(page)
    gutter = None if table else find_gutter(blocks, page.rect.width)
    gt = "\n".join(b.text for b in reading_order(blocks, gutter))
    return None, gt, {"two_column": gutter is not None, "table": table,
                      "special": any(c in gt for c in SPECIAL_CHARS)}


def render_page(page: "pymupdf.Page", target_height: int) -> Image.Image:
    """Render trang PDF thành ảnh RGB cao target_height px (giữ tỷ lệ)."""
    zoom = target_height / page.rect.height
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def degrade(img: Image.Image, rng: random.Random) -> tuple[bytes, dict]:
    """
    Làm xấu nhẹ ảnh render cho giống ảnh quét: xoay ±1°, mờ và / hoặc nhiễu Gauss, JPEG chất lượng 60-75.

    Returns:
        (byte JPEG, tham số đã dùng).
    """
    params: dict = {"angle": round(rng.uniform(-1, 1), 3), "mode": rng.choice(["blur", "noise", "both"]),
                    "jpeg_quality": rng.randint(60, 75)}
    img = img.rotate(params["angle"], resample=Image.BICUBIC, expand=False, fillcolor="white")
    if params["mode"] in ("blur", "both"):
        params["blur_radius"] = round(rng.uniform(0.5, 1.0), 3)
        img = img.filter(ImageFilter.GaussianBlur(params["blur_radius"]))
    if params["mode"] in ("noise", "both"):
        params["noise_sigma"] = round(rng.uniform(3, 8), 3)
        arr = np.asarray(img, dtype=np.float32)
        arr = arr + np.random.default_rng(rng.randrange(2**32)).normal(0, params["noise_sigma"], arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=params["jpeg_quality"])
    return buf.getvalue(), params


def page_image_bytes(page: "pymupdf.Page") -> tuple[bytes, str]:
    """
    Ảnh gốc của trang PDF quét (như vi_corpus.common.ocr.page_image nhưng giữ nguyên byte, không nén lại).

    Returns:
        (byte ảnh, phần mở rộng "jpg" / "png" / ...). Trang không đúng một ảnh thì render 200 dpi PNG.
    """
    images = page.get_images()
    if len(images) == 1:
        info = page.parent.extract_image(images[0][0])
        ext = info.get("ext", "png")
        return info["image"], "jpg" if ext in ("jpeg", "jpg") else ext
    return page.get_pixmap(dpi=200).tobytes("png"), "png"


def parse_forced(items: list[str]) -> dict[str, list[int]]:
    """["1248:150,151", "1271:20"] -> {"1248": [150, 151], "1271": [20]} (số trang tính từ 1)."""
    out: dict[str, list[int]] = defaultdict(list)
    for it in items:
        pid, _, pages = it.partition(":")
        out[pid.strip()] += [int(p) for p in pages.split(",") if p.strip()]
    return dict(out)


def select_set_b(books: list[tuple[str, dict, Path]], out: Path, n_pages: int, min_books: int,
                 forced: dict[str, list[int]], rng: random.Random) -> list[dict]:
    """
    Chọn bộ B: các sách REQUIRED_BOOKS (nếu có) + sách ngẫu nhiên cho đủ min_books, trang ngẫu nhiên ở 5-95% cuốn sách
    (bỏ bìa, mục lục cuối), cộng các trang ép. Ghi ảnh gốc vào out/pages/.

    Returns:
        Bản ghi trang (page_id, set "B", image, gt None, source...).
    """
    by_id = {str(b["product_id"]): (slug, b, pdf) for slug, b, pdf in books}
    chosen = [pid for pid in REQUIRED_BOOKS if pid in by_id]
    rest = [pid for pid in by_id if pid not in chosen]
    rng.shuffle(rest)
    chosen += rest[: max(0, min_books - len(chosen))]
    chosen += [pid for pid in forced if pid in by_id and pid not in chosen]
    for pid, wanted in forced.items():
        if pid not in by_id:
            logger.warning("Bộ B: không thấy sách %s để lấy trang ép %s", pid, wanted)
    n_forced = sum(len(forced.get(pid, [])) for pid in chosen)
    remaining = max(0, n_pages - n_forced)
    k = max(len(chosen), 1)
    records = []
    for j, pid in enumerate(chosen):
        per_book = remaining // k + (j < remaining % k)  # chia đều phần còn lại, sách đầu danh sách nhận phần dư
        slug, book, pdf = by_id[pid]
        with pymupdf.open(pdf) as doc:
            n = doc.page_count
            forced_idx = [p - 1 for p in forced.get(pid, []) if 1 <= p <= n]
            pool = [i for i in range(int(0.05 * n), max(int(0.95 * n), 1)) if i not in forced_idx]
            for i in forced_idx + sorted(rng.sample(pool, min(per_book, len(pool)))):
                data, ext = page_image_bytes(doc[i])
                page_id = f"B{len(records) + 1:03d}"
                path = out / "pages" / f"{page_id}.{ext}"
                path.write_bytes(data)
                w, h = Image.open(io.BytesIO(data)).size
                records.append({"page_id": page_id, "set": "B", "image": f"pages/{path.name}", "gt": None,
                                "flags": {}, "size": [w, h],
                                "source": {"kind": "stbook", "slug": slug, "product_id": pid,
                                           "title": book.get("title") or book.get("name"), "page": i + 1}})
    return records


def select_set_a(pdfs: list[Path], root: Path, out: Path, n_pages: int, target_height: int, rng: random.Random,
                 max_files: int, scan_per_doc: int, max_per_doc: int, quotas: dict[str, int]) -> tuple[list[dict], dict]:
    """
    Chọn bộ A từ các PDF giáo trình (đã xáo theo rng): mỗi file thử tối đa scan_per_doc trang ngẫu nhiên, nhận tối đa
    max_per_doc trang; nhận trang nếu nó giúp một tầng chưa đủ chỉ tiêu, hoặc còn chỗ cho trang thường sau khi chừa chỗ
    cho các tầng còn thiếu. Dừng khi đủ n_pages và đủ chỉ tiêu, hoặc đã xét max_files file.

    Returns:
        (bản ghi trang, thống kê {"files_scanned", "pages_scanned", "rejected": {lý do: số trang}}).
    """
    records: list[dict] = []
    counts = dict.fromkeys(quotas, 0)
    stats: dict = {"files_scanned": 0, "pages_scanned": 0, "rejected": defaultdict(int)}

    def need() -> int:
        """Số trang còn phải chừa cho các tầng chưa đủ chỉ tiêu."""
        return sum(max(0, q - counts[k]) for k, q in quotas.items())

    for pdf in pdfs[:max_files]:
        if len(records) >= n_pages and need() == 0:
            break
        stats["files_scanned"] += 1
        try:
            doc = pymupdf.open(pdf)
        except Exception as e:  # noqa: BLE001 - file hỏng / không phải PDF: bỏ qua, ghi lý do
            stats["rejected"][f"open:{type(e).__name__}"] += 1
            continue
        with doc:
            if doc.is_repaired or doc.needs_pass:
                stats["rejected"]["broken_or_password"] += 1
                continue
            taken = 0
            for i in rng.sample(range(doc.page_count), min(scan_per_doc, doc.page_count)):
                if taken >= max_per_doc or (len(records) >= n_pages and need() == 0):
                    break
                stats["pages_scanned"] += 1
                page = doc[i]
                problem, gt, flags = analyze_page(page)
                if problem:
                    stats["rejected"][problem] += 1
                    continue
                helps = any(flags.get(k) and counts[k] < q for k, q in quotas.items())
                if not helps and len(records) >= n_pages - need():
                    stats["rejected"]["quota_full"] += 1
                    continue
                for k in quotas:
                    counts[k] += bool(flags.get(k))
                pid = f"A{len(records) + 1:04d}"
                data, params = degrade(render_page(page, target_height), rng)
                (out / "pages" / f"{pid}.jpg").write_bytes(data)
                (out / "gt" / f"{pid}.txt").write_text(gt, encoding="utf-8")
                w, h = Image.open(io.BytesIO(data)).size
                rel = str(pdf.relative_to(root)) if pdf.is_relative_to(root) else str(pdf)
                records.append({"page_id": pid, "set": "A", "image": f"pages/{pid}.jpg", "gt": f"gt/{pid}.txt",
                                "flags": flags, "size": [w, h], "degrade": params, "gt_chars": len(normalize(gt)),
                                "source": {"kind": "giao_trinh", "file": rel, "page": i + 1}})
                taken += 1
    stats["rejected"] = dict(stats["rejected"])
    stats["strata"] = counts
    stats["shortfall"] = {k: q - counts[k] for k, q in quotas.items() if counts[k] < q}
    return records, stats


def corpus_pages(data_root: Path, books: list[tuple[str, dict, Path]]) -> dict:
    """
    Số trang cần OCR của kho (để quy ra giờ GPU, R-08): mọi trang sách stbook; trang giáo trình có method "ocr" /
    "needs_ocr" trong checkpoint interim/giao_trinh_text/ (chỉ đếm được nếu đã chạy trích text).
    """
    stbook = 0
    for _, _, pdf in books:
        try:
            with pymupdf.open(pdf) as doc:
                stbook += doc.page_count
        except Exception as e:  # noqa: BLE001 - PDF hỏng không làm sai lệch đáng kể tổng số trang
            logger.warning("Không mở được %s để đếm trang: %s", pdf, e)
    gt_files = sorted((data_root / "interim/giao_trinh_text").glob("*.json"))
    giao_trinh = 0
    for f in gt_files:
        try:
            pages = json.loads(f.read_text(encoding="utf-8")).get("pages") or []
        except (OSError, json.JSONDecodeError):
            continue
        giao_trinh += sum(p.get("method") in ("ocr", "needs_ocr") for p in pages)
    return {"stbook": stbook, "giao_trinh_ocr": giao_trinh, "giao_trinh_checkpoints": len(gt_files),
            "note": "giao_trinh_ocr chỉ đếm file đã có checkpoint trích text; VJOL / VISTA chưa có"}


def prepare_pages(data_root: Path, out: Path, n_a: int = 300, n_b: int = 40, min_books: int = 10,
                  forced_b: tuple[str, ...] = DEFAULT_FORCED_B, seed: int = 42, target_height: int | None = None,
                  max_files: int = 2000, scan_per_doc: int = 8, max_per_doc: int = 3,
                  stbook_root: Path | None = None, giao_trinh_root: Path | None = None,
                  quotas: dict[str, int] | None = None) -> dict:
    """
    Chọn bộ B rồi bộ A, ghi out/pages.jsonl và out/selection.json (xem docstring module).

    Args:
        target_height: Chiều cao ảnh bộ A; None = trung vị chiều cao ảnh bộ B (2000 px nếu không có sách stbook).
        quotas:        Chỉ tiêu tối thiểu theo tầng (mặc định QUOTAS); được ưu tiên hơn n_a.

    Returns:
        Nội dung selection.json.
    """
    from vi_corpus.stbook.ocr_books import find_books

    rng = random.Random(seed)
    (out / "pages").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(parents=True, exist_ok=True)
    stbook_root = stbook_root or data_root / "raw/stbook"
    books = list(find_books(stbook_root)) if stbook_root.exists() else []
    set_b = select_set_b(books, out, n_b, min_books, parse_forced(list(forced_b)), rng) if books else []
    if not books:
        logger.warning("Không thấy sách stbook ở %s: bộ B rỗng", stbook_root)
    height = target_height or (int(median(r["size"][1] for r in set_b)) if set_b else 2000)
    gt_root = giao_trinh_root or data_root / "raw/giao_trinh"
    pdfs = sorted(p for p in gt_root.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf") if gt_root.exists() else []
    rng.shuffle(pdfs)
    quotas = QUOTAS if quotas is None else quotas
    set_a, stats = select_set_a(pdfs, gt_root, out, n_a, height, rng, max_files, scan_per_doc, max_per_doc, quotas)
    if not set_a:
        logger.warning("Bộ A rỗng: %d PDF giáo trình ở %s, không trang nào có lớp chữ dùng được (lý do loại: %s)",
                       len(pdfs), gt_root, dict(stats["rejected"]))
    with open(out / "pages.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in set_a + set_b)
    selection = {"seed": seed, "set_a": len(set_a), "set_b": len(set_b), "target_height": height,
                 "books_b": sorted({r["source"]["product_id"] for r in set_b}), "pdfs_found": len(pdfs),
                 "quotas": quotas, **stats, "corpus_pages": corpus_pages(data_root, books)}
    (out / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    return selection


def read_pages(bakeoff_dir: Path) -> list[dict]:
    """Đọc pages.jsonl của một thư mục so sánh."""
    with open(bakeoff_dir / "pages.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
