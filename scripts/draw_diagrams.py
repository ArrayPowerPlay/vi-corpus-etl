"""
Vẽ lại các sơ đồ luồng của pipeline (docs/diagrams/00..08, 22, 23) từ một khung bố cục nhỏ viết sẵn trong file này.

Vì sao có file này: các sơ đồ SVG cũ vẽ tay theo code ở một commit cũ nên lệch dần khi pipeline đổi (fastText, token Qwen3,
dedup 2 vòng, VJOL / VISTA, chia phần theo shard...). Muốn sơ đồ đúng với code thì sửa nội dung trong hàm `diagram_*`
bên dưới rồi chạy lại file này, không sửa SVG bằng tay.

Cách dùng (từ gốc repo):
    uv run python scripts/draw_diagrams.py                 # ghi docs/diagrams/*.svg
    uv run python scripts/draw_diagrams.py --only 00,05    # chỉ vẽ một số sơ đồ (theo số đầu tên file)
    uv run python scripts/draw_diagrams.py --png           # thêm bản PNG (cần PyMuPDF, nhóm ocr hoặc đã cài sẵn); 00_pipeline.png

Khung bố cục: `Diagram.box` đặt một khối ở toạ độ tuyệt đối (chiều cao tự tính theo số dòng chữ sau khi tự xuống dòng),
`Diagram.link` nối hai khối bằng mũi tên (thẳng hoặc gấp khúc), `Diagram.save` ghi SVG kèm chú giải màu. Màu theo loại khối:
dữ liệu vào, bước xử lý, rẽ nhánh / điều kiện, loại / dừng, đầu ra, chưa có code, chạy trên GPU.
"""

import argparse
import html
import textwrap
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "diagrams"
FONT = "DejaVu Sans, Arial, sans-serif"
MONO = "DejaVu Sans Mono, Menlo, monospace"
CHAR_W = 7.0  # px mỗi ký tự ở cỡ chữ 12 (DejaVu Sans, tiếng Việt có dấu), dùng để tự xuống dòng
LINE_H = 16.5

# loại khối -> (màu nền, màu viền, nét đứt?, chú thích trong chú giải)
KINDS = {
    "in": ("#eef2f7", "#7a8aa0", False, "đầu vào / dữ liệu"),
    "step": ("#e9f5ee", "#3f9b6b", False, "bước xử lý (CPU)"),
    "gpu": ("#e8e4fb", "#6b4fd0", False, "bước chạy trên GPU"),
    "cond": ("#fff4dc", "#d19a1f", False, "rẽ nhánh / điều kiện"),
    "stop": ("#fdecea", "#c94a3f", False, "loại / dừng"),
    "out": ("#e3eefb", "#2a78d6", False, "đầu ra"),
    "plan": ("#f5f5f5", "#9a9a9a", True, "chưa có code"),
}


class Diagram:
    """Một sơ đồ: danh sách khối + mũi tên, ghi ra SVG tự chứa (nền trắng, chữ DejaVu Sans)."""

    def __init__(self, title: str, subtitle: str = "", width: int = 1000) -> None:
        """Tạo sơ đồ trống với tiêu đề (dòng 1) và dòng phụ ghi module / file code liên quan (dòng 2)."""
        self.title, self.subtitle, self.width = title, subtitle, width
        self.boxes: dict[str, dict] = {}
        self.parts: list[str] = []
        self.kinds_used: set[str] = set()
        self.bottom = 70 if subtitle else 50

    def box(self, key: str, x: int, y: int, w: int, kind: str, title: str, lines=(), tag: str = "", mono: str = "",
            min_h: int = 0) -> dict:
        """
        Đặt một khối. `lines` là các dòng chữ (tự xuống dòng theo bề rộng); `tag` ghi mờ ở góc phải trên (vd "Ray");
        `mono` là dòng đường dẫn / tên file in chữ đơn cách ở cuối khối. Trả về dict toạ độ (x, y, w, h) để nối mũi tên.
        """
        fill, stroke, dashed, _ = KINDS[kind]
        self.kinds_used.add(kind)
        per_line = max(10, int((w - 24) / CHAR_W))
        wrapped: list[str] = []
        for ln in lines:
            sub = textwrap.wrap(ln, per_line, break_long_words=False, break_on_hyphens=False) or [""]
            wrapped.extend(sub)
        mono_lines = textwrap.wrap(mono, int((w - 24) / 6.6), break_long_words=True) if mono else []
        h = max(min_h, 14 + 20 + len(wrapped) * LINE_H + len(mono_lines) * 15 + (6 if wrapped or mono_lines else -4))
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        gray = kind == "plan"
        txt = "#6a6a6a" if gray else "#1b2733"
        p = [f'<rect x="{x}" y="{y}" width="{w}" height="{h:.0f}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.6"{dash}/>',
             f'<text x="{x + 12}" y="{y + 22}" font-size="14" font-weight="700" fill="{txt}">{html.escape(title)}</text>']
        if tag:
            p.append(f'<text x="{x + w - 10}" y="{y + 20}" text-anchor="end" font-family="{MONO}" font-size="11" fill="#455566">{html.escape(tag)}</text>')
        yy = y + 22 + 4
        for ln in wrapped:
            yy += LINE_H
            p.append(f'<text x="{x + 12}" y="{yy - 3:.1f}" font-size="12" fill="{txt}" xml:space="preserve">{html.escape(ln)}</text>')
        for ln in mono_lines:
            yy += 15
            p.append(f'<text x="{x + 12}" y="{yy - 3:.1f}" font-family="{MONO}" font-size="11" fill="#455566" xml:space="preserve">{html.escape(ln)}</text>')
        self.parts.append("\n".join(p))
        b = {"x": x, "y": y, "w": w, "h": h}
        self.boxes[key] = b
        self.bottom = max(self.bottom, y + h)
        return b

    def after(self, key: str, gap: int = 24) -> int:
        """Toạ độ y đặt khối kế tiếp ngay dưới khối `key`, cách `gap` px (tránh khối chồng lên nhau khi chiều cao tự tính)."""
        b = self.boxes[key]
        return int(b["y"] + b["h"] + gap)

    def _anchor(self, key: str, side: str, frac: float = 0.5) -> tuple[float, float]:
        """Điểm nối trên cạnh `side` (t, b, l, r) của khối `key`; `frac` chọn vị trí dọc cạnh (0..1)."""
        b = self.boxes[key]
        if side == "t":
            return b["x"] + b["w"] * frac, b["y"]
        if side == "b":
            return b["x"] + b["w"] * frac, b["y"] + b["h"]
        if side == "l":
            return b["x"], b["y"] + b["h"] * frac
        return b["x"] + b["w"], b["y"] + b["h"] * frac

    def link(self, a: str, b: str, label: str = "", sa: str = "b", sb: str = "t", fa: float | None = None, fb: float | None = None,
             mid: float | None = None, dashed: bool = False) -> None:
        """
        Mũi tên từ cạnh `sa` của khối `a` tới cạnh `sb` của khối `b`. Hai đầu lệch nhau thì gấp khúc (đi ngang hoặc
        đứng tại `mid`, mặc định giữa hai đầu). `label` đặt cạnh đoạn đầu tiên, nền trắng cho dễ đọc.
        """
        if fa is None and fb is None and sa in "tb" and sb in "tb":  # hai khối chồng nhau theo chiều ngang: nối thẳng đứng giữa phần chung
            ba, bb = self.boxes[a], self.boxes[b]
            lo, hi = max(ba["x"], bb["x"]), min(ba["x"] + ba["w"], bb["x"] + bb["w"])
            if hi > lo:
                mid_x = (lo + hi) / 2
                fa, fb = (mid_x - ba["x"]) / ba["w"], (mid_x - bb["x"]) / bb["w"]
        fa = 0.5 if fa is None else fa
        fb = 0.5 if fb is None else fb
        x1, y1 = self._anchor(a, sa, fa)
        x2, y2 = self._anchor(b, sb, fb)
        pts = [(x1, y1)]
        vertical = sa in "tb"
        if vertical != (sb in "tb"):  # một đầu đứng, một đầu ngang: gấp khúc một góc vuông
            pts.append((x1, y2) if vertical else (x2, y1))
        elif vertical and abs(x1 - x2) > 1:
            my = (y1 + y2) / 2 if mid is None else mid
            pts += [(x1, my), (x2, my)]
        elif not vertical and abs(y1 - y2) > 1:
            mx = (x1 + x2) / 2 if mid is None else mid
            pts += [(mx, y1), (mx, y2)]
        pts.append((x2, y2))
        d = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.parts.append(f'<polyline points="{d}" fill="none" stroke="#455566" stroke-width="1.6"{dash} marker-end="url(#ah)"/>')
        if label:
            (sx, sy), (ex, ey) = pts[0], pts[1]
            lx, ly = (sx + ex) / 2, (sy + ey) / 2
            if vertical:
                lx, anchor = sx + 6, "start"
            else:
                anchor, ly = "middle", sy - 6
            width = len(label) * 6.4 + 6
            bx = lx - (0 if anchor == "start" else width / 2)
            self.parts.append(f'<rect x="{bx - 2:.1f}" y="{ly - 11:.1f}" width="{width:.1f}" height="15" fill="#ffffff" opacity="0.9"/>')
            self.parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" font-size="11.5" fill="#455566">{html.escape(label)}</text>')

    def note(self, x: int, y: int, text: str, size: float = 12, anchor: str = "start", color: str = "#455566") -> None:
        """Dòng chữ tự do (không khung) tại toạ độ (x, y)."""
        self.parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{color}">{html.escape(text)}</text>')
        self.bottom = max(self.bottom, y + 6)

    def save(self, path: Path) -> None:
        """Ghi SVG: tiêu đề, các khối / mũi tên đã thêm và chú giải màu (chỉ các loại đã dùng) ở dưới cùng."""
        legend_y = int(self.bottom + 28)
        legend, x = [], 30
        for kind in KINDS:
            if kind not in self.kinds_used:
                continue
            fill, stroke, dashed, label = KINDS[kind]
            dash = ' stroke-dasharray="4 3"' if dashed else ""
            legend.append(f'<rect x="{x}" y="{legend_y}" width="18" height="14" rx="3" fill="{fill}" stroke="{stroke}"{dash}/>'
                          f'<text x="{x + 24}" y="{legend_y + 11}" font-size="11.5" fill="#455566">{html.escape(label)}</text>')
            x += 24 + int(len(label) * 6.3) + 26
        height = legend_y + 36
        head = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {height}" width="{self.width}" height="{height}" font-family="{FONT}">',
                f"<title>{html.escape(self.title)}</title>",
                '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                '<path d="M0,0 L10,5 L0,10 z" fill="#455566"/></marker></defs>',
                f'<rect width="{self.width}" height="{height}" fill="#ffffff"/>',
                f'<text x="{self.width / 2}" y="30" text-anchor="middle" font-size="19" font-weight="700" fill="#1b2733">{html.escape(self.title)}</text>']
        if self.subtitle:
            head.append(f'<text x="{self.width / 2}" y="50" text-anchor="middle" font-family="{MONO}" font-size="11" fill="#6a7a8a">{html.escape(self.subtitle)}</text>')
        path.write_text("\n".join(head + self.parts + legend + ["</svg>"]) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------------------
# 00: luồng chính
# --------------------------------------------------------------------------------------------------
def diagram_00() -> Diagram:
    """Sơ đồ tổng: 5 nguồn có code -> raw / interim -> 8 stage của run_pipeline.py -> kết quả, dedup vòng 2, cpt_pack."""
    d = Diagram("vi-corpus-etl: luồng xử lý dữ liệu", "cập nhật 2026-11-19 theo code ở nhánh worktree-vjol-pipeline")
    names = [("SEA (Hugging Face)", "Instruct-2602, PILE v2, PILE v1; chỉ phần tiếng Việt"),
             ("stbook.vn", "sách PDF toàn ảnh (mỗi trang 1 JPEG)"),
             ("giao_trinh", "giáo trình đại học 16 ngành, Google Drive: PDF, pptx, doc, djvu"),
             ("VJOL", "bài báo khoa học (PDF) + metadata tạp chí; chỉ xử lý, không crawl"),
             ("VISTA", "bài báo khoa học (PDF) + metadata; chỉ xử lý, không crawl")]
    steps = [("Tải về", "scripts/sea/*.py, checkpoint theo file", "raw/sea_vi/<bộ>/"),
             ("Crawl + OCR chuỗi VLM", "Qwen3-VL-8B → rp110 → Paddle + VietOCR → bỏ trang", "interim/stbook_ocr/"),
             ("Trích text", "rclone + extract.py: PyMuPDF, TCVN3, parser docx / ppt / djvu, OCR", "interim/giao_trinh_text/"),
             ("Trích text 2 lượt", "PyMuPDF mọi trang; trang scan + trang lỗi font → OCR VLM", "interim/vjol_text/"),
             ("Trích text (chung bộ VJOL)", "cùng bộ xử lý bài báo; luật producer OCR đời cũ", "interim/vista_text/")]
    for i, (n, sub) in enumerate(names):
        d.box(f"s{i}", 30 + i * 190, 62, 180, "in", n, [sub], min_h=96)
    for i, (n, sub, path) in enumerate(steps):
        d.box(f"p{i}", 30 + i * 190, 190, 180, "step", n, [sub], mono=path, min_h=132)
        d.link(f"s{i}", f"p{i}")
    d.box("raw", 30, 342, 640, "in", "data/raw/ (giữ nguyên, không sửa)  +  data/interim/ (kết quả OCR / trích text)",
          ["Source registry (vi_corpus/common/registry.py): owner, license, domain, đường dẫn, định dạng; rights_status mặc định unknown "
           "→ quarantine, hiện chỉ gắn nhãn (R-01)"])
    d.box("future", 690, 342, 280, "plan", "Nguồn đã chọn, chưa có code ingest",
          ["phap_luat, anle (đã tải), wiki_vi, fineweb2_vi, finepdfs_edu_vi, finepdfs_vi (cần lọc chủ đề trước), luanvan_vnu"])
    for i in range(5):
        d.link(f"p{i}", "raw", sa="b", sb="t", fb=0.1 + 0.2 * i)
    d.box("run", 30, 462, 940, "in", "scripts/run_pipeline.py",
          ["--total / --mix (lấy mẫu) hoặc --take-all; --shard I/N (chỉ VJOL / VISTA, cả kho chia phần); --until, --force-from, --keep-stale; "
           "--executor / --num-gpus (NeMo Curator + Ray). Mỗi stage ghi 1 file Parquet + vân tay; chạy lại thì bỏ qua stage khớp vân tay."])
    d.link("raw", "run")
    stages = [
        ("ingest", "1. ingest", "step", "SEA: lấy mẫu ngẫu nhiên theo seed. Sách / bài báo: nguyên cuốn (đã làm sạch theo trang, VJOL / VISTA bỏ bìa, mục lục, tài liệu tham khảo, nối metadata). Về schema chung, giữ source_path + sha256.", "01_ingest.parquet", ""),
        ("prepare", "2. prepare", "step", "NFC, bỏ ký tự ẩn, ƣ → ư, font Symbol; xoá dòng lặp; cắt đoạn ~1.024 token (tối đa 2.048) bằng tokenizer Qwen3; đếm token.", "02_prepare.parquet", ""),
        ("language", "3. language", "step", "fastText lid.176 theo từng đoạn văn → language, lang_score, lang_mix; bỏ công thức / code trước khi nhận diện.", "03_language.parquet", "Ray"),
        ("quality", "4. quality", "step", "Số đo, điểm 0-100, band A/B/C/D, reason code. Lỗi cứng gồm chữ hỏng do font (VJOL / VISTA); perplexity Qwen3-0.6B tuỳ chọn.", "04_quality.parquet", "Ray"),
        ("dedup", "5. dedup (vòng 1)", "step", "Cổng quyền (chỉ gắn nhãn), exact + MinHash cả văn bản + L1 bao hàm + exact theo đoạn, gán dedup_family_id; ghi kho dấu vân tay.", "05_dedup.parquet", ""),
        ("embed", "6. embed (tuỳ chọn)", "gpu", "multilingual-e5-base (hoặc tfidf trên CPU). Bản chạy cả kho tắt (--embedder none).", "06_embeddings.parquet", "Ray, GPU"),
        ("reduce", "7. reduce (tuỳ chọn)", "step", "HDBSCAN trên embedding gốc; PCA + UMAP 2D chỉ để vẽ bản đồ.", "07_reduced.parquet", ""),
        ("final", "8. finalize", "step", "Áp verdict dedup vòng 2 (nếu có), clean, knowledge unit, segment CPT, quét nhiễm benchmark, audit, bản đồ, report.", "clean.parquet, ...", ""),
    ]
    y = 560
    prev = "run"
    for key, title, kind, text, out, tag in stages:
        d.box(key, 30, y, 620, kind, title, [text], tag=tag, mono=out)
        d.link(prev, key)
        prev = key
        y = d.boxes[key]["y"] + d.boxes[key]["h"] + 24
    r = d.boxes
    d.box("global", 680, r["dedup"]["y"], 290, "step", "Dedup vòng 2", ["scripts/dedup_global.py: gộp kho dấu vân tay (state/dedup_index) của mọi nguồn, họ trùng toàn cục, chỉ đánh dấu."],
          mono="processed/dedup_global/verdict.parquet")
    d.link("dedup", "global", sa="r", sb="l")
    d.link("global", "final", "--global-verdict", sa="b", sb="r", fb=0.25)
    d.box("outs", 680, r["final"]["y"] + 40, 290, "out", "Kết quả của một lần chạy",
          ["clean.parquet (status = kept), knowledge_units.parquet (CPT / SFT, split_bucket theo họ trùng), cpt_segments.parquet, audit.json, manifest.json, report.html"])
    d.link("final", "outs", sa="r", sb="l", fa=0.75)
    d.box("pack", 680, r["outs"]["y"] + r["outs"]["h"] + 24, 290, "step", "scripts/cpt_pack.py",
          ["Cắt segment theo ngữ cảnh 8.192 token, ghép best-fit, EOS chỉ ở cuối văn bản thật (T-07..T-09)."])
    d.link("outs", "pack")
    d.box("seq", 680, r["pack"]["y"] + r["pack"]["h"] + 24, 290, "out", "Chuỗi huấn luyện CPT", ["(chưa có bước trộn tỉ lệ nguồn cpt_mix)"])
    d.link("pack", "seq")
    d.box("par", 680, r["language"]["y"], 290, "in", "Song song", ["NeMo Curator + Ray cho language, quality, embed, OCR (sơ đồ 09); VJOL / VISTA cả kho: 128 phần song song (sơ đồ 23)."])
    d.box("todo", 680, r["seq"]["y"] + r["seq"]["h"] + 24, 290, "plan", "Chưa có (docs/project/TASKS.md)",
          ["Bộ chấm chất lượng bằng mô hình (L-05), semantic dedup, cpt_mix (L-06), quét nhiễm theo từng câu (L-01)."])
    return d


# --------------------------------------------------------------------------------------------------
# 01: runner
# --------------------------------------------------------------------------------------------------
def diagram_01() -> Diagram:
    """Luồng con 1: runner, vân tay stage và checkpoint theo stage."""
    d = Diagram("Luồng con 1: runner, vân tay và checkpoint theo stage", "vi_corpus/pipeline/runner.py · scripts/run_pipeline.py")
    d.box("cli", 30, 70, 600, "in", "scripts/run_pipeline.py",
          ["--data-root, --total, --mix, --take-all, --shard I/N, --seed, --run-name", "--until, --force-from, --keep-stale, --executor, --num-gpus",
           "--tokenizer, --lang-model, --embedder, --perplexity-model, --hyphen-lm-model, --global-verdict, --no-dedup-index"])
    d.box("dir", 30, 190, 600, "step", "Chuẩn bị run_dir", ["<data-root>/processed/pipeline_runs/<tên lần chạy>/; --force-from xoá kết quả từ stage đó trở đi (kể cả clean, KU, report)."])
    d.box("fp", 30, 270, 600, "step", "Tính vân tay mong đợi của từng stage",
          ["băm(tham số stage + phiên bản code + vân tay stage trước); ingest dùng danh sách file đầu vào + kích thước. Mỗi stage chỉ băm các trường hồ sơ nguồn nó đọc, nên chỉnh ngưỡng quality chỉ chạy lại từ quality."])
    d.box("chk", 30, 380, 600, "cond", "Vân tay trong manifest khớp?", ["so từng stage đã có file kết quả"])
    d.box("stale", 670, 380, 300, "stop", "Lệch", ["xoá kết quả stage đó và mọi stage sau rồi chạy lại; --keep-stale chỉ cảnh báo và dùng kết quả cũ; run cũ chưa có vân tay: cảnh báo một lần."])
    d.link("cli", "dir"); d.link("dir", "fp"); d.link("fp", "chk"); d.link("chk", "stale", "lệch", sa="r", sb="l")
    d.box("loop", 30, 470, 600, "cond", "Với từng stage ingest → prepare → language → quality → dedup",
          ["đã có file kết quả (và vân tay khớp)?"])
    d.link("chk", "loop", "khớp")
    d.box("skip", 30, 560, 290, "step", "Đã có", ["đọc lại Parquet, bỏ qua. Chạy lại đúng lệnh cũ = làm tiếp từ stage còn thiếu."])
    d.box("run", 340, 560, 290, "step", "Chưa có: chạy hàm stage",
          ["ghi vân tay TRƯỚC, rồi ghi Parquet nguyên tử; manifest += rows, seconds, peak_rss_mb, thống kê. Toàn bộ list[dict] nằm trong RAM giữa các stage."])
    d.link("loop", "skip", "có"); d.link("loop", "run", "chưa")
    d.box("par", 670, 560, 300, "in", "CuratorBackend (tuỳ chọn)", ["language, quality, embed chạy song song khi có --executor (sơ đồ 09); không có thì chạy tuần tự trong tiến trình này."])
    d.link("run", "par", sa="r", sb="l", dashed=True)
    d.box("idx", 30, 700, 600, "step", "Sau dedup: kho dấu vân tay vòng 1 (D-09)",
          ["ghi state/dedup_index/<nguồn>/<run>.parquet cho mọi văn bản của lần chạy (bản bị loại ghi thành bia mộ); kho gắn với vân tay của 05_dedup.parquet, lệch thì ghi lại, --no-dedup-index thì xoá."])
    d.link("skip", "idx"); d.link("run", "idx")
    d.box("until", 30, 810, 600, "cond", "Đã tới stage --until?", ["chạy từng bước: until=ingest, rồi prepare, ..."])
    d.box("halt", 670, 810, 300, "stop", "Dừng", ["trả manifest của các stage đã chạy"])
    d.link("idx", "until"); d.link("until", "halt", "có", sa="r", sb="l")
    d.box("emb", 30, 900, 600, "step", "embed → reduce (nếu --embedder khác none)", ["mỗi stage một file kết quả, cùng cơ chế vân tay; --embed-scope all | kept."], tag="06, 07")
    d.link("until", "emb", "chưa")
    d.box("fin", 30, 985, 600, "step", "finalize: luôn chạy lại (rẻ, không có vân tay)", ["xoá kết quả finalize cũ rồi ghi lại; áp --global-verdict nếu có."])
    d.link("emb", "fin")
    d.box("res", 30, 1070, 600, "out", "Kết quả trong run_dir",
          ["01_ingest … 05_dedup (.parquet); 06_embeddings, 07_reduced nếu bật embed", "clean.parquet, knowledge_units.parquet, cpt_segments.parquet, audit.json, manifest.json, report.html"])
    d.link("fin", "res")
    return d


# --------------------------------------------------------------------------------------------------
# 02: ingest
# --------------------------------------------------------------------------------------------------
def diagram_02() -> Diagram:
    """Luồng con 2: ingest (nguồn text lấy mẫu, nguồn sách / bài báo nguyên cuốn, chia phần theo sha256)."""
    d = Diagram("Luồng con 2: ingest (từ raw / interim về schema chung)", "vi_corpus/pipeline/ingest.py · sea/reader.py · stbook/ocr_books.py · giao_trinh/records.py · vjol/records.py · vista/records.py")
    d.box("mix", 30, 70, 940, "in", "cfg.mix và registry", ["cfg.mix = số mẫu mỗi nguồn (--take-all: lấy hết, không lấy mẫu); registry SourceSpec = đường dẫn, định dạng, cột text, license, owner, domain."])
    d.box("ch", 30, 150, 940, "cond", "Nguồn chunked? (SourceProfile.chunked)", ["chunked = stbook, giao_trinh, vjol, vista (sách / bài dài, nguyên cuốn); còn lại là nguồn text (SEA)."])
    d.link("mix", "ch")
    d.box("txt", 30, 250, 450, "step", "sample_text_source (SEA)", ["xáo thứ tự file theo seed '{seed}:{khoá nguồn}'; lần lượt từng file, lấy k = min(rows_per_file = 250, còn thiếu) dòng."])
    d.box("fmt", 30, 360, 450, "cond", "Định dạng file", ["parquet: chọn chỉ số dòng ngẫu nhiên từ metadata, iter_batches(1000) chỉ cột text, chỉ take() dòng được chọn.",
                                                          "jsonl.gz (SEA-PILE v1): reservoir sampling một lượt, chỉ json.loads dòng được giữ."])
    d.box("rec", 30, 500, 450, "step", "row_to_record", ["→ CORPUS_SCHEMA; source_path tương đối data-root; SEA-Instruct: conversations là chuỗi repr Python → ast.literal_eval, meta = các lượt thoại."])
    d.box("bk", 510, 250, 460, "step", "iter_book_source (nguyên cuốn / bài)", ["mỗi cuốn hoặc bài một bản ghi; số mẫu cắt xuống ở stage prepare (mẫu của nguồn này là một đoạn)."])
    d.link("ch", "bk", "có")
    d.box("b1", 510, d.after("bk", 20), 460, "step", "stbook", ["interim/stbook_ocr/*.json → clean_pages (bỏ tiêu đề / chân trang chạy, số trang, ghép dòng); trang flagged là chuỗi rỗng."])
    d.box("b2", 510, d.after("b1", 20), 460, "step", "giao_trinh", ["interim/giao_trinh_text/*.json → clean_pages (slides = True cho pptx); giữ sách tiếng Anh."])
    d.box("b3", 510, d.after("b2", 20), 460, "step", "vjol, vista", ["interim/vjol_text | vista_text → bỏ bìa, mục lục, tài liệu tham khảo, tiêu đề chạy, dòng rác; khôi phục 'ư' ('ph-ơng', tuỳ chọn --hyphen-lm-model); che email / số điện thoại; nối metadata (VISTA: 1 bản ghi / PDF, meta.publication_ids)."])
    d.box("sh", 510, d.after("b3", 20), 460, "in", "--shard I/N (chỉ vjol, vista)", ["chỉ đọc phần I trong N phần chia theo sha256 PDF; nguồn khác dùng --shard thì báo lỗi."])
    d.link("bk", "b1"); d.link("b1", "b2"); d.link("b2", "b3"); d.link("b3", "sh", dashed=True)
    d.box("warn", 510, d.after("sh", 20), 460, "stop", "Cảnh báo, không dừng", ["nguồn không có file, chưa có sách nào (chưa OCR / trích text), hoặc lấy không đủ mẫu (cả hai nhánh)."])
    d.link("txt", "fmt"); d.link("fmt", "rec")
    d.link("ch", "txt", "không")
    d.box("out", 30, d.after("warn", 30), 940, "out", "01_ingest.parquet", ["manifest: số bản ghi theo nguồn. Vân tay stage = INGEST_VERSION (nâng khi đổi iter_records / row_to_record / cột meta) + danh sách file đầu vào và kích thước (kể cả metadata và tập âm tiết của VJOL / VISTA)."])
    d.link("rec", "out"); d.link("warn", "out", fb=0.75)
    return d


# --------------------------------------------------------------------------------------------------
# 03: prepare
# --------------------------------------------------------------------------------------------------
def diagram_03() -> Diagram:
    """Luồng con 3: prepare (chuẩn hoá, xoá dòng lặp, cắt đoạn theo token, đếm token)."""
    d = Diagram("Luồng con 3: prepare (chuẩn hoá, xoá dòng lặp, cắt đoạn, đếm token)", "vi_corpus/pipeline/normalize.py · lines.py · chunk.py · tokens.py")
    W = 620
    d.box("in", 30, 70, W, "in", "01_ingest.parquet", [])
    d.box("norm", 30, 130, W, "step", "normalize_text (NORMALIZER_VERSION 4)",
          ["CRLF → LF, Unicode NFC; bỏ ký tự điều khiển và vô hình (zero-width, BOM, soft hyphen); mọi khoảng trắng Unicode → dấu cách; gộp dấu cách và dòng trống thừa.",
           "'ƣ' → 'ư' (phông Việt cũ); ký tự vùng riêng của font Symbol → Unicode. Văn bản có code giữ thụt lề đầu dòng (C-09).",
           "Không sửa nội dung (không paraphrase, không sửa số liệu / tên); source_sha256 vẫn là của text gốc."])
    d.box("lines", 30, 290, W, "step", "lines.clean_rows: xoá dòng lặp (LINES_VERSION 6)",
          ["A. Trong văn bản: gộp dòng giống nhau liền nhau; dòng ngắn lặp ≥ 3 lần chỉ giữ lần đầu; bỏ dòng số trang. Miễn cho văn bản có code / bảng và SEA-Instruct.",
           "B. Tiêu đề / chân trang sách: đã làm lúc ingest (clean_pages).",
           "C. Liên văn bản, chỉ nguồn web: xoá dòng có ở ≥ max(20; 0,01%) số văn bản của nguồn (menu, chân trang)."])
    d.box("need", 30, 460, W, "cond", "Cần cắt đoạn?", ["nguồn chunked (stbook, giáo trình, VJOL, VISTA): luôn cắt; bài web: chỉ khi > 2.048 token; hội thoại SEA-Instruct: không cắt."])
    d.box("chunk", 30, 550, W, "step", "chunk_rows (CHUNK_VERSION 5)",
          ["gắn doc_sha256 và doc_minhash của CẢ văn bản gốc (dedup chạy theo văn bản, R-19).",
           "cắt theo token Qwen3: mục tiêu 1.024, tối đa 2.048; đoạn cuối < 256 gộp vào trước, không gộp được thì mượn các đoạn văn cuối của đoạn trước.",
           "chỗ cắt ưu tiên: dòng tiêu đề (Chương, Bài, Mục...) → dòng trống → hết câu → cắt cứng chia đều.",
           "doc_id = <doc>#c<i>, parent_doc_id, chunk_index; mỗi đoạn giữ source_path / source_sha256 của văn bản gốc."])
    d.box("samp", 30, 730, W, "cond", "Nguồn chunked và không --take-all?", ["có: lấy mẫu đoạn theo băm (seed, nguồn, parent_doc_id, chunk_index) xuống cfg.mix[nguồn], nên sửa một bài không đổi đoạn được chọn của bài khác. --take-all: giữ hết."])
    d.box("cnt", 30, 830, W, "step", "set_counts", ["char_count, word_count, token_count bằng tokenizer thật (mặc định hf:Qwen/Qwen3-0.6B, đếm theo lô; --tokenizer words chỉ để thử, thấp hơn token thật ~1,37 lần)."])
    d.box("out", 30, 920, W, "out", "02_prepare.parquet", ["manifest: normalize, lines (kèm 30 dòng bị xoá nhiều nhất mỗi nguồn), chunk (số sách, số đoạn trước / sau lấy mẫu)."])
    for a, b in (("in", "norm"), ("norm", "lines"), ("lines", "need"), ("need", "chunk"), ("chunk", "samp"), ("samp", "cnt"), ("cnt", "out")):
        d.link(a, b, "có" if a == "need" else "")
    d.box("skip", 690, 460, 280, "in", "Không cắt", ["bản ghi đi thẳng tới bước đếm (bài web ngắn, hội thoại SFT)."])
    d.link("need", "skip", "không", sa="r", sb="l"); d.link("skip", "cnt", sa="b", sb="r", mid=None)
    return d


# --------------------------------------------------------------------------------------------------
# 04: language
# --------------------------------------------------------------------------------------------------
def diagram_04() -> Diagram:
    """Luồng con 4: language id bằng fastText lid.176 theo đoạn văn."""
    d = Diagram("Luồng con 4: language id (fastText lid.176, theo đoạn văn)", "vi_corpus/pipeline/language.py · spans.py")
    W = 620
    d.box("in", 30, 70, W, "in", "02_prepare.parquet", [])
    d.box("math", 30, 130, W, "step", "strip_math_code (từng lượt thoại riêng)", ["bỏ khối ```, $$…$$, \\[…\\], \\(…\\), $…$ có ký hiệu toán trước khi nhận diện; số $$ lẻ thì không bỏ vùng nào."])
    d.box("seg", 30, 230, W, "step", "Tách đoạn văn", ["theo dòng trống (chỉ có một đoạn thì theo dòng); bỏ đoạn không có chữ cái hoặc < 10 chữ cái (trừ khi mọi đoạn đều ngắn); văn bản rất dài chỉ lấy tối đa lang_max_segments đoạn rải đều, mỗi đoạn tối đa 2.000 ký tự."])
    d.box("sft", 690, 230, 280, "cond", "Hội thoại SEA-Instruct", ["bỏ lượt system và tiền tố vai; language / lang_score tính trên lượt người dùng; lang_mix trên mọi lượt; lang_answer cho phía trả lời."])
    d.link("seg", "sft", sa="r", sb="l", dashed=True)
    d.box("ft", 30, 360, W, "step", "Bộ nhận diện (cfg.lang_model)", ["fasttext:lid.176 (mặc định): tải lid.176.bin một lần vào VI_CORPUS_MODEL_DIR hoặc ~/.cache/vi_corpus; đoán từng đoạn → (nhãn ISO, độ tin cậy).",
                                                                       "heuristic: tỷ lệ chữ có dấu tiếng Việt + từ chức năng vi / en; chỉ để thử, không cần mạng."], tag="Ray")
    d.box("mix", 30, 500, W, "step", "Tổng hợp theo số CHỮ CÁI", ["lang_mix = phần chữ cái của mỗi ngôn ngữ; language = ngôn ngữ nhiều nhất; lang_score = Σ(chữ cái × độ tin cậy) của đoạn thuộc language / tổng chữ cái."])
    d.box("und", 690, 500, 280, "stop", "Không còn chữ cái", ["language = und, lang_score = 0, lang_mix = {} (stage quality gắn lang_unknown, không loại vì ngôn ngữ)."])
    d.link("mix", "und", "rỗng", sa="r", sb="l", dashed=True)
    d.box("out", 30, 620, W, "out", "03_language.parquet", ["cột language, lang_score, lang_mix (JSON), lang_answer (SFT); song song qua CuratorBackend khi có --executor. Ngưỡng min_lang_score và mixed_language dùng ở stage quality."])
    for a, b in (("in", "math"), ("math", "seg"), ("seg", "ft"), ("ft", "mix"), ("mix", "out")):
        d.link(a, b)
    return d


# --------------------------------------------------------------------------------------------------
# 05: quality
# --------------------------------------------------------------------------------------------------
def diagram_05() -> Diagram:
    """Luồng con 5: quality (số đo, lỗi cứng, lỗi mềm, band, reason code)."""
    d = Diagram("Luồng con 5: quality (điểm 0-100, band, reason code)", "vi_corpus/pipeline/quality.py · config.py (SourceProfile) · common/garbled.py · common/defects.py")
    W = 620
    d.box("in", 30, 70, W, "in", "03_language.parquet", [])
    d.box("met", 30, 130, W, "step", "compute_metrics (trên văn xuôi sau strip_math_code)",
          ["ký tự: alpha, digit, symbol, upper, replacement, mean_word_len, vi_diacritic; dòng: dup_line, short_line, bullet, end_punct, boilerplate_line, ref_line; khác: url, rep_trigram, words, lines, math_share.",
           "chữ hỏng (VJOL / VISTA): garbled, pua_chars, junk_script_chars, unaccented, reversed_gain, hyphen_lost; bộ đo tone_shift, dropped_letters, gibberish, joined_words."], tag="Ray")
    d.box("odd", 30, 300, W, "step", "annotate_odd_words (toàn lần chạy, tuần tự)", ["từ Latinh không phải âm tiết Việt hợp lệ và chỉ có ở < 3 tạp chí → odd_word (cần tập âm tiết syllables.json)."])
    d.box("ppl", 690, 300, 280, "gpu", "annotate_perplexity (tuỳ chọn)", ["--perplexity-model Qwen/Qwen3-0.6B, 512 token đầu, GPU nhỏ; mặc định tắt."])
    d.link("odd", "ppl", sa="r", sb="l", dashed=True)
    d.box("hard", 30, 390, W, "cond", "Có lỗi cứng?",
          ["cơ bản: empty, too_short, too_long, lang_not_allowed, low_alpha, replacement_chars, repeated_ngrams, duplicate_lines.",
           "chữ hỏng: garbled_font (≥ 8% từ), private_use_chars, math_font_junk, lost_diacritics, reversed_text, lost_letters, odd_words (từ lạ ≥ 15%, hoặc ≥ 8% kèm perplexity cao).",
           "tone_shift, dropped_letters, gibberish, joined_words chỉ đo, chưa loại (HARD_DETECTOR_THRESHOLDS = None, chờ ngưỡng từ nhãn chuẩn)."])
    d.box("d", 690, 420, 280, "stop", "Band D", ["điểm bị chặn ở 40; vẫn ghi đủ reason code. Văn bản gần như toàn công thức / code được miễn các mã hình dạng, trừ vòng lặp (rep_trigram_raw)."])
    d.link("hard", "d", "có", sa="r", sb="l")
    d.box("soft", 30, 590, W, "step", "Lỗi mềm: trừ điểm từ 100",
          ["repetitive −20, dup_lines_some −15, digit_heavy −15, symbol_heavy −15, mostly_upper −15, odd_word_length −15, low_diacritic −15, low_lang_score −10, some_replacement_chars −10, high_perplexity −10.",
           "nguồn web (SEA-PILE): url_heavy −20, boilerplate −20, list_like −15, bullet_heavy −10, few_sentences −10.",
           "chỉ gắn nhãn, không trừ điểm: mixed_language, lang_unknown, reference_list."])
    d.link("hard", "soft", "không")
    d.box("band", 30, 760, W, "step", "band_of(điểm)", ["A ≥ 85, B ≥ 70, C ≥ 55, còn lại D; ngưỡng và bật / tắt kiểm tra theo SourceProfile của nguồn (config.py), vẫn là điểm khởi đầu chưa chỉnh."])
    d.box("out", 30, 850, W, "out", "04_quality.parquet", ["quality_metrics (JSON số đo thô), quality_score, quality_band, reason_codes; song song qua CuratorBackend khi có --executor."])
    d.link("in", "met"); d.link("met", "odd"); d.link("odd", "hard"); d.link("soft", "band"); d.link("band", "out")
    d.link("d", "band", sa="b", sb="r")
    return d


# --------------------------------------------------------------------------------------------------
# 06: dedup
# --------------------------------------------------------------------------------------------------
def diagram_06() -> Diagram:
    """Luồng con 6: dedup vòng 1 (theo văn bản) + cổng quyền + kho dấu vân tay + vòng 2 toàn cục."""
    d = Diagram("Luồng con 6: dedup (vòng 1 trong lần chạy, vòng 2 toàn cục) và cổng quyền", "vi_corpus/pipeline/dedup.py · dedup_global.py · scripts/dedup_global.py")
    W = 620
    d.box("in", 30, 70, W, "in", "04_quality.parquet", [])
    d.box("rg", 30, 130, W, "step", "Cổng quyền", ["rights_status ∈ OPEN_RIGHTS → pass, còn lại quarantine. Mặc định chỉ gắn nhãn (R-01); chế độ enforce mới loại."])
    d.box("band", 30, 220, W, "cond", "quality_band ∈ keep_bands (A, B, C)?", [])
    d.box("rq", 690, 220, 280, "stop", "rejected:quality", ["band D bị loại khỏi clean, nhưng vẫn nằm trong 05_dedup.parquet."])
    d.link("band", "rq", "không", sa="r", sb="l")
    d.box("grp", 30, 290, W, "step", "Gom đoạn theo văn bản (parent_doc_id), 2 nhóm", ["cpt: mọi nguồn dedup chung với nhau; sft: SEA-Instruct chỉ dedup trong nhóm của mình (Q4)."])
    d.box("ex", 30, 385, W, "step", "Exact", ["trùng doc_sha256 của cả văn bản gốc → dup_kind = exact."])
    d.box("fz", 30, 460, W, "step", "Fuzzy: MinHash cả văn bản", ["chữ ký doc_minhash (128 hoán vị, shingle 5 từ, văn bản ≥ 20 từ); LSH 16 băng × 8 hàng, bucket lớn chỉ so với phần tử đầu; xác nhận Jaccard ước lượng ≥ 0,8 → dup_kind = fuzzy."])
    d.box("l1", 30, 575, W, "step", "L1 bao hàm", ["băm đoạn văn (≥ para_min_words từ); văn bản có ≥ 3 đoạn dài mà ≥ 80% nằm trong MỘT văn bản lớn hơn → dup_kind = contained, chung họ với văn bản chứa nó. Chỉ trong phạm vi lần chạy."])
    d.box("rep", 30, 690, W, "step", "Chọn bản giữ của mỗi họ trùng", ["union-find; ưu tiên nguồn (VJOL 0 → SEA 1 → stbook 2 → khác 3, R-15) → điểm rule trung bình → doc_id. dedup_family_id = fam_<sha1 của đại diện>; mọi đoạn của văn bản trùng → rejected:duplicate (dup_kind, dup_of)."])
    d.box("pg", 30, 810, W, "step", "Exact theo đoạn", ["đoạn có text giống hệt một đoạn tốt hơn ở văn bản khác trong nhóm → loại."])
    d.box("out", 30, 885, W, "out", "05_dedup.parquet", ["TẤT CẢ bản ghi, kể cả bị loại; status = kept | rejected:quality | rejected:duplicate | rejected:rights; thống kê exact / fuzzy / contained, họ nhiều bản, quarantined."])
    for a, b in (("in", "rg"), ("rg", "band"), ("band", "grp"), ("grp", "ex"), ("ex", "fz"), ("fz", "l1"), ("l1", "rep"), ("rep", "pg"), ("pg", "out")):
        d.link(a, b, "có" if a == "band" else "")
    d.box("idx", 690, 690, 280, "out", "Kho dấu vân tay vòng 1", ["state/dedup_index/<nguồn>/<run>.parquet: doc_id, sha256, MinHash, điểm, quyền của mọi văn bản (bản bị loại = bia mộ, alive = False). Không có text."])
    d.link("rep", "idx", sa="r", sb="l")
    d.box("g2", 690, 810, 280, "step", "Vòng 2: scripts/dedup_global.py", ["gộp kho của MỌI nguồn (dòng written_at mới nhất), ưu tiên tính lại, họ trùng toàn cục → processed/dedup_global/verdict.parquet (kept / duplicate / rights; chỉ đánh dấu)."])
    d.link("idx", "g2")
    d.box("ap", 690, 960, 280, "cond", "Áp ở finalize", ["run_pipeline.py --global-verdict ...: knowledge unit và chia split sinh SAU vòng 2."])
    d.link("g2", "ap")
    d.box("l3", 690, 470, 280, "plan", "L3 semantic dedup", ["SemDeDup trên embedding (D-03): chưa làm; chờ chốt mô hình embedding (K-07)."])
    return d


# --------------------------------------------------------------------------------------------------
# 07: embed + reduce
# --------------------------------------------------------------------------------------------------
def diagram_07() -> Diagram:
    """Luồng con 7: embed, reduce (PCA / UMAP / HDBSCAN) và bản đồ embedding trong report.html."""
    d = Diagram("Luồng con 7: embed → reduce → bản đồ embedding", "vi_corpus/pipeline/embed.py · reduce.py · viz.py")
    W = 620
    d.box("in", 30, 70, W, "in", "05_dedup.parquet (mọi bản ghi, hoặc chỉ bản giữ lại nếu --embed-scope kept)", [])
    d.box("gate", 30, 130, W, "cond", "Chạy khi cfg.embed_spec khác None", ["--embedder none (bản chạy cả kho VJOL / VISTA, K-07 chờ chốt) thì bỏ qua cả embed và reduce, report không có bản đồ."])
    d.box("hf", 30, 235, 300, "gpu", "hf:<model>", ["mặc định intfloat/multilingual-e5-base (tiền tố 'passage: ', cắt 6.000 ký tự, ≤ 512 token), mean pooling + chuẩn hoá L2; fp16 trên GPU. Mỗi actor Curator giữ 1 bản mô hình trên 1 GPU."], tag="Ray")
    d.box("tf", 350, 235, 300, "step", "tfidf", ["TF-IDF + SVD 64 chiều, CPU, fit trên toàn corpus nên chạy tuần tự; dùng khi thử không GPU."])
    d.box("e", 30, 420, W, "out", "06_embeddings.parquet", ["(doc_id, embedding float32)"])
    d.box("red", 30, 490, W, "step", "reduce_embeddings", ["HDBSCAN chạy trên embedding GỐC (hoặc pca50 / umap10 qua --cluster-space), min_cluster_size mặc định max(2, min(20, n / 10)); -1 = nhiễu.",
                                                           "PCA 2D và UMAP 2D chỉ để vẽ (cuML nếu có GPU, không thì scikit-learn / umap-learn); fit MỘT lần trên toàn ma trận."])
    d.box("r", 30, 625, W, "out", "07_reduced.parquet", ["doc_id, pca_x, pca_y, umap_x, umap_y, cluster_id; manifest ghi số cụm, tỉ lệ nhiễu, tham số."])
    d.box("viz", 30, d.after("r", 24), W, "step", "build_embedding_section (viz.py)", ["plotly scattergl, nhúng một lần vào report.html; chọn UMAP / PCA × tô theo nguồn, trạng thái, band, ngôn ngữ, cụm; rê chuột xem đoạn đầu + reason code."])
    d.box("lim", 690, 490, 280, "plan", "Giới hạn chưa gỡ", ["fit trên mẫu phân tầng và Datashader khi > 100k điểm; HDBSCAN scikit-learn không scale tới hàng triệu đoạn."])
    for a, b in (("in", "gate"), ("e", "red"), ("red", "r"), ("r", "viz")):
        d.link(a, b)
    d.link("gate", "hf"); d.link("gate", "tf"); d.link("hf", "e"); d.link("tf", "e")
    return d


# --------------------------------------------------------------------------------------------------
# 08: finalize
# --------------------------------------------------------------------------------------------------
def diagram_08() -> Diagram:
    """Luồng con 8: finalize (verdict vòng 2, clean, knowledge unit, segment CPT, quét nhiễm, audit, báo cáo) + cpt_pack."""
    d = Diagram("Luồng con 8: finalize (clean, knowledge unit, segment CPT, audit, báo cáo)", "vi_corpus/pipeline/runner.py · knowledge.py · contamination.py · audit.py · report.py · cpt_pack.py")
    W = 620
    d.box("in", 30, 70, W, "in", "Bản ghi sau dedup (+ 07_reduced.parquet nếu có)", [])
    d.box("gv", 30, 130, W, "cond", "Có --global-verdict?", ["có: apply_verdict (dedup_global.py) cập nhật status và họ trùng theo dedup vòng 2 (D-09), đếm superseded; không có thì họ trùng chỉ là của lần chạy này."])
    d.box("clean", 30, 240, W, "step", "clean.parquet", ["chỉ các bản ghi status = kept."])
    d.box("ku", 30, 315, W, "step", "build_units → knowledge_units.parquet", ["CPT: mỗi đoạn giữ lại là 1 đơn vị (candidate). SFT: mỗi cặp (người → trợ lý) liền kề của SEA-Instruct là 1 đơn vị (instruction).",
                                                                            "split_bucket = sha1(dedup_family_id) % 100: chia train / val / test sau này theo họ trùng, không chia ở tầng corpus. Hybrid ghép ở bước huấn luyện."])
    d.box("seg", 30, 450, W, "step", "build_cpt_segments → cpt_segments.parquet", ["nối MỌI đoạn liền nhau còn giữ của cùng văn bản thành 1 segment, không giới hạn độ dài (gặp đoạn bị loại thì ngắt); kèm lineage, quyền, họ trùng, số token, cờ đầu / cuối văn bản; metadata (tiêu đề, tạp chí, năm, ngành...) ở cột riêng."])
    d.box("con", 30, 575, W, "step", "contamination.scan", ["13-gram với configs/benchmarks/*.txt | *.jsonl; danh sách hiện RỖNG nên chưa quét gì. Chỉ đếm / đánh dấu, không loại."])
    d.box("l01", 690, 640, 280, "plan", "Viết lại (L-01)", ["quét theo từng câu benchmark (cả đáp án), 8-gram, vòng ngữ nghĩa; chờ K-01."])
    d.link("con", "l01", sa="r", sb="l", dashed=True)
    d.box("aud", 30, 665, W, "step", "audit_rows → audit.json", ["lineage_rate (đủ source_key, source_path, source_sha256) mục tiêu ≥ 0,95; phễu theo nguồn: vào → rejected:* → kept; token (và so với tokenizer khác); PII: đếm email / số điện thoại VN (không sửa text)."])
    d.box("man", 30, 785, 300, "out", "manifest.json", ["cấu hình, vân tay, giờ, RAM đỉnh, thống kê từng stage"])
    d.box("rep", 350, 785, 300, "out", "report.html", ["một file tự chứa: KPI, phễu, band, reason code, ví dụ, bản đồ embedding"])
    for a, b in (("in", "gv"), ("gv", "clean"), ("clean", "ku"), ("ku", "seg"), ("seg", "con"), ("con", "aud")):
        d.link(a, b)
    d.link("aud", "man"); d.link("aud", "rep")
    d.box("pack", 690, 450, 280, "step", "scripts/cpt_pack.py (ngoài pipeline)", ["cắt segment theo ngữ cảnh 8.192 token, ghép best-fit, EOS chỉ ở cuối văn bản thật; tiền tố metadata tắt (--use-metadata-prefix). Đổi cỡ ngữ cảnh không phải chạy lại pipeline."])
    d.link("seg", "pack", sa="r", sb="l")
    d.box("judge", 690, 315, 280, "step", "make_judge_sample.py (ngoài pipeline)", ["mẫu 50k phân tầng nguồn × độ dài để LLM chấm (chưa có judge)."])
    d.link("ku", "judge", sa="r", sb="l", dashed=True)
    return d


# --------------------------------------------------------------------------------------------------
# 22: VJOL / VISTA từ PDF tới bản ghi
# --------------------------------------------------------------------------------------------------
def diagram_22() -> Diagram:
    """Sơ đồ 22: bộ xử lý bài báo PDF dùng chung của VJOL và VISTA (từ PDF thô tới bản ghi cho stage ingest)."""
    d = Diagram("Luồng con 22: VJOL và VISTA, từ PDF tới bản ghi", "vi_corpus/vjol/ (extract, layout, cli, records, clean, reocr) · vi_corpus/vista/ · scripts/vjol/ · scripts/vista/")
    W = 640
    d.box("raw", 30, 70, W, "in", "raw/vjol/<năm>/<article_id>_<galley_id>.pdf  ·  raw/vista/<năm>/<publication_id>.pdf (+ unknown/)",
          ["metadata: vjol/metadata/vjol_merged.jsonl; vista/metadata/vista_detail.jsonl (join theo publication_id; 1 PDF có thể thuộc nhiều bài)."])
    d.box("s1", 30, 170, W, "step", "① Trích lớp chữ, mọi trang (CPU, PyMuPDF)",
          ["method mỗi trang: text | tcvn3 (đổi font cũ → Unicode) | empty | needs_ocr (trang scan). Chọn mẫu --sample-docs N chia đều theo năm, hoặc cả kho. Nhiều tiến trình chia PDF bằng khoá flock; checkpoint theo sha256 PDF."])
    d.box("s2", 30, 290, W, "step", "② Dò trang hỏng (CPU)",
          ["chữ hỏng do font: 4 tín hiệu (≥ 2 dấu thanh, hoa-thường lẫn, ñ ö ï ø å, từ ngoài tập âm tiết) → trang bị OCR lại cả trang. PDF do phần mềm OCR đời cũ tạo (producer) hoặc ≥ 30% trang hỏng → OCR lại cả bài; lỗi tone_shift / dropped_letters (REOCR_SIGNALS) cũng đưa bài đi OCR lại."])
    d.box("c", 30, 440, W, "cond", "Trang scan hoặc lỗi font?", ["--no-ocr: bỏ ③, trang scan giữ needs_ocr (bản thử CPU, PP-01); đếm số trang cần OCR bằng make count-ocr."])
    d.box("s3", 30, 540, W, "gpu", "③ OCR chuỗi VLM (GPU, mỗi GPU một server vLLM + một tiến trình)",
          ["qwen3vl_8b_mt4k → rp110 nếu guard báo → Paddle + VietOCR → bỏ trang (ocr_vlm: tier, status ok | blank | flagged, rejected_text). Trang vẽ lại cả trang; cache OCR theo trang (interim/vjol_ocr_cache/); lưu text_md (T-06). ServerDown / EngineUnavailable dừng cả lượt."],
          mono="scripts/vjol/run_extract_vlm.sh · scripts/vista/run_extract_vlm.sh")
    d.box("ck", 30, 700, W, "out", "Checkpoint interim/vjol_text | vista_text/<sha256>.json", ["pages[{text, method, tier, status}], extract_version (đổi → làm lại; VISTA thêm +op<băm danh sách producer>), xuất Parquet cuối lượt."])
    d.box("s4", 30, 790, W, "step", "④ Sinh bản ghi (CPU, records.py + clean.py)",
          ["bỏ bìa, mục lục, tài liệu tham khảo, tiêu đề chạy ở đầu / giữa trang, dòng rác; khôi phục 'ư' ('ph-ơng', tuỳ chọn hyphen-lm); che email và số điện thoại (J-63); nối metadata. VISTA: một bản ghi / PDF, meta.publication_ids, meta.shared_pdf."])
    d.box("out", 30, 925, W, "out", "→ stage ingest của run_pipeline.py", ["PROFILES['vjol'] (vi + en, chunked); --shard I/N chia theo sha256; xem sơ đồ 02 và 23."])
    for a, b in (("raw", "s1"), ("s1", "s2"), ("s2", "c"), ("c", "s3"), ("s3", "ck"), ("ck", "s4"), ("s4", "out")):
        d.link(a, b, "có" if a == "c" else "")
    d.link("c", "ck", "không", sa="r", sb="r", mid=688)
    d.box("tools", 700, 170, 270, "in", "Công cụ đo, chỉ CPU trừ khi ghi chú", ["make count-ocr: đếm trang cần OCR (S-27).", "check_garbled.py, check_garbled_missed.py: đo chữ hỏng.", "audit_run.py, groundtruth.py, compare_runs.py --paired: nhãn chuẩn, so hai lần chạy.", "plan_reocr.py: ước giờ OCR lại cả bài.", "font_canary.py (GPU): đo nhóm font (mẫu 3 bài / nhóm)."])
    d.box("vnpt", 700, 440, 270, "plan", "OCR nội bộ VNPT (OI-01)", ["scripts/ocr_internal/: thử 30 + 30 PDF, dựng trang duyệt tay; code xong, chưa chạy thật."])
    d.box("todo", 700, 540, 270, "plan", "Chưa làm", ["cmap_healer (V-10), parser bố cục, lọc trang baseline theo tỉ lệ âm tiết (O-27 c), thay nhóm font xấu theo bài (J-48 mục 3), bảng lệch dấu riêng của VISTA."])
    return d


# --------------------------------------------------------------------------------------------------
# 23: chạy cả kho theo phần
# --------------------------------------------------------------------------------------------------
def diagram_23() -> Diagram:
    """Sơ đồ 23: chạy pipeline chung trên cả kho VJOL / VISTA bằng nhiều phần (shard) rồi dedup toàn cục."""
    d = Diagram("Luồng con 23: pipeline cả kho VJOL + VISTA chia phần (shard)", "make pipeline-cpu · scripts/run_pipeline_cpu.sh · scripts/vista/run_vista.sh · scripts/dedup_global.py · scripts/vista/summarize_runs.py")
    W = 640
    d.box("ck", 30, 70, W, "in", "Checkpoint đã trích", ["interim/vjol_text, interim/vista_text (bản CPU thuần: trang needs_ocr bị bỏ qua; sau OCR GPU: đủ chữ)."])
    d.box("sh", 30, 160, W, "step", "1. shards: N phần chạy song song (JOBS tiến trình)",
          ["run_pipeline.py --mix <nguồn>=... --take-all --shard I/N --until dedup (ingest → prepare → language → quality → dedup vòng 1). Phần chia theo sha256 PDF; mỗi phần là một lần chạy độc lập, có kho dấu vân tay riêng, RAM vừa; mỗi tiến trình 1 luồng; --embedder none."],
          mono="make pipeline-cpu ROOT=<data-root> JOBS=128 SHARDS=128 STEP=shards")
    d.box("dd", 30, 320, W, "step", "2. dedup: dedup vòng 2 toàn cục", ["scripts/dedup_global.py gộp kho dấu vân tay của MỌI phần và mọi nguồn → processed/dedup_global/verdict.parquet; bắt cả trùng giữa các phần và giữa VJOL với VISTA."], mono="STEP=dedup")
    d.box("fi", 30, 440, W, "step", "3. finalize: N phần chạy lại với --global-verdict", ["tới hết finalize: clean, knowledge_units, cpt_segments, audit, report.html (không bản đồ)."], mono="STEP=finalize")
    d.box("su", 30, 540, W, "step", "4. summary", ["summarize_runs.py: đoạn, token giữ, phễu, RAM đỉnh (peak_rss_mb), giây của từng phần → processed/<nguồn>_full_summary.json; cảnh báo phần dang dở hoặc chưa áp verdict."], mono="STEP=summary")
    d.box("out", 30, 650, W, "out", "Kết quả", ["các clean.parquet / cpt_segments.parquet của từng phần; ghép các phần thành một đầu vào cho scripts/cpt_pack.py: chưa làm."])
    for a, b in (("ck", "sh"), ("sh", "dd"), ("dd", "fi"), ("fi", "su"), ("su", "out")):
        d.link(a, b)
    d.box("stop", 700, 160, 270, "stop", "Phần lỗi → dừng", ["ghi logs/pipe_failed_<bước>.txt, không chạy bước sau. Chạy lại đúng lệnh cũ = làm tiếp phần lỗi (vân tay stage)."])
    d.link("sh", "stop", sa="r", sb="l", dashed=True)
    d.box("prog", 700, 320, 270, "in", "make pipeline-progress", ["xem tiến độ + giờ dự kiến xong (chỉ đọc file, chạy ở terminal khác)."])
    d.box("vista", 700, 430, 270, "in", "VISTA: run_vista.sh", ["check → survey → sample-extract → sample-pipeline → (duyệt mẫu) → full-extract → full-pipeline (SHARDS = 8) → dedup → summary."])
    d.box("final", 700, 590, 270, "plan", "Bản cuối (sau OCR GPU, PP-01)", ["chạy lại từ ingest với --perplexity-model và --hyphen-lm-model Qwen/Qwen3-0.6B bật; PIPE_ARGS phải giống nhau giữa các lần chạy. make pipeline-cpu mới chạy thử với dữ liệu giả, chưa chạy thật."])
    return d


DIAGRAMS = {
    "00": ("00_pipeline", diagram_00), "01": ("01_runner", diagram_01), "02": ("02_ingest", diagram_02),
    "03": ("03_prepare", diagram_03), "04": ("04_language", diagram_04), "05": ("05_quality", diagram_05),
    "06": ("06_dedup", diagram_06), "07": ("07_embed_reduce", diagram_07), "08": ("08_finalize", diagram_08),
    "22": ("22_vjol_vista", diagram_22), "23": ("23_shard_run", diagram_23),
}


def main() -> None:
    """Dòng lệnh: vẽ các sơ đồ đã chọn vào docs/diagrams/ (và PNG nếu có --png)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", default=",".join(DIAGRAMS), help="Số đầu tên file, cách nhau bằng dấu phẩy (mặc định: tất cả).")
    ap.add_argument("--png", action="store_true", help="Ghi thêm bản PNG (bằng PyMuPDF), dùng cho 00_pipeline.png.")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Thư mục ghi (mặc định docs/diagrams/).")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for num in [s.strip() for s in args.only.split(",") if s.strip()]:
        name, fn = DIAGRAMS[num]
        path = args.out_dir / f"{name}.svg"
        fn().save(path)
        print("đã ghi", path)
        if args.png:
            import pymupdf  # PyMuPDF đọc được SVG nhưng chỉ nhận một tên font, nên đổi sang DejaVu Sans khi dựng PNG

            tmp = path.with_suffix(".tmp.svg")
            tmp.write_text(path.read_text(encoding="utf-8").replace(f'font-family="{FONT}"', 'font-family="DejaVu Sans"'), encoding="utf-8")
            doc = pymupdf.open(str(tmp))
            doc[0].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(str(path.with_suffix(".png")))
            tmp.unlink()
            print("đã ghi", path.with_suffix(".png"))


if __name__ == "__main__":
    main()
