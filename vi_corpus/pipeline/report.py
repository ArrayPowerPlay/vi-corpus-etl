"""
Báo cáo trực quan (report.html) của một lần chạy pipeline: một file HTML tự đủ, không cần thư viện vẽ hay mạng.

Gồm: thẻ KPI, phễu lọc theo nguồn, band chất lượng, ngôn ngữ, phân phối số từ và điểm chất lượng (mỗi nguồn một ô,
cùng trục), top reason code, bảng số liệu (audit) và các mẫu văn bản thật (ngẫu nhiên giữ lại / bị loại / bản trùng) để
đọc bằng mắt, vì biểu đồ không thay được việc đọc thử. Biểu đồ là SVG nội tuyến; rê chuột lên cột để xem số liệu
(thẻ <title>). Màu theo CSS variable, có chế độ tối; mỗi chuỗi >= 2 đều có chú giải, không dựa riêng vào màu.
"""

import html
import json
import math
import random
from pathlib import Path

from vi_corpus.pipeline.io import loads

STATUS_SERIES = [("kept", "Giữ lại", "s1"), ("rejected:quality", "Loại: chất lượng", "s2"),
                 ("rejected:duplicate", "Loại: trùng lặp", "s3"), ("rejected:rights", "Loại: quyền", "s4")]
BAND_SERIES = [("A", "A (tốt)", "b1"), ("B", "B", "b2"), ("C", "C", "b3"), ("D", "D (loại)", "b4")]
LANG_SERIES = [("vi", "Tiếng Việt", "s1"), ("en", "Tiếng Anh", "s2"), ("other", "Khác / không rõ", "s3")]

CSS = """
:root{color-scheme:light;--bg:#fcfcfb;--panel:#fff;--ink:#0b0b0b;--ink2:#52514e;--grid:#e6e5e1;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--b1:#1c5cab;--b2:#3987e5;--b3:#86b6ef;--b4:#eb6834;--gap:#fcfcfb}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#1a1a19;--panel:#222221;--ink:#fff;
--ink2:#c3c2b7;--grid:#383835;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--b1:#6da7ec;--b2:#3987e5;--b3:#1c5cab;--b4:#d95926;--gap:#222221}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#1a1a19;--panel:#222221;--ink:#fff;--ink2:#c3c2b7;--grid:#383835;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--b1:#6da7ec;--b2:#3987e5;--b3:#1c5cab;--b4:#d95926;--gap:#222221}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:1120px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 4px}
p.sub{color:var(--ink2);margin:0 0 12px;font-size:14px}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:16px}
.kpi{background:var(--panel);border:1px solid var(--grid);border-radius:8px;padding:12px 14px}.kpi b{display:block;font-size:24px}
.kpi span{color:var(--ink2);font-size:13px}.card{background:var(--panel);border:1px solid var(--grid);border-radius:8px;padding:14px;overflow-x:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}svg{display:block;width:100%;height:auto}
svg text{fill:var(--ink2);font-size:11px}svg .val{fill:var(--ink)}svg .axis{stroke:var(--grid)}
.s1{fill:var(--s1);background:var(--s1)}.s2{fill:var(--s2);background:var(--s2)}.s3{fill:var(--s3);background:var(--s3)}.s4{fill:var(--s4);background:var(--s4)}
.b1{fill:var(--b1);background:var(--b1)}.b2{fill:var(--b2);background:var(--b2)}.b3{fill:var(--b3);background:var(--b3)}.b4{fill:var(--b4);background:var(--b4)}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin:0 0 8px;font-size:13px;color:var(--ink2)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right}
th:first-child,td:first-child{text-align:left}details{background:var(--panel);border:1px solid var(--grid);border-radius:8px;margin:6px 0;padding:8px 12px}
summary{cursor:pointer;font-size:14px}summary small{color:var(--ink2)}pre{white-space:pre-wrap;word-break:break-word;font:13px/1.5 ui-monospace,monospace;margin:8px 0 0}
.tag{display:inline-block;border:1px solid var(--grid);border-radius:4px;padding:0 6px;margin-right:4px;font-size:12px;color:var(--ink2)}
"""


def _legend(series: list[tuple[str, str, str]]) -> str:
    """Chú giải: ô màu + tên, theo thứ tự chuỗi."""
    return '<div class="legend">' + "".join(f'<span><i class="{c}"></i>{html.escape(n)}</span>' for _, n, c in series) + "</div>"


def stacked_bars(data: dict[str, dict[str, int]], series: list[tuple[str, str, str]], title: str) -> str:
    """
    Biểu đồ cột ngang xếp chồng theo nguồn (mỗi hàng một nguồn, độ dài = số bản ghi, khe 2px giữa các đoạn).

    Args:
        data:   {nguồn: {khóa chuỗi: số lượng}}.
        series: [(khóa, tên hiển thị, class màu)].
        title:  Mô tả ngắn cho trình đọc màn hình.
    """
    left, width, row_h = 150, 640, 30
    top = max((sum(d.values()) for d in data.values()), default=1) or 1
    out = [f'<svg viewBox="0 0 {left + width + 90} {row_h * len(data) + 8}" role="img" aria-label="{html.escape(title)}">']
    for r, (src, d) in enumerate(data.items()):
        y, x = r * row_h + 4, left
        total = sum(d.values())
        out.append(f'<text x="{left - 8}" y="{y + 15}" text-anchor="end" class="val">{html.escape(src)}</text>')
        for key, name, cls in series:
            v = d.get(key, 0)
            w = width * v / top
            if v:
                out.append(f'<rect class="{cls}" x="{x:.1f}" y="{y}" width="{max(w - 2, 1):.1f}" height="22" rx="3">'
                           f'<title>{html.escape(src)} · {html.escape(name)}: {v} ({100 * v / total:.1f}%)</title></rect>')
                if w > 34:
                    out.append(f'<text x="{x + w / 2 - 1:.1f}" y="{y + 15}" text-anchor="middle" '
                               f'style="fill:#fff;font-weight:600">{v}</text>')
            x += w
        out.append(f'<text x="{x + 6:.1f}" y="{y + 15}">{total}</text>')
    out.append("</svg>")
    return _legend(series) + "".join(out)


def histogram(values: list[float], edges: list[float], label: str, log_axis: bool = False) -> str:
    """
    Biểu đồ cột phân phối (một ô): đếm giá trị theo khoảng [edges[i], edges[i+1]), trục y bắt đầu từ 0.

    Args:
        values:   Các giá trị.
        edges:    Biên các khoảng (tăng dần); giá trị ngoài biên được dồn vào khoảng đầu / cuối.
        label:    Nhãn trục x.
        log_axis: True nếu biên là lũy thừa (chỉ ảnh hưởng nhãn: in dạng 1k, 10k).
    """
    counts = [0] * (len(edges) - 1)
    for v in values:
        i = next((k for k in range(len(counts)) if v < edges[k + 1]), len(counts) - 1)
        counts[max(i, 0)] += 1
    peak = max(counts) or 1
    w, h, pad_l, pad_b = 340, 150, 34, 24
    bw = (w - pad_l - 6) / len(counts)
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(label)}">',
           f'<line class="axis" x1="{pad_l}" y1="{h - pad_b}" x2="{w - 6}" y2="{h - pad_b}"/>',
           f'<text x="{pad_l - 4}" y="12" text-anchor="end">{peak}</text><text x="{pad_l - 4}" y="{h - pad_b}" text-anchor="end">0</text>']
    for i, c in enumerate(counts):
        bh = (h - pad_b - 8) * c / peak
        out.append(f'<rect class="s1" x="{pad_l + i * bw + 1:.1f}" y="{h - pad_b - bh:.1f}" width="{max(bw - 2, 1):.1f}" '
                   f'height="{bh:.1f}" rx="2"><title>{edges[i]:g} – {edges[i + 1]:g}: {c}</title></rect>')

    def fmt(x: float) -> str:
        """Nhãn biên: 1000 -> 1k."""
        return f"{x / 1000:g}k" if log_axis and x >= 1000 else f"{x:g}"
    out.append(f'<text x="{pad_l}" y="{h - 8}">{fmt(edges[0])}</text><text x="{w - 6}" y="{h - 8}" text-anchor="end">{fmt(edges[-1])}</text>'
               f'<text x="{(w + pad_l) / 2}" y="{h - 8}" text-anchor="middle">{html.escape(label)}</text></svg>')
    return "".join(out)


def hbars(items: list[tuple[str, int]], title: str) -> str:
    """Biểu đồ cột ngang đơn sắc (đã sắp xếp), nhãn giá trị ở đầu cột, dùng cho top reason code."""
    if not items:
        return "<p class='sub'>Không có dữ liệu.</p>"
    left, width, row_h = 190, 560, 26
    top = max(v for _, v in items) or 1
    out = [f'<svg viewBox="0 0 {left + width + 60} {row_h * len(items) + 6}" role="img" aria-label="{html.escape(title)}">']
    for i, (name, v) in enumerate(items):
        y, w = i * row_h + 3, width * v / top
        out.append(f'<text x="{left - 8}" y="{y + 15}" text-anchor="end" class="val">{html.escape(name)}</text>'
                   f'<rect class="s1" x="{left}" y="{y}" width="{max(w, 1):.1f}" height="18" rx="3"><title>{html.escape(name)}: {v}</title></rect>'
                   f'<text x="{left + w + 6:.1f}" y="{y + 14}">{v}</text>')
    return "".join(out) + "</svg>"


def _excerpt(row: dict, n: int = 700) -> str:
    """Một mẫu văn bản dạng <details>: tiêu đề (id, điểm, band, mã), nội dung đầu n ký tự (đã escape)."""
    text = row["text"] or ""
    tags = "".join(f'<span class="tag">{html.escape(c)}</span>' for c in row["reason_codes"])
    dup = f' · trùng với <code>{html.escape(row["dup_of"])}</code> ({row["dup_kind"]})' if row.get("dup_of") else ""
    return (f'<details><summary><code>{html.escape(row["doc_id"])}</code> · band {row["quality_band"]} '
            f'({row["quality_score"]}) · {row["word_count"]} từ · {row["language"]}{dup} <small>{tags}</small></summary>'
            f'<pre>{html.escape(text[:n])}{"…" if len(text) > n else ""}</pre></details>')


def _samples(rows: list[dict], per_group: int = 4, seed: int = 7) -> str:
    """Mẫu đọc bằng mắt, mỗi nguồn: ngẫu nhiên giữ lại, điểm thấp nhất bị loại vì chất lượng, và cặp trùng lặp."""
    rng = random.Random(seed)
    parts = []
    for src in sorted({r["source_key"] for r in rows}):
        mine = [r for r in rows if r["source_key"] == src]
        kept = [r for r in mine if r["status"] == "kept"]
        bad = sorted((r for r in mine if r["status"] == "rejected:quality"), key=lambda r: r["quality_score"])
        dups = [r for r in mine if r["status"] == "rejected:duplicate"]
        parts.append(f"<h3>{html.escape(src)}</h3>")
        for title, group in (("Giữ lại (ngẫu nhiên)", rng.sample(kept, min(per_group, len(kept)))),
                             ("Bị loại vì chất lượng (điểm thấp nhất)", bad[:per_group]),
                             ("Bị loại vì trùng lặp", dups[:per_group])):
            if group:
                parts.append(f"<p class='sub'>{title}</p>" + "".join(_excerpt(r) for r in group))
    return "".join(parts)


def _table(audit: dict) -> str:
    """Bảng funnel + token theo nguồn (cùng số liệu với biểu đồ, để đọc chính xác)."""
    head = "<tr><th>Nguồn</th><th>Vào</th><th>Giữ lại</th><th>Loại: chất lượng</th><th>Loại: trùng</th><th>Từ vào</th><th>Từ giữ lại</th><th>Token giữ lại</th></tr>"
    body = ""
    for src, f in audit["funnel"].items():
        t = audit["tokens"][src]
        body += (f"<tr><td>{html.escape(src)}</td><td>{f['input']}</td><td>{f.get('kept', 0)}</td><td>{f.get('rejected:quality', 0)}</td>"
                 f"<td>{f.get('rejected:duplicate', 0)}</td><td>{t['words_in']:,}</td><td>{t['words_kept']:,}</td><td>{t['tokens_kept']:,}</td></tr>")
    return f"<table>{head}{body}</table>"


def _viz_section(run_dir: Path, files: list[Path]) -> str:
    """Mục bản đồ embedding: các nút chọn bản đồ (iframe tới viz/*.html, plotly) hoặc ghi chú nếu chưa có."""
    if not files:
        return "<p class='sub'>Chưa có (cần stage embed + reduce và <code>uv sync --group viz</code>).</p>"
    rel = [f.relative_to(run_dir).as_posix() for f in files]
    buttons = "".join(f'<button onclick="show({json.dumps(r)})">{html.escape(Path(r).stem.replace("scatter-", ""))}</button>' for r in rel)
    return (f'<p class="sub">Mỗi điểm là một mẫu; rê chuột để đọc đoạn đầu và reason code. Gồm cả mẫu bị loại (tô theo trạng thái).</p>'
            f'<div class="legend">{buttons}</div><iframe id="viz" src="{html.escape(rel[0])}" style="width:100%;height:660px;border:1px solid var(--grid);border-radius:8px"></iframe>'
            f'<script>function show(u){{document.getElementById("viz").src=u}}</script>')


def write_report(run_dir: Path, rows: list[dict], audit: dict, manifest: dict, viz_files: list[Path] | None = None) -> Path:
    """
    Sinh <run_dir>/report.html từ các bản ghi (đủ mọi status), kết quả audit và manifest.

    Args:
        viz_files: Các file bản đồ embedding (viz.write_scatters) để nhúng bằng iframe; rỗng / None thì ghi chú chưa tính.

    Returns:
        Đường dẫn file báo cáo.
    """
    sources = sorted({r["source_key"] for r in rows})
    by = {s: [r for r in rows if r["source_key"] == s] for s in sources}
    count = lambda key, s: {k: sum(1 for r in by[s] if r[key] == k) for k in {r[key] for r in by[s]}}  # noqa: E731
    total, kept = len(rows), sum(r["status"] == "kept" for r in rows)
    tok_kept = sum(t["tokens_kept"] for t in audit["tokens"].values())
    kpis = [("Mẫu đầu vào", f"{total:,}"), ("Giữ lại", f"{kept:,} ({100 * kept / max(total, 1):.0f}%)"),
            ("Token giữ lại", f"{tok_kept:,}"), ("Truy vết nguồn (giữ lại)", f"{100 * audit['lineage_rate_kept']:.1f}%"),
            ("Bị đánh dấu quarantine", f"{sum(r['rights_gate'] == 'quarantine' for r in rows):,}"),
            ("Knowledge units", f"{manifest.get('finalize', {}).get('knowledge_units', 0):,}")]
    reasons: dict[str, int] = {}
    for r in rows:
        for c in r["reason_codes"]:
            reasons[c] = reasons.get(c, 0) + 1
    word_edges = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 100000]
    hist_words = "".join(f'<div class="card"><b>{html.escape(s)}</b>{histogram([r["word_count"] or 1 for r in by[s]], word_edges, "số từ (thang log)", True)}</div>' for s in sources)
    hist_score = "".join(f'<div class="card"><b>{html.escape(s)}</b>{histogram([r["quality_score"] for r in by[s]], list(range(0, 105, 5)), "điểm chất lượng")}</div>' for s in sources)
    body = f"""<h1>Báo cáo chất lượng pipeline</h1>
<p class="sub">{html.escape(str(run_dir))} · seed {manifest.get('config', {}).get('seed')} · token = {'tokenizer ' + str(manifest.get('config', {}).get('tokenizer')) if manifest.get('config', {}).get('tokenizer') else 'số từ (ước lượng thấp)'}</p>
<div class="kpis">{''.join(f'<div class="kpi"><b>{v}</b><span>{html.escape(k)}</span></div>' for k, v in kpis)}</div>
<h2>Phễu lọc theo nguồn</h2><p class="sub">Mỗi mẫu đầu vào kết thúc ở đúng một trạng thái.</p>
<div class="card">{stacked_bars({s: count('status', s) for s in sources}, STATUS_SERIES, 'Phễu lọc theo nguồn')}</div>
<h2>Band chất lượng</h2><p class="sub">A ≥ 85, B ≥ 70, C ≥ 55, D còn lại hoặc dính lỗi cứng. Giữ A, B, C.</p>
<div class="card">{stacked_bars({s: count('quality_band', s) for s in sources}, BAND_SERIES, 'Band chất lượng theo nguồn')}</div>
<h2>Ngôn ngữ</h2><div class="card">{stacked_bars({s: count('language', s) for s in sources}, LANG_SERIES, 'Ngôn ngữ theo nguồn')}</div>
<h2>Phân phối số từ</h2><p class="sub">Mỗi nguồn một ô, cùng thang đo.</p><div class="grid">{hist_words}</div>
<h2>Phân phối điểm chất lượng</h2><div class="grid">{hist_score}</div>
<h2>Reason code phổ biến</h2><div class="card">{hbars(sorted(reasons.items(), key=lambda kv: -kv[1])[:14], 'Reason code phổ biến')}</div>
<h2>Bản đồ embedding</h2>{_viz_section(run_dir, viz_files or [])}
<h2>Số liệu</h2><div class="card">{_table(audit)}</div>
<p class="sub">PII trong bản ghi giữ lại: {html.escape(json.dumps(audit['pii'], ensure_ascii=False))}. Quét rò rỉ benchmark: {html.escape(audit['contamination'])}.</p>
<h2>Đọc mẫu thật</h2><p class="sub">Phần quan trọng nhất: mở vài mẫu mỗi nhóm và tự đánh giá band / lý do loại có hợp lý không.</p>
{_samples(rows)}"""
    out = run_dir / "report.html"
    out.write_text(f'<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                   f'<title>Báo cáo pipeline</title><style>{CSS}</style></head><body><main>{body}</main></body></html>', encoding="utf-8")
    return out
