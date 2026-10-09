"""
Báo cáo trực quan (report.html) của một lần chạy pipeline: một file HTML tự đủ, không cần thư viện vẽ hay mạng.

Gồm: thẻ KPI, phễu lọc theo nguồn, band chất lượng, ngôn ngữ, phân phối số token và điểm chất lượng (mỗi nguồn một ô,
cùng trục), top reason code, dòng bị xoá nhiều nhất (D-04), bảng số liệu (audit, tỷ lệ token / từ theo nguồn), kết quả
quét rò rỉ benchmark và các mẫu văn bản thật (ngẫu nhiên giữ lại / bị loại / bản trùng) để
đọc bằng mắt, vì biểu đồ không thay được việc đọc thử. Biểu đồ là SVG nội tuyến; rê chuột lên cột để xem số liệu
(thẻ <title>). Màu theo CSS variable, có chế độ tối; mỗi chuỗi >= 2 đều có chú giải, không dựa riêng vào màu.
"""

import html
import json
import random
from pathlib import Path

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
            f'({row["quality_score"]}) · {row["token_count"]} token · {row["word_count"]} từ · {row["language"]} '
            f'{html.escape(row.get("lang_mix") or "")}{dup} <small>{tags}</small></summary>'
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
    head = ("<tr><th>Nguồn</th><th>Vào</th><th>Giữ lại</th><th>Loại: chất lượng</th><th>Loại: trùng</th><th>Từ vào</th>"
            "<th>Từ giữ lại</th><th>Token giữ lại</th><th>Token / từ</th></tr>")
    body = ""
    for src, f in audit["funnel"].items():
        t = audit["tokens"][src]
        ratio = t.get("tokens_per_word_kept")
        body += (f"<tr><td>{html.escape(src)}</td><td>{f['input']}</td><td>{f.get('kept', 0)}</td><td>{f.get('rejected:quality', 0)}</td>"
                 f"<td>{f.get('rejected:duplicate', 0)}</td><td>{t['words_in']:,}</td><td>{t['words_kept']:,}</td><td>{t['tokens_kept']:,}</td>"
                 f"<td>{ratio if ratio is not None else '-'}</td></tr>")
    return f"<table>{head}{body}</table>"


def _dedup_summary(manifest: dict) -> str:
    """Bảng dedup vòng 1 theo nhóm (cpt / sft): số văn bản, số trùng theo loại, số họ > 1 văn bản, cỡ họ lớn nhất (R-19)."""
    groups = manifest.get("dedup", {}).get("groups")
    if not groups:
        return "<p class='sub'>Chưa có thống kê dedup theo nhóm.</p>"
    cols = ("docs", "exact", "fuzzy", "contained", "chunk_exact", "families_multi", "largest_family")
    head = "".join(f"<th>{c}</th>" for c in ("nhóm",) + cols)
    body = "".join("<tr><td>" + html.escape(g) + "</td>" + "".join(f"<td>{st.get(c, 0)}</td>" for c in cols) + "</tr>"
                   for g, st in sorted(groups.items()))
    return f"<table><tr>{head}</tr>{body}</table>"


def _removed_lines(manifest: dict) -> str:
    """Bảng số dòng bị xoá theo nguồn (D-04 mục A, B, C) và danh sách các dòng bị xoá nhiều nhất."""
    lstats = manifest.get("prepare", {}).get("lines")
    if not lstats:
        return "<p class='sub'>Chưa có thống kê (run cũ).</p>"
    pages = lstats.get("page_lines_removed", {})
    rows = "".join(
        f"<tr><td>{html.escape(src)}</td><td>{pages.get(src, 0):,}</td>"
        f"<td>{'tắt' if st.get('in_doc_off') else format(st.get('in_doc_lines', 0), ',')}</td>"
        f"<td>{st.get('in_doc_exempt_docs', 0):,}</td><td>{st.get('cross_doc', {}).get('lines_removed', '-')}</td><td>{st.get('cross_doc', {}).get('threshold', '-')}</td>"
        f"<td>{st.get('chars_removed', 0):,}</td></tr>" for src, st in lstats.get("per_source", {}).items())
    # F-01: danh sách dòng bị xoá tách theo nguồn; run cũ chỉ có một danh sách chung lstats["top_removed"]
    tops = {src: st.get("top_removed", []) for src, st in lstats.get("per_source", {}).items()}
    if "top_removed" in lstats:
        tops = {"(mọi nguồn)": lstats["top_removed"]}
    top = "".join(
        f"<details><summary>{html.escape(src)}: các dòng bị xoá nhiều nhất (đọc lại để chắc không xoá nhầm)</summary>"
        "<table><tr><th>Dòng</th><th>Số lần</th></tr>"
        + "".join(f"<tr><td>{html.escape(ln)}</td><td>{n}</td></tr>" for ln, n in items) + "</table></details>"
        for src, items in tops.items() if items)
    return (f"<table><tr><th>Nguồn</th><th>B: tiêu đề / số trang (ingest)</th><th>A: trong văn bản</th>"
            f"<th>Văn bản miễn A (có code / bảng)</th><th>C: liên văn bản</th>"
            f"<th>Ngưỡng C (số văn bản)</th><th>Ký tự bị xoá (A + C)</th></tr>{rows}</table>{top}")


MATH_CHECK_CODES = ("repetitive", "symbol_heavy", "odd_word_length", "duplicate_lines", "repeated_ngrams")


def _math_sft_checks(rows: list[dict], manifest: dict) -> str:
    """
    Số đo kiểm của F-03 / F-07 theo nguồn: số bản ghi có công thức / code (math_share > 0,1), trong đó bao nhiêu còn dính
    mã hình dạng (kỳ vọng gần 0); số hội thoại có câu trả lời không phải tiếng Việt (để rà bằng mắt / LLM judge).
    """
    stats: dict[str, dict[str, int]] = {}
    for r in rows:
        m = json.loads(r.get("quality_metrics") or "{}")
        st = stats.setdefault(r["source_key"], {"math": 0, "math_flagged": 0, "und": 0})
        st["und"] += r.get("language") == "und"
        if m.get("math_share", 0) > 0.1:
            st["math"] += 1
            st["math_flagged"] += any(c in MATH_CHECK_CODES for c in r["reason_codes"])
    answers = manifest.get("language", {}).get("sft_answers", {})
    body = "".join(
        f"<tr><td>{html.escape(src)}</td><td>{st['math']:,}</td><td>{st['math_flagged']:,}</td><td>{st['und']:,}</td>"
        f"<td>{answers.get(src, {}).get('answer_not_vi', '-')}</td>"
        f"<td>{answers.get(src, {}).get('answer_not_vi_mixed', '-')}</td></tr>" for src, st in sorted(stats.items()))
    return ("<table><tr><th>Nguồn</th><th>Có công thức / code (math_share &gt; 0,1)</th>"
            f"<th>... còn dính {', '.join(MATH_CHECK_CODES)}</th><th>Ngôn ngữ und</th>"
            f"<th>Hội thoại trả lời không phải vi</th><th>... có mixed_language</th></tr>{body}</table>")


def _contamination(audit: dict) -> str:
    """Một dòng tóm tắt kết quả quét rò rỉ benchmark."""
    c = audit.get("contamination") or {}
    if not c.get("benchmarks"):
        return html.escape(c.get("note", "chưa chạy"))
    return "; ".join(f"{html.escape(k)}: {v} bản ghi chạm 13-gram" for k, v in c["flagged"].items())


def write_report(run_dir: Path, rows: list[dict], audit: dict, manifest: dict, embedding_html: str | None = None) -> Path:
    """
    Sinh <run_dir>/report.html từ các bản ghi (đủ mọi status), kết quả audit và manifest.

    Args:
        embedding_html: Khối HTML bản đồ embedding (viz.build_embedding_section) nhúng thẳng vào báo cáo; None thì ghi chú chưa tính.

    Returns:
        Đường dẫn file báo cáo.
    """
    sources = sorted({r["source_key"] for r in rows})
    by = {s: [r for r in rows if r["source_key"] == s] for s in sources}
    count = lambda key, s: {k: sum(1 for r in by[s] if r[key] == k) for k in {r[key] for r in by[s]}}  # noqa: E731
    lang_count = lambda s: {k: sum(1 for r in by[s] if (r["language"] if r["language"] in ("vi", "en") else "other") == k)  # noqa: E731
                            for k in ("vi", "en", "other")}
    total, kept = len(rows), sum(r["status"] == "kept" for r in rows)
    tok_kept = sum(t["tokens_kept"] for t in audit["tokens"].values())
    tokenizer = audit.get("tokenizer", "?")
    fin = manifest.get("finalize", {})
    kpis = [("Mẫu đầu vào", f"{total:,}"), ("Giữ lại", f"{kept:,} ({100 * kept / max(total, 1):.0f}%)"),
            (f"Token giữ lại ({tokenizer})", f"{tok_kept:,}"),
            ("Truy vết nguồn (giữ lại)", f"{100 * audit['lineage_rate_kept']:.1f}%"),
            ("Bị đánh dấu quarantine", f"{sum(r['rights_gate'] == 'quarantine' for r in rows):,}"),
            ("Candidates (đoạn CPT)", f"{fin.get('candidates', 0):,}"), ("Instructions (SFT)", f"{fin.get('instructions', 0):,}")]
    other_tok = audit.get("tokens_kept_other_tokenizers") or {}
    tok_note = (" · token giữ lại theo tokenizer khác: " + ", ".join(
        f"{html.escape(k)} {v:,} ({100 * (v - tok_kept) / max(tok_kept, 1):+.1f}%)" for k, v in other_tok.items())) if other_tok else ""
    reasons: dict[str, int] = {}
    for r in rows:
        for c in r["reason_codes"]:
            reasons[c] = reasons.get(c, 0) + 1
    tok_edges = [1, 16, 64, 128, 256, 512, 768, 1024, 1536, 2048, 4096, 100000]
    hist_words = "".join(f'<div class="card"><b>{html.escape(s)}</b>{histogram([r["token_count"] or 1 for r in by[s]], tok_edges, "số token (thang log)", True)}</div>' for s in sources)
    hist_score = "".join(f'<div class="card"><b>{html.escape(s)}</b>{histogram([r["quality_score"] for r in by[s]], list(range(0, 105, 5)), "điểm chất lượng")}</div>' for s in sources)
    body = f"""<h1>Báo cáo chất lượng pipeline</h1>
<p class="sub">{html.escape(str(run_dir))} · seed {manifest.get('config', {}).get('seed')} · token đếm bằng <code>{html.escape(tokenizer)}</code>{' (số từ: ước lượng THẤP, không dùng đo KPI)' if tokenizer == 'words' else ''} · ngôn ngữ: <code>{html.escape(str(manifest.get('language', {}).get('model', '?')))}</code>{tok_note}</p>
<div class="kpis">{''.join(f'<div class="kpi"><b>{v}</b><span>{html.escape(k)}</span></div>' for k, v in kpis)}</div>
<h2>Phễu lọc theo nguồn</h2><p class="sub">Mỗi mẫu đầu vào kết thúc ở đúng một trạng thái.</p>
<div class="card">{stacked_bars({s: count('status', s) for s in sources}, STATUS_SERIES, 'Phễu lọc theo nguồn')}</div>
<h2>Band chất lượng</h2><p class="sub">A ≥ 85, B ≥ 70, C ≥ 55, D còn lại hoặc dính lỗi cứng. Giữ A, B, C.</p>
<div class="card">{stacked_bars({s: count('quality_band', s) for s in sources}, BAND_SERIES, 'Band chất lượng theo nguồn')}</div>
<h2>Ngôn ngữ</h2><p class="sub">Ngôn ngữ chiếm nhiều nhất theo đoạn văn (fastText). Văn bản trộn ngôn ngữ (ngôn ngữ thứ hai ≥ 20%) có mã <code>mixed_language</code>, chỉ gắn nhãn, không loại (R-33).</p>
<div class="card">{stacked_bars({s: lang_count(s) for s in sources}, LANG_SERIES, 'Ngôn ngữ theo nguồn')}</div>
<h2>Công thức / code và hội thoại SFT</h2><p class="sub">Số đo kiểm của F-03 / F-07: số đo hình dạng tính trên văn xuôi (đã bỏ công thức / code); cột "còn dính" kỳ vọng gần 0. Hội thoại nhận diện ngôn ngữ theo lượt người dùng; mẫu có câu trả lời không phải tiếng Việt được giữ (có nhãn) để rà.</p><div class="card">{_math_sft_checks(rows, manifest)}</div>
<h2>Phân phối số token</h2><p class="sub">Mỗi nguồn một ô, cùng thang đo. Đoạn sách mục tiêu ~1.024, tối đa 2.048 token (D-10).</p><div class="grid">{hist_words}</div>
<h2>Phân phối điểm chất lượng</h2><div class="grid">{hist_score}</div>
<h2>Dòng bị xoá</h2><p class="sub">Xoá dòng lặp (D-04): B tiêu đề / số trang sách lúc ingest, A dòng lặp trong một văn bản (văn bản có code / bảng được miễn, C-07), C dòng lặp ở nhiều văn bản web.</p><div class="card">{_removed_lines(manifest)}</div>
<h2>Họ trùng (dedup vòng 1)</h2><p class="sub">Theo văn bản: exact / fuzzy (MinHash cả văn bản) / contained (nằm trong văn bản lớn hơn) / chunk_exact (đoạn giống hệt ở văn bản khác). <code>largest_family</code> = số văn bản của họ trùng lớn nhất; họ quá lớn là dấu hiệu nối nhầm.</p><div class="card">{_dedup_summary(manifest)}</div>
<h2>Reason code phổ biến</h2><div class="card">{hbars(sorted(reasons.items(), key=lambda kv: -kv[1])[:14], 'Reason code phổ biến')}</div>
<h2>Bản đồ embedding</h2>{embedding_html or "<p class='sub'>Chưa có (cần stage embed + reduce và <code>uv sync --group viz</code>).</p>"}
<h2>Số liệu</h2><div class="card">{_table(audit)}</div>
<p class="sub">PII trong bản ghi giữ lại: {html.escape(json.dumps(audit['pii'], ensure_ascii=False))}. Quét rò rỉ benchmark (13-gram): {_contamination(audit)}.</p>
<h2>Đọc mẫu thật</h2><p class="sub">Phần quan trọng nhất: mở vài mẫu mỗi nhóm và tự đánh giá band / lý do loại có hợp lý không.</p>
{_samples(rows)}"""
    out = run_dir / "report.html"
    out.write_text(f'<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                   f'<title>Báo cáo pipeline</title><style>{CSS}</style></head><body><main>{body}</main></body></html>', encoding="utf-8")
    return out
