"""
Dựng bộ trang HTML để duyệt bằng mắt kết quả chạy thử OCR nội bộ (scripts/ocr_internal/trial_vjol_vista.py): đặt chữ OCR cạnh PDF gốc.

Đọc `report.csv` và `results/<sha256>.json` của lần chạy thử (<data-root>/processed/ocr_internal_trial/<run-name>/) rồi ghi ra thư mục `review/` cạnh đó:
    review/index.html      một bảng 60 dòng: mỗi dòng một file với các chỉ số (ô đỏ = vượt ngưỡng), phần giải thích từng chỉ số + cách tính ở đầu trang,
                           cột "Kết luận của bạn" (Tốt / Tạm được / Kém + ghi chú, lưu trong trình duyệt) và nút "Xuất CSV" để gửi lại kết quả duyệt;
    review/files/<id>.html chi tiết một file: với mỗi trang được chọn, 3 cột cạnh nhau: ẢNH TRANG PDF GỐC | CHỮ OCR NỘI BỘ | LỚP CHỮ CÓ SẴN của PDF
                           (file 1 lớp không có cột thứ ba), kèm độ trùng từ của riêng trang đó để biết trang nào lệch;
    review/images/*.jpg    ảnh các trang đã vẽ ra từ PDF;
    review.zip             cả thư mục trên nén lại, để tải về máy rồi mở index.html (trên server Jupyter không mở được HTML trực tiếp).

Mặc định mỗi file chỉ vẽ 3 trang rải đều (đầu, giữa, cuối) để thư mục gọn (~10 MB cho 60 file); đổi bằng --pages N (0 = mọi trang).

CÁC THÔNG SỐ (cũng hiện ở đầu index.html; nguồn duy nhất là hằng GLOSSARY bên dưới, ngưỡng lấy từ QUALITY_LIMITS của trial_vjol_vista.py):
    layer 1 / layer 2   PDF 1 lớp (chỉ ảnh scan) / 2 lớp (ảnh + lớp chữ có sẵn). layer2_kind: image_text (lớp chữ nằm trên ảnh trang, tức scan đã OCR
                        từ trước) hoặc text_only (PDF sinh thẳng từ phần mềm soạn thảo, không có ảnh nền).
    trang có chữ        số trang mà OCR trả về có chữ / tổng số trang PDF.
    âm tiết lạ          tỉ lệ từ OCR không có trong tập âm tiết tiếng Việt (syllables.json).
    dấu sai             tỉ lệ từ có >= 2 dấu thanh (âm tiết thật chỉ có tối đa 1).
    ký tự điều khiển    tỉ lệ ký tự điều khiển (\\x00-\\x1f) trên ký tự không phải khoảng trắng.
    dòng lặp            1 - (số dòng khác nhau / số dòng) trên các dòng dài hơn 15 ký tự.
    trùng lớp chữ       hệ số Dice trên túi từ giữa chữ OCR và lớp chữ có sẵn của PDF (chỉ file 2 lớp).
    ít_chữ              tổng ký tự OCR < 200 x số trang.
    (Cách tính chi tiết từng chỉ số ở GLOSSARY.)

Lệnh:
    python scripts/ocr_internal/review_report.py --data-root <root> [--run-name trial] [--pages 3] [--dpi 80] [--explain]
`--explain` chỉ in phần giải thích thông số ra màn hình rồi thoát.
"""

import argparse
import csv
import html
import json
import shutil
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trial_vjol_vista as trial  # noqa: E402  (dùng lại ngưỡng và hàm đo của lần chạy thử)

from vi_corpus.vjol.font_canary import word_overlap  # noqa: E402

LIM = trial.QUALITY_LIMITS

# Mỗi thông số: (khoá cột trong report.csv, tên hiển thị, ý nghĩa, cách tính, ngưỡng cờ hoặc "").
GLOSSARY: list[dict] = [
    {"key": "layer", "name": "Loại PDF (layer)", "meaning": "PDF 1 lớp = chỉ có ảnh scan, không có chữ bên trong. PDF 2 lớp = có ảnh trang và một lớp chữ có sẵn.",
     "how": "Dò tối đa 5 trang rải đều mỗi PDF. Trang 'có chữ' khi lớp chữ có >= 30 ký tự; trang 'có ảnh nền' khi có ảnh phủ >= 60% diện tích trang. "
            "Không trang nào có chữ mà có ảnh nền -> 1 lớp. >= 60% trang dò có chữ -> 2 lớp. layer2_kind = image_text nếu >= 60% trang dò còn có ảnh nền "
            "(scan đã OCR từ trước), ngược lại text_only (PDF sinh từ phần mềm soạn thảo).", "limit": ""},
    {"key": "page_coverage", "name": "Trang có chữ", "meaning": "Tỉ lệ trang mà OCR trả về có chữ. 1.0 = trang nào cũng có chữ; thấp = OCR bỏ sót trang (hoặc nhiều trang trắng thật).",
     "how": "Số trang có chữ trong kết quả OCR (đếm theo page_id) chia tổng số trang của PDF.", "limit": f"cờ 'thiếu_trang' khi < {LIM['min_page_coverage']}"},
    {"key": "unknown_rate", "name": "Âm tiết lạ", "meaning": "Tỉ lệ từ OCR đọc ra không phải âm tiết tiếng Việt thật (ví dụ 'hieọu', 'Baỷng'). Càng thấp càng tốt. "
     "Sách tiếng Anh hay nhiều thuật ngữ sẽ cao một cách hợp lệ.",
     "how": "Tách các từ chữ Latinh (viết thường, chuẩn hoá NFC; từ có chữ Hán, Kirin... bỏ qua), đếm từ không có trong tập âm tiết "
            "processed/ocr_bakeoff/syllables.json (rút từ SEA-PILE), chia tổng số từ Latinh. Thiếu file đó thì cột này trống.",
     "limit": f"cờ 'âm_tiết_lạ' khi > {LIM['max_unknown_rate']}"},
    {"key": "garbled", "name": "Dấu sai", "meaning": "Tỉ lệ từ có từ 2 dấu thanh trở lên. Một âm tiết tiếng Việt thật chỉ có tối đa 1 dấu thanh, nên từ như 'Baỷng' là chữ sai "
     "(lỗi font hoặc OCR đọc nhầm dấu). 0.0 là tốt.",
     "how": "Trong các từ >= 2 chữ cái: đếm từ có >= 2 dấu thanh (huyền, sắc, ngã, hỏi, nặng; không tính dấu mũ, trăng, râu) chia tổng số từ.",
     "limit": f"cờ 'dấu_sai' khi > {LIM['max_garbled']}"},
    {"key": "control", "name": "Ký tự điều khiển", "meaning": "Tỉ lệ ký tự điều khiển lạ lẫn trong chữ (dấu hiệu lỗi mã hoá, chữ bị cắt vụn). Bình thường là 0.",
     "how": "Số ký tự mã \\x00-\\x1f (trừ xuống dòng, tab) chia số ký tự không phải khoảng trắng.", "limit": f"cờ 'ký_tự_điều_khiển' khi > {LIM['max_control']}"},
    {"key": "dup_lines", "name": "Dòng lặp", "meaning": "Tỉ lệ dòng bị lặp y hệt; cao thường là máy OCR bị kẹt và lặp vòng một đoạn. "
     "Văn bản thật cũng có ít dòng lặp (tiêu đề chạy, đầu trang) nên không cần về 0.",
     "how": "Lấy các dòng dài hơn 15 ký tự rồi tính 1 - (số dòng khác nhau / số dòng).", "limit": f"cờ 'dòng_lặp' khi > {LIM['max_dup_lines']}"},
    {"key": "overlap_vs_text_layer", "name": "Trùng lớp chữ", "meaning": "Mức giống nhau giữa chữ OCR nội bộ và lớp chữ có sẵn trong PDF (chỉ file 2 lớp). "
     "1.0 = cùng từ; thấp = hai bên đọc ra khác nhau. Thấp KHÔNG tự nhiên có nghĩa OCR sai: lớp chữ cũ có thể là bên hỏng (font sai, mất dấu) - hãy nhìn cột chữ.",
     "how": "Hệ số Dice trên túi từ: viết thường cả hai văn bản, tách từ (chuỗi chữ-số), tính 2 x số từ chung (tính cả lần lặp) / (số từ OCR + số từ lớp chữ). "
            "Không xét thứ tự từ nên đọc khác thứ tự cột không bị phạt. Trong trang chi tiết, mỗi trang có số riêng.",
     "limit": f"cờ 'lệch_lớp_chữ' khi < {LIM['min_overlap']}"},
    {"key": "flag_low_chars", "name": "ít_chữ", "meaning": "OCR trả về quá ít chữ so với số trang (trang trắng, ảnh, hoặc OCR bỏ sót).",
     "how": "Tổng số ký tự chữ OCR < 200 x số trang PDF.", "limit": ""},
    {"key": "elapsed_s", "name": "Thời gian, trang tính sản lượng", "meaning": "elapsed_s: giây từ lúc upload đến lúc có kết quả. billed_pages: số trang API tính vào hạn mức (num_of_pages). "
     "session: mã phiên, đưa cho VNPT tra log khi báo lỗi.", "how": "Lấy nguyên từ phản hồi API.", "limit": ""},
    {"key": "verdict", "name": "Kết luận chung (report.md)", "meaning": "'OK' khi tỉ lệ file lỗi API và tỉ lệ file bị cờ chất lượng đều dưới ngưỡng.",
     "how": "OK khi file lỗi API <= " + f"{LIM['max_error_rate']:.0%} và file ok-nhưng-bị-cờ <= {LIM['max_flagged_rate']:.0%}; không thì KHÔNG ĐẠT. "
            "Mọi ngưỡng là ngưỡng THỬ, chưa chỉnh trên dữ liệu thật: nhìn mắt (trang này) mới là căn cứ cuối.", "limit": ""},
]

CSS = """
:root{--bg:#fafafa;--fg:#1c1c1c;--muted:#666;--line:#ddd;--card:#fff;--bad:#fde2e2;--badfg:#9b1c1c;--ok:#e3f6e8;--link:#0b5cad}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#e8e8e8;--muted:#9a9a9a;--line:#333;--card:#1f1f1f;--bad:#4a1f1f;--badfg:#ffb4b4;--ok:#1d3a27;--link:#7db7ff}}
*{box-sizing:border-box}body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}
a{color:var(--link)}h1{font-size:20px;margin:0 0 8px}h2{font-size:16px;margin:20px 0 8px}
table{border-collapse:collapse;width:100%;background:var(--card)}th,td{border:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}
th{position:sticky;top:0;background:var(--card)}td.bad{background:var(--bad);color:var(--badfg);font-weight:600}td.ok{background:var(--ok)}
.wrap{overflow-x:auto}.muted{color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
@media (max-width:900px){.grid{grid-template-columns:1fr}}
.col{background:var(--card);border:1px solid var(--line);padding:8px;min-width:0}.col h3{margin:0 0 6px;font-size:13px;color:var(--muted)}
.col img{max-width:100%;height:auto;display:block}pre{white-space:pre-wrap;word-break:break-word;margin:0;font:13px/1.45 ui-monospace,monospace;max-height:80vh;overflow:auto}
.page{margin:16px 0;padding-bottom:8px;border-bottom:2px solid var(--line)}.pill{display:inline-block;padding:0 8px;border-radius:10px;background:var(--bad);color:var(--badfg);margin-right:4px}
details{background:var(--card);border:1px solid var(--line);padding:8px 12px;margin:8px 0}dt{font-weight:600;margin-top:10px}dd{margin:2px 0 0 0}
button,select,input{font:inherit}input.note{width:14em}
"""


def explain_text() -> str:
    """Phần giải thích thông số ở dạng chữ thường (cho --explain)."""
    out = []
    for g in GLOSSARY:
        out.append(f"{g['name']}\n  Ý nghĩa : {g['meaning']}\n  Cách tính: {g['how']}" + (f"\n  Ngưỡng  : {g['limit']}" if g["limit"] else ""))
    return "\n\n".join(out)


def glossary_html() -> str:
    """Phần giải thích thông số ở dạng HTML (danh sách định nghĩa), nằm đầu index.html."""
    items = []
    for g in GLOSSARY:
        lim = f"<dd class='muted'>Ngưỡng: {html.escape(g['limit'])}</dd>" if g["limit"] else ""
        items.append(f"<dt>{html.escape(g['name'])}</dt><dd>{html.escape(g['meaning'])}</dd><dd class='muted'>Cách tính: {html.escape(g['how'])}</dd>{lim}")
    return "<details open><summary><b>Các thông số là gì, tính thế nào</b></summary><dl>" + "".join(items) + "</dl></details>"


def is_bad(key: str, value: float | None) -> bool:
    """Giá trị của một chỉ số có vượt ngưỡng cờ không (None = không đo được, không tô)."""
    if value is None:
        return False
    if key == "page_coverage":
        return value < LIM["min_page_coverage"]
    if key == "overlap_vs_text_layer":
        return value < LIM["min_overlap"]
    limit = {"unknown_rate": "max_unknown_rate", "garbled": "max_garbled", "control": "max_control", "dup_lines": "max_dup_lines"}[key]
    return value > LIM[limit]


def num(s: str) -> float | None:
    """Đổi ô CSV thành số; ô trống thành None."""
    return float(s) if s not in ("", None) else None


def pick_pages(total: int, k: int) -> list[int]:
    """Chỉ số trang (từ 0) cần vẽ: k trang rải đều gồm trang đầu và trang cuối; k <= 0 hoặc k >= total thì mọi trang."""
    if k <= 0 or k >= total:
        return list(range(total))
    if k == 1:
        return [0]
    return sorted({round(i * (total - 1) / (k - 1)) for i in range(k)})


def render_file_page(row: dict, rec: dict, out_dir: Path, pages_wanted: int, dpi: int, file_id: str) -> str:
    """
    Dựng và ghi files/<id>.html cho một file; trả về đường dẫn tương đối của nó từ index.html.

    Với mỗi trang chọn: vẽ ảnh trang PDF ra images/<id>_p<n>.jpg, cột OCR lấy `ocr_pages[n]` (n tính từ 1 theo page_id của API), cột lớp chữ lấy PyMuPDF của chính trang đó
    (chỉ file 2 lớp). Exporter md không tách được trang: cả chữ OCR hiện một khối dưới các ảnh.
    """
    e = html.escape
    pdf_path = Path(rec["path"])
    parts = [f"<!doctype html><html lang='vi'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>{e(row['rel'])}</title><style>{CSS}</style></head><body>",
             f"<p><a href='../index.html'>&larr; Về bảng</a></p><h1>{e(row['source'])}/{e(row['rel'])}</h1>",
             f"<p>Layer {e(row['layer'])} {e(row['layer2_kind'])} &middot; {e(row['pages'])} trang &middot; {e(row['size_mb'])} MB &middot; "
             f"{e(row['elapsed_s'])}s &middot; {e(row['billed_pages'])} trang tính sản lượng &middot; session <code>{e(row['session_id'])}</code></p>"]
    parts.append("<p>" + (" ".join(f"<span class='pill'>{e(f)}</span>" for f in row["flags"].split()) or "<span class='muted'>không có cờ</span>") + "</p>")
    cells = "".join(f"<td class='{'bad' if is_bad(g['key'], num(row[g['key']])) else ''}'>{e(g['name'])}: {e(row[g['key']] or '-')}</td>"
                    for g in GLOSSARY if g["key"] in ("page_coverage", "unknown_rate", "garbled", "control", "dup_lines", "overlap_vs_text_layer"))
    parts.append(f"<div class='wrap'><table><tr>{cells}</tr></table></div>")

    ocr_pages: dict = rec.get("ocr_pages") or {}
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:  # noqa: BLE001  - PDF mất / hỏng thì vẫn cho xem chữ OCR
        doc = None
        parts.append(f"<p class='pill'>Không mở được PDF gốc ({e(str(exc))})</p>")
    total = doc.page_count if doc else int(row["pages"])
    for i in pick_pages(total, pages_wanted):
        n = i + 1
        ocr = ocr_pages.get(str(n), "")
        layer_txt = doc[i].get_text("text") if (doc and row["layer"] == "2") else None
        img = ""
        if doc:
            name = f"{file_id}_p{n}.jpg"
            (out_dir / "images" / name).write_bytes(doc[i].get_pixmap(dpi=dpi).tobytes("jpg", jpg_quality=70))
            img = f"<img loading='lazy' src='../images/{name}' alt='trang {n}'>"
        badge = ""
        if layer_txt is not None and ocr_pages:
            ov = word_overlap(ocr, layer_txt)
            badge = f" <span class='{'pill' if ov < LIM['min_overlap'] else 'muted'}'>trùng lớp chữ của trang: {ov:.2f}</span>"
        parts.append(f"<div class='page'><h2>Trang {n} / {total}{badge}</h2><div class='grid'><div class='col'><h3>PDF GỐC (ảnh trang)</h3>{img}</div>"
                     f"<div class='col'><h3>CHỮ OCR NỘI BỘ ({len(ocr)} ký tự)</h3><pre>{e(ocr) if ocr_pages else '(exporter md: xem khối chữ OCR cả file ở cuối trang)'}</pre></div>")
        if layer_txt is not None:
            parts.append(f"<div class='col'><h3>LỚP CHỮ CÓ SẴN trong PDF ({len(layer_txt.strip())} ký tự)</h3><pre>{e(layer_txt)}</pre></div>")
        else:
            parts.append("<div class='col'><h3>LỚP CHỮ CÓ SẴN</h3><p class='muted'>PDF 1 lớp: không có lớp chữ, chỉ so được bằng mắt với ảnh.</p></div>")
        parts.append("</div></div>")
    if not ocr_pages:
        parts.append(f"<h2>Chữ OCR cả file</h2><div class='col'><pre>{e(rec.get('text', ''))}</pre></div>")
    parts.append("</body></html>")
    (out_dir / "files" / f"{file_id}.html").write_text("".join(parts), encoding="utf-8")
    return f"files/{file_id}.html"


INDEX_JS = """
const KEY=id=>'rv:'+id;
function load(){document.querySelectorAll('tr[data-id]').forEach(tr=>{try{const v=JSON.parse(localStorage.getItem(KEY(tr.dataset.id))||'{}');
 tr.querySelector('select').value=v.v||'';tr.querySelector('input').value=v.n||'';}catch(e){}});}
function save(tr){try{localStorage.setItem(KEY(tr.dataset.id),JSON.stringify({v:tr.querySelector('select').value,n:tr.querySelector('input').value}));}catch(e){}}
document.addEventListener('input',ev=>{const tr=ev.target.closest('tr[data-id]');if(tr)save(tr);});
function q(s){return '"'+String(s).replace(/"/g,'""')+'"';}
function exportCsv(){const rows=[['nguon','file','layer','co','ket_luan','ghi_chu']];
 document.querySelectorAll('tr[data-id]').forEach(tr=>rows.push([tr.dataset.src,tr.dataset.rel,tr.dataset.layer,tr.dataset.flags,tr.querySelector('select').value,tr.querySelector('input').value]));
 const blob=new Blob(['\\ufeff'+rows.map(r=>r.map(q).join(',')).join('\\n')],{type:'text/csv;charset=utf-8'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='ket_qua_duyet_ocr.csv';a.click();}
load();
"""


def render_index(rows: list[dict], links: dict[str, str], summary: dict) -> str:
    """
    Dựng index.html: kết luận chung, phần giải thích thông số, bảng mọi file (lỗi trước, rồi file bị cờ theo độ trùng tăng dần, rồi file còn lại) và nút xuất CSV.
    """
    e = html.escape
    order = sorted(rows, key=lambda r: (0 if r["status"] == "error" else 1 if r["flags"] else 2, num(r["overlap_vs_text_layer"]) if r["overlap_vs_text_layer"] else 9, r["rel"]))
    head = "".join(f"<th>{h}</th>" for h in ("#", "Nguồn", "Loại", "File", "Trang", "Cờ", "Trang có chữ", "Âm tiết lạ", "Dấu sai", "Ký tự ĐK", "Dòng lặp",
                                             "Trùng lớp chữ", "Kết luận của bạn", "Ghi chú"))
    body = []
    for i, r in enumerate(order, 1):
        sha = r["_sha"]
        cells = "".join(f"<td class='{'bad' if is_bad(k, num(r[k])) else ''}'>{e(r[k] or '-')}</td>"
                        for k in ("page_coverage", "unknown_rate", "garbled", "control", "dup_lines", "overlap_vs_text_layer"))
        name = f"<a href='{links[sha]}'>{e(r['rel'])}</a>" if sha in links else e(r["rel"])
        flags = e(r["flags"]) if r["status"] == "ok" else f"<b>{e(r['status'])}</b> {e(r['error'])}"
        body.append(f"<tr data-id='{sha[:10]}' data-src='{e(r['source'])}' data-rel='{e(r['rel'])}' data-layer='{e(r['layer'])}' data-flags='{e(r['flags'])}'>"
                    f"<td>{i}</td><td>{e(r['source'])}</td><td>{e(r['layer'])} {e(r['layer2_kind'])}</td><td>{name}</td><td>{e(r['pages'])}</td><td>{flags}</td>{cells}"
                    "<td><select><option value=''>-</option><option>Tốt</option><option>Tạm được</option><option>Kém</option></select></td>"
                    "<td><input class='note' type='text'></td></tr>")
    verdict = f"<p><b>Kết luận tự động: {e(summary['verdict'])}</b> &middot; {summary['files_run']}/{summary['files_total']} file đã chạy, " \
              f"{summary['errors']} lỗi API, {summary['flagged']} file bị cờ, {summary['total_billed_pages']} trang tính sản lượng. " \
              f"{e('; '.join(summary['reasons']))}</p>"
    return (f"<!doctype html><html lang='vi'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Duyệt OCR nội bộ</title><style>{CSS}</style></head><body><h1>Duyệt kết quả OCR nội bộ (VJOL + VISTA)</h1>{verdict}"
            f"<p>Bấm tên file để xem ảnh PDF gốc cạnh chữ OCR. Ô <span class='pill'>đỏ</span> = vượt ngưỡng thử. Chọn kết luận từng file rồi bấm "
            f"<button onclick='exportCsv()'>Xuất CSV</button> (kết luận lưu trong trình duyệt này).</p>{glossary_html()}"
            f"<h2>Danh sách file</h2><div class='wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
            f"<script>{INDEX_JS}</script></body></html>")


def main(argv: list[str] | None = None) -> int:
    """Đọc report.csv + results của lần chạy thử, ghi review/ và review.zip. Trả mã thoát 0; 2 nếu chưa có kết quả."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", default="./data", help="gốc dữ liệu (chứa processed/ocr_internal_trial/)")
    ap.add_argument("--run-name", default="trial", help="tên lần chạy thử (khớp --run-name của trial_vjol_vista.py)")
    ap.add_argument("--pages", type=int, default=3, help="số trang vẽ mỗi file, rải đều (0 = mọi trang)")
    ap.add_argument("--dpi", type=int, default=80, help="độ phân giải ảnh trang")
    ap.add_argument("--explain", action="store_true", help="chỉ in giải thích thông số rồi thoát")
    args = ap.parse_args(argv)
    if args.explain:
        print(explain_text())
        return 0

    run_dir = Path(args.data_root).resolve() / "processed" / "ocr_internal_trial" / args.run_name
    report_csv, report_json = run_dir / "report.csv", run_dir / "report.json"
    if not report_csv.exists():
        print(f"Chưa có {report_csv}; chạy trial_vjol_vista.py (hoặc thêm --report-only) trước.", file=sys.stderr)
        return 2
    rows = list(csv.DictReader(open(report_csv, encoding="utf-8")))
    summary = json.loads(report_json.read_text(encoding="utf-8"))
    sample = {(e["source"], e["rel"]): e["sha256"] for e in json.loads((run_dir / "sample.json").read_text(encoding="utf-8"))["sample"]}

    out_dir = run_dir / "review"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "files").mkdir(parents=True)
    (out_dir / "images").mkdir()
    links: dict[str, str] = {}
    for r in rows:
        sha = sample[(r["source"], r["rel"])]
        r["_sha"] = sha
        res_path = run_dir / "results" / f"{sha}.json"
        if r["status"] != "ok" or not res_path.exists():
            continue
        links[sha] = render_file_page(r, json.loads(res_path.read_text(encoding="utf-8")), out_dir, args.pages, args.dpi, sha[:10])
    (out_dir / "index.html").write_text(render_index(rows, links, summary), encoding="utf-8")
    zip_path = shutil.make_archive(str(run_dir / "review"), "zip", out_dir)
    size_mb = Path(zip_path).stat().st_size / 1048576
    print(f"Đã dựng {len(links)} trang chi tiết ở {out_dir}\nMở: {out_dir / 'index.html'}\nĐể tải về: {zip_path} ({size_mb:.1f} MB), giải nén rồi mở index.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
