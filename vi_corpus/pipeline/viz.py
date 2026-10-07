"""
Bản đồ embedding tương tác (plotly) nhúng thẳng vào report.html: chỉ cần tải một file report.html là xem được.

Khác bản cũ (mỗi tổ hợp một file viz/*.html, nhúng iframe): plotly.js được chèn MỘT lần, dữ liệu điểm lưu MỘT lần dưới
dạng JSON, trình duyệt vẽ lại khi bấm nút chọn (thuật toán giảm chiều x cách tô màu). Nên file nhỏ (~vài MB thay vì
hàng chục MB), không cần mạng, không cần thư mục đi kèm, không dùng iframe nên không vướng lỗi 403 của Jupyter.
Vẽ WebGL (scattergl) để mượt với hàng chục nghìn điểm. Tô theo: nguồn, trạng thái, band chất lượng, ngôn ngữ, cụm.
Rê chuột vào điểm để đọc đoạn đầu văn bản và reason code.
"""

import html
import json
import logging

logger = logging.getLogger("vi_corpus")

COLOR_BY = {"source_key": "Nguồn", "status": "Trạng thái", "quality_band": "Band chất lượng", "language": "Ngôn ngữ",
            "cluster": "Cụm (HDBSCAN)"}
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]  # thứ tự slot cố định
FIXED = {"status": {"kept": "#2a78d6", "rejected:quality": "#eb6834", "rejected:duplicate": "#1baf7a",
                    "rejected:rights": "#eda100"},
         "quality_band": {"A": "#1c5cab", "B": "#3987e5", "C": "#86b6ef", "D": "#eb6834"}}
ALGOS = ("umap", "pca")

_JS = """
(function(){
var D=JSON.parse(document.getElementById('emb-data').textContent),el=document.getElementById('emb-plot'),cur=null;
function draw(algo,col){
  var cats=D.cats[col],groups={},order=[];
  cats.forEach(function(c,i){if(D[algo+'x'][i]===null)return;if(!(c in groups)){groups[c]=[];order.push(c)}groups[c].push(i)});
  var fixed=D.fixed[col]||{};order.sort();var pal=D.palette,traces=order.map(function(c,k){
    var idx=groups[c];return{type:'scattergl',mode:'markers',name:c+' ('+idx.length+')',
      x:idx.map(function(i){return D[algo+'x'][i]}),y:idx.map(function(i){return D[algo+'y'][i]}),
      text:idx.map(function(i){return D.hover[i]}),hovertemplate:'%{text}<extra></extra>',
      marker:{size:5,opacity:0.75,color:fixed[c]||pal[k%pal.length]}}});
  var ink=getComputedStyle(document.body).color;
  Plotly.react(el,traces,{title:algo.toUpperCase()+' · tô theo '+D.titles[col],height:620,
    margin:{l:20,r:20,t:50,b:20},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:ink},
    xaxis:{showgrid:false,zeroline:false,showticklabels:false},yaxis:{showgrid:false,zeroline:false,showticklabels:false},
    legend:{title:{text:D.titles[col]}}},{responsive:true});
  cur=algo+'-'+col;
  document.querySelectorAll('#emb-btns button').forEach(function(b){b.style.fontWeight=b.dataset.k===cur?'700':'400'});
}
document.querySelectorAll('#emb-btns button').forEach(function(b){
  b.onclick=function(){var p=b.dataset.k.split('-');draw(p[0],p.slice(1).join('-'))}});
var first=document.querySelector('#emb-btns button');if(first)first.click();
})();
"""


def build_embedding_section(rows: list[dict], reduced: list[dict]) -> str:
    """
    Dựng khối HTML tự đủ (nút chọn + biểu đồ + plotly.js + dữ liệu) để chèn vào report.html.

    Args:
        rows:    Mọi bản ghi của lần chạy (đã có status, quality_band, ...).
        reduced: Kết quả reduce_embeddings (doc_id, pca_x, ..., cluster_id).

    Returns:
        Chuỗi HTML; hoặc đoạn ghi chú nếu thiếu plotly hoặc không có điểm nào.
    """
    try:
        from plotly.offline import get_plotlyjs
    except ImportError:
        logger.warning("Chưa cài plotly (uv sync --group viz), bỏ qua bản đồ embedding")
        return "<p class='sub'>Chưa có (cần stage embed + reduce và <code>uv sync --group viz</code>).</p>"
    by_id = {r["doc_id"]: r for r in rows}
    pts = [(d, by_id[d["doc_id"]]) for d in reduced if d["doc_id"] in by_id]
    if not pts:
        return "<p class='sub'>Chưa có (cần stage embed + reduce).</p>"

    def coord(d: dict, key: str) -> float | None:
        """Toạ độ làm tròn 4 chữ số (đủ cho bản đồ, nhẹ file); None nếu thiếu (không có UMAP)."""
        return None if d[key] is None else round(d[key], 4)

    data = {"titles": COLOR_BY, "fixed": FIXED, "palette": PALETTE,
            "cats": {"source_key": [r["source_key"] for _, r in pts], "status": [r["status"] for _, r in pts],
                     "quality_band": [r["quality_band"] for _, r in pts], "language": [r["language"] for _, r in pts],
                     "cluster": [f"cụm {d['cluster_id']}" if d["cluster_id"] >= 0 else "nhiễu" for d, _ in pts]},
            "hover": [f"<b>{html.escape(r['doc_id'])}</b> · {r['word_count']} từ<br>mã: "
                      f"{html.escape(', '.join(r['reason_codes']) or '-')}<br>"
                      f"{html.escape((r['text'] or '')[:240].replace(chr(10), ' '))}…" for _, r in pts]}
    for algo in ALGOS:
        data[f"{algo}x"] = [coord(d, f"{algo}_x") for d, _ in pts]
        data[f"{algo}y"] = [coord(d, f"{algo}_y") for d, _ in pts]
    algos = [a for a in ALGOS if any(v is not None for v in data[f"{a}x"])]
    buttons = "".join(f'<button data-k="{a}-{c}">{a}-{c}</button>' for a in algos for c in COLOR_BY)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return (f'<p class="sub">Mỗi điểm là một mẫu; rê chuột để đọc đoạn đầu và reason code. Gồm cả mẫu bị loại '
            f'(tô theo trạng thái).</p><div class="legend" id="emb-btns">{buttons}</div><div id="emb-plot"></div>'
            f'<script>{get_plotlyjs()}</script><script type="application/json" id="emb-data">{payload}</script>'
            f'<script>{_JS}</script>')
