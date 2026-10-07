"""
Bản đồ embedding tương tác (plotly): mỗi cặp (thuật toán giảm chiều, cách tô màu) một file HTML trong <run_dir>/viz/.

Giống packages/visualizer/scatter.py của ViLA (plotly express, mỗi tổ hợp một file, report nhúng bằng iframe). Khác: vẽ
WebGL (px.scatter render_mode="webgl") để mượt với hàng chục nghìn điểm; mỗi file nhúng sẵn plotly.js
(include_plotlyjs=True, ~4-5 MB/file) nên mở riêng lẻ được, không cần mạng và không cần plotly.min.js đi kèm. Tô theo: nguồn, trạng thái (giữ / loại và lý do),
band chất lượng, ngôn ngữ, cụm. Rê chuột vào điểm để đọc đoạn đầu văn bản và reason code.
"""

import html
import logging
from pathlib import Path

logger = logging.getLogger("vi_corpus")

COLOR_BY = {"source_key": "Nguồn", "status": "Trạng thái", "quality_band": "Band chất lượng", "language": "Ngôn ngữ",
            "cluster": "Cụm (HDBSCAN)"}
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]  # thứ tự slot cố định
FIXED = {"status": {"kept": "#2a78d6", "rejected:quality": "#eb6834", "rejected:duplicate": "#1baf7a",
                    "rejected:rights": "#eda100"},
         "quality_band": {"A": "#1c5cab", "B": "#3987e5", "C": "#86b6ef", "D": "#eb6834"}}


def write_scatters(run_dir: Path, rows: list[dict], reduced: list[dict]) -> list[Path]:
    """
    Vẽ các bản đồ embedding và trả về danh sách file HTML đã ghi (rỗng nếu thiếu plotly / pandas).

    Args:
        run_dir: Thư mục kết quả của lần chạy (file ghi vào run_dir/viz/).
        rows:    Mọi bản ghi của lần chạy (đã có status, quality_band, ...).
        reduced: Kết quả reduce_embeddings (doc_id, pca_x, ..., cluster_id).
    """
    try:
        import pandas as pd
        import plotly.express as px
    except ImportError:
        logger.warning("Chưa cài plotly / pandas (uv sync --group viz), bỏ qua bản đồ embedding")
        return []
    by_id = {r["doc_id"]: r for r in rows}
    recs = []
    for d in reduced:
        r = by_id.get(d["doc_id"])
        if r is None:
            continue
        text = (r["text"] or "")[:240].replace("\n", " ")
        recs.append({**d, "source_key": r["source_key"], "status": r["status"], "quality_band": r["quality_band"],
                     "language": r["language"], "cluster": f"cụm {d['cluster_id']}" if d["cluster_id"] >= 0 else "nhiễu",
                     "words": r["word_count"], "reasons": ", ".join(r["reason_codes"]) or "-",
                     "excerpt": html.escape(text) + "…"})
    df = pd.DataFrame(recs)
    out_dir = run_dir / "viz"
    out_dir.mkdir(exist_ok=True)
    written = []
    for algo in ("umap", "pca"):
        if df.empty or df[f"{algo}_x"].isna().all():
            continue
        for col, title in COLOR_BY.items():
            fig = px.scatter(df, x=f"{algo}_x", y=f"{algo}_y", color=col, render_mode="webgl",
                             color_discrete_map=FIXED.get(col), color_discrete_sequence=PALETTE,
                             hover_data={"doc_id": True, "words": True, "reasons": True, "excerpt": True,
                                         f"{algo}_x": False, f"{algo}_y": False},
                             title=f"{algo.upper()} · tô theo {title}", opacity=0.75)
            fig.update_traces(marker={"size": 5})
            fig.update_layout(legend_title_text=title, margin={"l": 20, "r": 20, "t": 50, "b": 20}, height=620)
            path = out_dir / f"scatter-{algo}-{col}.html"
            fig.write_html(path, include_plotlyjs=True, full_html=True)
            written.append(path)
    return written
