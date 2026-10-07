"""
Runner của pipeline: nối các stage, mỗi stage ghi Parquet riêng vào <run_dir>, có checkpoint theo stage.

    run_dir/
      01_ingest.parquet      lấy mẫu từ raw/ (stbook: nguyên cuốn)
      02_prepare.parquet     chuẩn hóa Unicode, cắt đoạn sách, lấy mẫu đoạn, đếm từ / token
      03_language.parquet    + language, lang_score
      04_quality.parquet     + điểm, band, reason code, số đo thô
      05_dedup.parquet       + rights_gate, status, họ trùng (TẤT CẢ bản ghi, kể cả bị loại)
      06_embeddings.parquet  (doc_id, embedding) của mọi bản ghi, để vẽ bản đồ
      07_reduced.parquet     (doc_id, pca, umap, cluster_id)
      clean.parquet, knowledge_units.parquet, audit.json, manifest.json, report.html (đã nhúng bản đồ)   (stage finalize)

Stage đã có file thì bỏ qua (chạy lại đúng lệnh cũ để làm tiếp); force_from xoá từ stage đó trở đi; `until` dừng sau một stage
(chạy từng bước: lần lượt until=ingest, prepare, ...). Ba stage language / quality / embed chạy song song nhiều CPU / GPU
nếu truyền `backend` (embed tfidf luôn chạy tuần tự vì phải fit trên toàn corpus) (vi_corpus.pipeline.curator.CuratorBackend); không có backend thì chạy trong tiến trình này.
manifest.json lưu cấu hình, thời gian và thống kê từng stage. Các stage còn lại là hàm thuần trên list[dict].
"""

import logging
import time
from pathlib import Path

from vi_corpus.common.registry import SourceSpec
from vi_corpus.common.state import write_json_atomic
from vi_corpus.pipeline import audit, chunk, dedup, embed, io, knowledge, language, normalize, quality, reduce
from vi_corpus.pipeline.config import RunConfig
from vi_corpus.pipeline.ingest import ingest

logger = logging.getLogger("vi_corpus")

STAGES = ("ingest", "prepare", "language", "quality", "dedup", "embed", "reduce", "finalize")
_FILES = {"ingest": "01_ingest.parquet", "prepare": "02_prepare.parquet", "language": "03_language.parquet",
          "quality": "04_quality.parquet", "dedup": "05_dedup.parquet", "embed": "06_embeddings.parquet",
          "reduce": "07_reduced.parquet"}


def run_pipeline(specs: dict[str, SourceSpec], data_root: Path, run_dir: Path, cfg: RunConfig,
                 force_from: str | None = None, with_report: bool = True, until: str | None = None,
                 backend=None) -> dict:
    """
    Chạy pipeline, trả về manifest của lần chạy.

    Args:
        specs:       Registry nguồn.
        data_root:   Thư mục gốc dữ liệu (raw/, interim/).
        run_dir:     Thư mục kết quả của lần chạy này.
        cfg:         Cấu hình chạy.
        force_from:  Tên stage; xoá kết quả từ stage này trở đi và chạy lại.
        with_report: Sinh bản đồ embedding và report.html ở stage finalize.
        until:       Dừng sau stage này (chạy từng bước). None = chạy hết.
        backend:     Bộ chạy song song (CuratorBackend) cho language / quality / embed; None = tuần tự trong tiến trình này.

    Raises:
        ValueError: nếu force_from hoặc until không phải tên stage.
    """
    for name in (force_from, until):
        if name and name not in STAGES:
            raise ValueError(f"{name!r} không phải stage; các stage: {STAGES}")
    last = STAGES.index(until) if until else len(STAGES) - 1
    run_dir.mkdir(parents=True, exist_ok=True)
    if force_from:
        for stage_name in STAGES[STAGES.index(force_from):]:
            for f in [_FILES.get(stage_name, "clean.parquet")] + (["knowledge_units.parquet", "report.html"] if stage_name == "finalize" else []):
                (run_dir / f).unlink(missing_ok=True)
    counter = normalize.make_token_counter(cfg.tokenizer)

    def stage(name: str, fn) -> list[dict]:
        """Chạy một stage ghi bản ghi nếu chưa có file kết quả, ngược lại đọc lại; ghi thống kê vào manifest."""
        path = run_dir / _FILES[name]
        if path.exists():
            logger.info("[%s] đã có %s, bỏ qua", name, path.name)
            return io.read_rows(path)
        t0 = time.time()
        rows, stats = fn()
        n = io.write_rows(path, rows)
        io.update_manifest(run_dir, name, {"rows": n, "seconds": round(time.time() - t0, 1), **stats})
        logger.info("[%s] %d bản ghi (%.1fs) %s", name, n, time.time() - t0, stats)
        return rows

    def do_ingest():
        """Stage ingest."""
        rows = ingest(specs, data_root, cfg)
        return rows, {"per_source": {k: sum(r["source_key"] == k for r in rows) for k in cfg.mix}}

    def do_prepare(rows):
        """Stage prepare: chuẩn hóa rồi cắt đoạn."""
        rows, nstats = normalize.normalize_rows(rows)
        rows, cstats = chunk.chunk_rows(rows, cfg, counter)
        return rows, {"normalize": nstats, "chunk": cstats}

    def do_language(rows):
        """Stage language (song song nếu có backend)."""
        rows = backend.language(rows) if backend else language.annotate_language(rows)
        return rows, language.language_stats(rows)

    def do_quality(rows):
        """Stage quality (song song nếu có backend)."""
        rows = backend.quality(rows, cfg) if backend else quality.annotate_quality(rows, cfg)
        return rows, quality.quality_stats(rows)

    steps = [("ingest", do_ingest), ("prepare", None), ("language", None), ("quality", None), ("dedup", None)]
    rows: list[dict] = []
    for i, (name, _) in enumerate(steps):
        if i > last:
            return io.read_manifest(run_dir)
        if name == "ingest":
            rows = stage(name, do_ingest)
        elif name == "prepare":
            rows = stage(name, lambda: do_prepare(rows))
        elif name == "language":
            rows = stage(name, lambda: do_language(rows))
        elif name == "quality":
            rows = stage(name, lambda: do_quality(rows))
        else:
            rows = stage(name, lambda: dedup.dedup_rows(rows, cfg))

    emb_path, red_path = run_dir / _FILES["embed"], run_dir / _FILES["reduce"]
    if cfg.embed_spec and last >= STAGES.index("embed"):
        if emb_path.exists():
            logger.info("[embed] đã có %s, bỏ qua", emb_path.name)
        else:
            t0 = time.time()
            items = backend.embed(rows, cfg) if backend and cfg.embed_spec.startswith("hf:") else embed.embed_rows(
                rows, cfg.embed_spec, "auto", cfg.embed_prompt, cfg.embed_batch_size, cfg.embed_max_seq)
            embed.write_embeddings(emb_path, items)
            io.update_manifest(run_dir, "embed", {"rows": len(items), "spec": cfg.embed_spec,
                                                  "seconds": round(time.time() - t0, 1)})
            logger.info("[embed] %d vector (%.1fs)", len(items), time.time() - t0)
    if cfg.embed_spec and last >= STAGES.index("reduce"):
        if red_path.exists():
            logger.info("[reduce] đã có %s, bỏ qua", red_path.name)
        else:
            t0 = time.time()
            ids, matrix = embed.read_embeddings(emb_path)
            reduced = reduce.reduce_embeddings(ids, matrix, cfg.prefer_gpu)
            reduce.write_reduced(red_path, reduced)
            io.update_manifest(run_dir, "reduce", {"rows": len(reduced), "seconds": round(time.time() - t0, 1),
                                                   "clusters": len({d["cluster_id"] for d in reduced} - {-1}),
                                                   "umap": any(d["umap_x"] is not None for d in reduced)})
            logger.info("[reduce] %d điểm (%.1fs)", len(reduced), time.time() - t0)
    if last < STAGES.index("finalize"):
        return io.read_manifest(run_dir)

    kept = [r for r in rows if r["status"] == "kept"]
    n_clean = io.write_rows(run_dir / "clean.parquet", kept)
    n_units = knowledge.write_units(run_dir / "knowledge_units.parquet", knowledge.build_units(rows))
    report = audit.audit_rows(rows)
    write_json_atomic(run_dir / "audit.json", report)
    io.update_manifest(run_dir, "finalize", {"clean_rows": n_clean, "knowledge_units": n_units})
    io.update_manifest(run_dir, "config", cfg.to_dict())
    logger.info("[finalize] clean=%d, knowledge units=%d, lineage(kept)=%.1f%%", n_clean, n_units,
                100 * report["lineage_rate_kept"])
    if with_report:
        from vi_corpus.pipeline.report import write_report
        from vi_corpus.pipeline.viz import build_embedding_section
        embedding_html = build_embedding_section(rows, reduce.read_reduced(red_path)) if red_path.exists() else None
        write_report(run_dir, rows, report, io.read_manifest(run_dir), embedding_html)
    return io.read_manifest(run_dir)
