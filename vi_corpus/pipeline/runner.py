"""
Runner của pipeline: nối các stage, mỗi stage ghi Parquet riêng vào <run_dir>, có checkpoint theo stage.

    run_dir/
      01_ingest.parquet      lấy mẫu từ raw/ (sách: nguyên cuốn)
      02_prepare.parquet     chuẩn hóa Unicode, xoá dòng lặp (D-04), cắt đoạn theo token (D-10), đếm từ / token (D-01)
      03_language.parquet    + language, lang_score, lang_mix (fastText theo đoạn, D-02)
      04_quality.parquet     + điểm, band, reason code, số đo thô
      05_dedup.parquet       + rights_gate, status, họ trùng (TẤT CẢ bản ghi, kể cả bị loại)
      06_embeddings.parquet  (doc_id, embedding) để vẽ bản đồ (mọi bản ghi, hoặc chỉ bản giữ lại nếu embed_scope=kept)
      07_reduced.parquet     (doc_id, pca, umap, cluster_id)
      clean.parquet, knowledge_units.parquet, cpt_blocks.parquet, audit.json, manifest.json, report.html  (finalize)
    <data_root>/state/dedup_index/<nguồn>/<run>.parquet   kho dấu vân tay dedup vòng 1 (D-09), ghi ở stage dedup

Vân tay stage (G-02): mỗi stage có vân tay = băm(tham số của stage + phiên bản code + vân tay stage trước; ingest thì
thay vân tay stage trước bằng danh sách file đầu vào và kích thước). Vân tay ghi vào manifest["fingerprints"].
Khi chạy lại: stage đã có file và vân tay khớp thì bỏ qua; lệch (đổi tokenizer, ngưỡng, bộ nhúng, dữ liệu đầu vào...)
thì xoá kết quả stage đó cùng các stage sau rồi chạy lại, trừ khi keep_stale=True (cố ý dùng kết quả cũ, có cảnh báo).
Run cũ chưa có vân tay: cảnh báo một lần và dùng kết quả cũ.

force_from xoá từ stage đó trở đi; `until` dừng sau một stage (chạy từng bước). Ba stage language / quality / embed chạy
song song nhiều CPU / GPU nếu truyền `backend` (vi_corpus.pipeline.curator.CuratorBackend; embed tfidf luôn tuần tự vì
phải fit trên toàn corpus); không có backend thì chạy trong tiến trình này. manifest.json lưu cấu hình, vân tay, thời
gian và thống kê từng stage. Các stage còn lại là hàm thuần trên list[dict].
"""

import hashlib
import json
import logging
import time
from pathlib import Path

from vi_corpus.common.registry import SourceSpec
from vi_corpus.common.state import write_json_atomic
from vi_corpus.pipeline import (
    audit,
    chunk,
    contamination,
    dedup,
    embed,
    io,
    knowledge,
    language,
    lines,
    normalize,
    quality,
    reduce,
    spans,
)
from vi_corpus.pipeline.config import RunConfig
from vi_corpus.pipeline.ingest import INGEST_VERSION, ingest, input_files
from vi_corpus.pipeline.tokens import get_counter

logger = logging.getLogger("vi_corpus")

STAGES = ("ingest", "prepare", "language", "quality", "dedup", "embed", "reduce", "finalize")
_FILES = {"ingest": "01_ingest.parquet", "prepare": "02_prepare.parquet", "language": "03_language.parquet",
          "quality": "04_quality.parquet", "dedup": "05_dedup.parquet", "embed": "06_embeddings.parquet",
          "reduce": "07_reduced.parquet"}
_FINAL_FILES = ("clean.parquet", "knowledge_units.parquet", "cpt_blocks.parquet", "audit.json", "report.html")


def _files_of(stage: str) -> list[str]:
    """Các file kết quả của một stage trong run_dir."""
    return list(_FINAL_FILES) if stage == "finalize" else [_FILES[stage]]


def _digest(obj) -> str:
    """Băm ngắn (16 ký tự hex) của một đối tượng JSON được."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


# Trường SourceProfile mà mỗi stage đọc: vân tay của stage chỉ băm các trường này, để chỉnh ngưỡng chất lượng (R-11)
# không làm chạy lại prepare / language (tokenizer, fastText trên mọi bản ghi).
PROFILE_FIELDS = {"ingest": ("chunked",), "prepare": ("chunked", "split_long", "in_doc_line_clean", "cross_line_clean"),
                  "quality": ("min_words", "max_words", "min_alpha", "web_checks", "allowed_langs", "min_lang_score"),
                  "dedup": ("dedup_group",)}


def _profiles(cfg: RunConfig, stage_name: str) -> dict:
    """Các trường hồ sơ nguồn mà stage này đọc, theo từng nguồn (để băm vân tay)."""
    fields = PROFILE_FIELDS.get(stage_name, ())
    return {k: {f: getattr(p, f) for f in fields} for k, p in sorted(cfg.profiles.items())}


def stage_params(name: str, cfg: RunConfig) -> dict:
    """
    Tham số (và phiên bản code) quyết định kết quả của một stage; đổi bất kỳ giá trị nào thì stage đó và các stage sau
    phải chạy lại. Mỗi stage chỉ ghi các tham số nó thật sự đọc (xem PROFILE_FIELDS).
    """
    return {
        "ingest": lambda: {"version": INGEST_VERSION, "mix": cfg.mix, "seed": cfg.seed, "rows_per_file": cfg.rows_per_file,
                           "profiles": _profiles(cfg, "ingest")},
        "prepare": lambda: {"normalizer": normalize.NORMALIZER_VERSION, "lines": lines.LINES_VERSION,
                            "chunk": chunk.CHUNK_VERSION, "minhash": dedup.MINHASH_VERSION, "tokenizer": cfg.tokenizer,
                            "chunk_tokens": [cfg.chunk_target_tokens, cfg.chunk_max_tokens, cfg.chunk_min_tokens],
                            "line": [cfg.line_short_words, cfg.line_repeat_min, cfg.cross_line_min_docs,
                                     cfg.cross_line_min_frac], "mix": cfg.mix, "seed": cfg.seed,
                            "profiles": _profiles(cfg, "prepare")},
        "language": lambda: {"version": language.LANG_VERSION, "spans": spans.SPANS_VERSION, "model": cfg.lang_model,
                             "segments": cfg.lang_max_segments},
        "quality": lambda: {"version": quality.QUALITY_VERSION, "spans": spans.SPANS_VERSION,
                            "profiles": _profiles(cfg, "quality")},
        "dedup": lambda: {"version": dedup.DEDUP_VERSION, "minhash": dedup.MINHASH_VERSION, "keep_bands": cfg.keep_bands,
                          "rights_gate": cfg.rights_gate, "fuzzy": cfg.fuzzy_threshold, "priority": cfg.source_priority,
                          "profiles": _profiles(cfg, "dedup"),
                          "contained": [cfg.contained_frac, cfg.contained_min_paras, cfg.para_min_words]},
        "embed": lambda: {"spec": cfg.embed_spec, "prompt": cfg.embed_prompt, "max_seq": cfg.embed_max_seq,
                          "scope": cfg.embed_scope},
        "reduce": lambda: {"space": cfg.cluster_space, "min_cluster_size": cfg.hdbscan_min_cluster_size,
                           "min_samples": cfg.hdbscan_min_samples, "prefer_gpu": cfg.prefer_gpu},
    }[name]()


def input_inventory(specs: dict[str, SourceSpec], data_root: Path, cfg: RunConfig) -> list:
    """Danh sách [nguồn, đường dẫn tương đối, kích thước] mọi file đầu vào của ingest (vân tay đầu vào)."""
    out = []
    for key in sorted(cfg.mix):
        for path in input_files(specs[key], data_root, cfg.profile(key).chunked):
            rel = str(path.relative_to(data_root)) if path.is_relative_to(data_root) else str(path)
            out.append([key, rel, path.stat().st_size])
    return out


def fingerprints(specs: dict[str, SourceSpec], data_root: Path, cfg: RunConfig) -> dict[str, str]:
    """Vân tay mong đợi của mọi stage (trừ finalize, luôn chạy lại) theo cấu hình và dữ liệu đầu vào hiện tại."""
    out: dict[str, str] = {}
    prev = _digest(input_inventory(specs, data_root, cfg))
    for name in STAGES[:-1]:
        prev = out[name] = _digest({"stage": name, "params": stage_params(name, cfg), "input": prev})
    return out


def _delete_from(run_dir: Path, stage_name: str) -> None:
    """Xoá file kết quả từ stage này trở đi."""
    for name in STAGES[STAGES.index(stage_name):]:
        for f in _files_of(name):
            (run_dir / f).unlink(missing_ok=True)


def check_fingerprints(run_dir: Path, expected: dict[str, str], keep_stale: bool) -> None:
    """
    So vân tay đã lưu với vân tay mong đợi; stage đầu tiên có file mà lệch vân tay thì xoá nó và mọi stage sau
    (trừ khi keep_stale). Stage có file nhưng chưa có vân tay (run cũ): cảnh báo một lần, giữ nguyên.
    """
    stored = io.read_manifest(run_dir).get("fingerprints", {})
    legacy = []
    for name in STAGES[:-1]:
        if not (run_dir / _FILES[name]).exists():
            continue
        if name not in stored:
            legacy.append(name)
        elif stored[name] != expected[name]:
            if keep_stale:
                logger.warning("[%s] cấu hình / đầu vào đã đổi so với kết quả cũ (vân tay %s != %s) nhưng --keep-stale: "
                               "dùng lại kết quả cũ", name, stored[name], expected[name])
                continue
            logger.warning("[%s] cấu hình / đầu vào đã đổi (vân tay %s -> %s): chạy lại từ stage này", name,
                           stored[name], expected[name])
            _delete_from(run_dir, name)
            return
    if legacy:
        logger.warning("Run cũ chưa có vân tay cho stage %s: dùng lại kết quả cũ, KHÔNG kiểm được cấu hình có đổi hay "
                       "không (chạy lại với --force-from <stage> nếu đã đổi tham số)", ", ".join(legacy))


def run_pipeline(specs: dict[str, SourceSpec], data_root: Path, run_dir: Path, cfg: RunConfig,
                 force_from: str | None = None, with_report: bool = True, until: str | None = None,
                 backend=None, keep_stale: bool = False) -> dict:
    """
    Chạy pipeline, trả về manifest của lần chạy.

    Args:
        specs:       Registry nguồn.
        data_root:   Thư mục gốc dữ liệu (raw/, interim/, state/).
        run_dir:     Thư mục kết quả của lần chạy này (tên thư mục là tên lần chạy trong kho dấu vân tay dedup).
        cfg:         Cấu hình chạy.
        force_from:  Tên stage; xoá kết quả từ stage này trở đi và chạy lại.
        with_report: Sinh bản đồ embedding và report.html ở stage finalize.
        until:       Dừng sau stage này (chạy từng bước). None = chạy hết.
        backend:     Bộ chạy song song (CuratorBackend) cho language / quality / embed; None = tuần tự trong tiến trình này.
        keep_stale:  Dùng lại kết quả stage dù vân tay lệch (chỉ cảnh báo).

    Raises:
        ValueError: nếu force_from hoặc until không phải tên stage.
    """
    for name in (force_from, until):
        if name and name not in STAGES:
            raise ValueError(f"{name!r} không phải stage; các stage: {STAGES}")
    last = STAGES.index(until) if until else len(STAGES) - 1
    run_dir.mkdir(parents=True, exist_ok=True)
    if force_from:
        _delete_from(run_dir, force_from)
    expected = fingerprints(specs, data_root, cfg)
    check_fingerprints(run_dir, expected, keep_stale)
    counter = get_counter(cfg.tokenizer)

    def save_fingerprint(name: str) -> None:
        """Ghi vân tay của stage vừa chạy vào manifest."""
        manifest = io.read_manifest(run_dir)
        manifest.setdefault("fingerprints", {})[name] = expected[name]
        write_json_atomic(run_dir / "manifest.json", manifest)

    def stage(name: str, fn) -> list[dict]:
        """Chạy một stage ghi bản ghi nếu chưa có file kết quả, ngược lại đọc lại; ghi thống kê + vân tay vào manifest."""
        path = run_dir / _FILES[name]
        if path.exists():
            logger.info("[%s] đã có %s, bỏ qua", name, path.name)
            return io.read_rows(path)
        t0 = time.time()
        rows, stats = fn()
        save_fingerprint(name)  # ghi trước file kết quả: chết giữa chừng thì file chưa có, stage chạy lại
        n = io.write_rows(path, rows)
        io.update_manifest(run_dir, name, {"rows": n, "seconds": round(time.time() - t0, 1), **stats})
        logger.info("[%s] %d bản ghi (%.1fs) %s", name, n, time.time() - t0,
                    {k: v for k, v in stats.items() if k != "lines"})
        return rows

    def do_ingest():
        """Stage ingest."""
        rows = ingest(specs, data_root, cfg)
        return rows, {"per_source": {k: sum(r["source_key"] == k for r in rows) for k in cfg.mix}}

    def do_prepare(rows):
        """Stage prepare: chuẩn hóa -> xoá dòng lặp trong văn bản -> liên văn bản -> cắt đoạn -> đếm token."""
        page_removed: dict[str, int] = {}
        for r in rows:
            removed = io.loads(r.get("meta")).get("page_lines_removed") or 0
            page_removed[r["source_key"]] = page_removed.get(r["source_key"], 0) + removed
        rows, nstats = normalize.normalize_rows(rows)
        rows, lstats = lines.clean_rows(rows, cfg)
        lstats["page_lines_removed"] = page_removed  # D-04 mục B (clean_pages lúc ingest)
        rows, cstats = chunk.chunk_rows(rows, cfg, counter)
        return rows, {"normalize": nstats, "lines": lstats, "chunk": cstats}

    def do_language(rows):
        """Stage language (song song nếu có backend)."""
        rows = (backend.language(rows, cfg) if backend
                else language.annotate_language(rows, cfg.lang_model, cfg.lang_max_segments))
        return rows, {**language.language_stats(rows), "model": cfg.lang_model}

    def do_quality(rows):
        """Stage quality (song song nếu có backend)."""
        rows = backend.quality(rows, cfg) if backend else quality.annotate_quality(rows, cfg)
        return rows, quality.quality_stats(rows)

    def do_dedup(rows):
        """Stage dedup vòng 1 + ghi kho dấu vân tay (D-09); không ghi kho thì xoá kho cũ của lần chạy (đã hết hiệu lực)."""
        rows, stats = dedup.dedup_rows(rows, cfg)
        if cfg.dedup_index:
            stats["index"] = write_dedup_index(rows, expected["dedup"])
        else:
            drop_dedup_index()
        return rows, stats

    def write_dedup_index(rows, dedup_fp: str | None) -> dict[str, int]:
        """
        Ghi kho dấu vân tay vòng 1 của lần chạy này (mọi nguồn trong mix, kể cả nguồn không còn văn bản nào) và ghi vào
        manifest["dedup_index"] vân tay của 05_dedup.parquet mà kho được dựng từ đó.
        """
        written = dedup.write_index(data_root, dedup.index_rows(rows, cfg, run_dir.name), run_dir.name, cfg.mix)
        io.update_manifest(run_dir, "dedup_index", {"fingerprint": dedup_fp, "sources": written})
        return written

    def drop_dedup_index() -> None:
        """Xoá kho dấu vân tay của lần chạy này và mục manifest["dedup_index"]."""
        n = dedup.remove_index(data_root, run_dir.name)
        manifest = io.read_manifest(run_dir)
        if manifest.pop("dedup_index", None) is not None or n:
            write_json_atomic(run_dir / "manifest.json", manifest)
            logger.info("[dedup] xoá %d file kho dấu vân tay cũ của lần chạy %s", n, run_dir.name)

    if not (run_dir / _FILES["dedup"]).exists():  # kết quả dedup đã bị xoá (đổi tham số, --force-from): kho cũ hết hiệu lực
        drop_dedup_index()
    rows: list[dict] = []
    for name, fn in (("ingest", do_ingest), ("prepare", do_prepare), ("language", do_language),
                     ("quality", do_quality), ("dedup", do_dedup)):
        if STAGES.index(name) > last:
            return io.read_manifest(run_dir)
        rows = stage(name, fn) if name == "ingest" else stage(name, lambda fn=fn, rows=rows: fn(rows))
    manifest = io.read_manifest(run_dir)
    dedup_fp = manifest.get("fingerprints", {}).get("dedup")
    index_info = manifest.get("dedup_index")
    if cfg.dedup_index and (index_info is None or index_info.get("fingerprint") != dedup_fp
                            or not dedup.has_index(data_root, run_dir.name)):
        # kho thiếu hoặc dựng từ 05_dedup cũ (vd dedup từng chạy với --no-dedup-index, run cũ chưa có mục này)
        logger.info("[dedup] kho dấu vân tay chưa khớp 05_dedup.parquet, ghi lại: %s", write_dedup_index(rows, dedup_fp))

    emb_path, red_path = run_dir / _FILES["embed"], run_dir / _FILES["reduce"]
    if cfg.embed_spec and last >= STAGES.index("embed"):
        if emb_path.exists():
            logger.info("[embed] đã có %s, bỏ qua", emb_path.name)
        else:
            t0 = time.time()
            targets = rows if cfg.embed_scope == "all" else [r for r in rows if r["status"] == "kept"]
            items = backend.embed(targets, cfg) if backend and cfg.embed_spec.startswith("hf:") else embed.embed_rows(
                targets, cfg.embed_spec, "auto", cfg.embed_prompt, cfg.embed_batch_size, cfg.embed_max_seq)
            save_fingerprint("embed")
            embed.write_embeddings(emb_path, items)
            io.update_manifest(run_dir, "embed", {"rows": len(items), "spec": cfg.embed_spec, "scope": cfg.embed_scope,
                                                  "seconds": round(time.time() - t0, 1)})
            logger.info("[embed] %d vector (%.1fs)", len(items), time.time() - t0)
    if cfg.embed_spec and last >= STAGES.index("reduce"):
        if red_path.exists():
            logger.info("[reduce] đã có %s, bỏ qua", red_path.name)
        else:
            t0 = time.time()
            ids, matrix = embed.read_embeddings(emb_path)
            reduced, info = reduce.reduce_embeddings(ids, matrix, cfg.prefer_gpu, cfg.cluster_space,
                                                     cfg.hdbscan_min_cluster_size, cfg.hdbscan_min_samples)
            save_fingerprint("reduce")
            reduce.write_reduced(red_path, reduced)
            io.update_manifest(run_dir, "reduce", {"rows": len(reduced), "seconds": round(time.time() - t0, 1),
                                                   "clusters": info["clusters"], "umap": info["umap"], "cluster": info})
            logger.info("[reduce] %d điểm, %d cụm, nhiễu %.1f%% (%.1fs)", len(reduced), info["clusters"],
                        100 * info["noise_rate"], time.time() - t0)
    if last < STAGES.index("finalize"):
        return io.read_manifest(run_dir)
    return finalize(rows, run_dir, cfg, counter, with_report, red_path)


def finalize(rows: list[dict], run_dir: Path, cfg: RunConfig, counter, with_report: bool, red_path: Path) -> dict:
    """
    Stage finalize: (tuỳ chọn) áp verdict dedup vòng 2, ghi clean / knowledge unit / khối CPT, audit, report.

    Luôn chạy lại (rẻ), không có vân tay.
    """
    for f in _FINAL_FILES:
        (run_dir / f).unlink(missing_ok=True)
    stats: dict = {}
    if cfg.global_verdict:
        from vi_corpus.pipeline.dedup_global import apply_verdict
        stats["global_verdict"] = {"path": cfg.global_verdict, **apply_verdict(rows, Path(cfg.global_verdict))}
    kept = [r for r in rows if r["status"] == "kept"]
    n_clean = io.write_rows(run_dir / "clean.parquet", kept)
    units = knowledge.build_units(rows, counter)
    n_units = knowledge.write_units(run_dir / "knowledge_units.parquet", units)
    n_blocks = knowledge.write_blocks(run_dir / "cpt_blocks.parquet", knowledge.pack_cpt_blocks(rows, cfg.cpt_context_tokens))
    bench = contamination.load_benchmarks(Path(cfg.benchmarks_dir) if cfg.benchmarks_dir else None)
    compare = {}
    for spec in cfg.compare_tokenizers:
        compare[spec] = sum(get_counter(spec).count_batch([r["text"] or "" for r in kept]))
    report = audit.audit_rows(rows, cfg.tokenizer, contamination.scan(rows, bench), compare)
    write_json_atomic(run_dir / "audit.json", report)
    io.update_manifest(run_dir, "finalize", {
        "clean_rows": n_clean, "knowledge_units": n_units, "cpt_blocks": n_blocks,
        "candidates": sum(u["unit_type"] == "cpt" for u in units),
        "instructions": sum(u["unit_type"] == "sft" for u in units), **stats})
    io.update_manifest(run_dir, "config", cfg.to_dict())
    logger.info("[finalize] clean=%d, knowledge units=%d, khối CPT=%d, lineage(kept)=%.1f%%", n_clean, n_units, n_blocks,
                100 * report["lineage_rate_kept"])
    manifest = io.read_manifest(run_dir)
    if with_report:
        from vi_corpus.pipeline.report import write_report
        from vi_corpus.pipeline.viz import build_embedding_section
        embedding_html = (build_embedding_section(rows, reduce.read_reduced(red_path),
                                                  manifest.get("reduce", {}).get("cluster"))
                          if red_path.exists() else None)
        write_report(run_dir, rows, report, manifest, embedding_html)
    return manifest
