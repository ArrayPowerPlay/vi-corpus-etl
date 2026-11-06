"""
Dòng lệnh trích text dùng chung cho các nguồn bài báo PDF (VJOL, VISTA): scripts/vjol/extract.py và scripts/vista/extract.py chỉ gọi main() với bố cục của nguồn.

Chọn mẫu PDF chia đều theo năm, trích text từng trang (trang scan OCR bằng chuỗi engine VLM, O-20), rồi xuất Parquet.
Đọc <data-root>/<đường dẫn raw của nguồn> (hoặc --root: thư mục năm + metadata/), ghi:
    <layout.text_dir>/<sha256>.json          text từng trang + method / engine / trạng thái (checkpoint theo PDF)
    <layout.out_path>                        bản ghi CORPUS_SCHEMA, mỗi PDF một dòng (để xem; pipeline đọc thẳng checkpoint + metadata)
Có checkpoint theo PDF: bị ngắt thì chạy lại đúng lệnh cũ; tăng --sample-docs rồi chạy lại thì chỉ làm phần mới. Chỉ OCR trang scan
(lần đọc thứ hai); trang có chữ giữ lượt đọc PyMuPDF. Server vLLM chỉ được dựng khi gặp trang scan đầu tiên (--serve).
Nhiều GPU: scripts/<nguồn>/run_extract_vlm.sh (một server + một tiến trình mỗi GPU, PDF chia nhau bằng khoá).

Ví dụ (VJOL; VISTA thay vjol bằng vista):
    uv run python scripts/vjol/extract.py --data-root /duong/dan/data --sample-docs 20 --no-ocr          # thử nhanh, không GPU
    uv run --no-sync python scripts/vjol/extract.py --data-root /duong/dan/data --sample-docs 3000 --seed 0 \
        --gpu 0 --serve --vllm-bin $(which vllm)                                                         # một GPU
    uv run python scripts/vjol/extract.py --data-root /duong/dan/data --status
    uv run python scripts/vjol/extract.py --data-root /duong/dan/data --export-only
"""

import argparse
import json
import logging
import os
from collections.abc import Callable, Iterator
from pathlib import Path

from vi_corpus.common.garbled import load_vocab
from vi_corpus.common.ocr_cascade import CascadeOcrGetter
from vi_corpus.common.parquet_export import export_parquet
from vi_corpus.common.registry import SOURCES
from vi_corpus.common.state import acquire_run_lock, setup_logging
from vi_corpus.ocr_bakeoff.cascade import DEFAULT_TIERS, EngineUnavailable, ServerDown, build_tiers
from vi_corpus.ocr_bakeoff.engines import DEFAULT_ENGINES, load_engines
from vi_corpus.vjol.extract import SYLLABLES_REL, report, run_extract
from vi_corpus.vjol.layout import PaperLayout


def format_report(rep: dict, key: str = "vjol") -> str:
    """Tóm tắt report() thành vài dòng dễ đọc (số PDF, lỗi, trang theo method, trang OCR theo engine / trạng thái, giờ); `key` là tiền tố mỗi dòng."""
    pages = ", ".join(f"{m}={n}" for m, n in sorted(rep["pages"].items())) or "chưa có"
    lines = [(f"[{key}] {rep['pdfs_in_raw']} PDF trong raw; {rep['checkpoints']} đã trích text "
              f"({rep['errors']} lỗi, {rep['skipped']} bỏ qua); trang theo method: {pages}")]
    if rep["ocr_status"]:
        lines.append(f"[{key}] OCR VLM: trạng thái {rep['ocr_status']}, engine được chọn {rep['ocr_tier']}, "
                     f"tỷ lệ flagged {rep['flagged_rate']:.2%}, trang baseline {rep['ocr_tier'].get('baseline', 0)}")
    if rep["garbled_pages"]:
        lines.append(f"[{key}] Lỗi font (J-10): {rep['garbled_docs']} bài, trang theo lý do {rep['garbled_pages']}, tín hiệu {rep['garbled_signals']}")
    if rep["no_text_layer_pages"] or rep["blank_pages"] or rep["empty_pages_no_reason"] or rep["empty_docs"]:
        lines.append(f"[{key}] Trang không có lớp chữ (J-33 B): {rep['no_text_layer_pages']} trang có nội dung được OCR (no_text_layer), {rep['blank_pages']} trang trắng (blank_page), "
                     f"{rep['empty_pages_no_reason']} trang empty chưa ghi lý do (checkpoint cũ); {rep['empty_docs']} bài không có chữ nào")
    if rep["vlm_noise_pages"]:
        lines.append(f"[{key}] {rep['vlm_noise_pages']} trang bị lọc câu rác của VLM (\"There is no text in the image\", J-33 C)")
    if rep["without_vocab"]:
        lines.append(f"[{key}] CẢNH BÁO: {rep['without_vocab']} checkpoint dò chữ hỏng khi thiếu tập âm tiết (syllables.json); thêm file rồi chạy lại đúng lệnh cũ (các checkpoint này tự bị làm lại)")
    if rep["old_version"]:
        lines.append(f"[{key}] {rep['old_version']} checkpoint do bản trích text cũ ghi: chạy lại sẽ làm lại các file này (J-10)")
    if rep["checkpoints"]:
        lines.append(f"[{key}] {rep['pages_per_doc']:.1f} trang/bài; tổng {rep['seconds'] / 3600:.2f} giờ trích text (cộng dồn mọi tiến trình)")
    return "\n".join(lines)


def export(spec, data_root: Path, layout: PaperLayout, iter_records: Callable[..., Iterator[dict]]) -> int:
    """Xuất layout.out_path từ checkpoint (không có cột meta, chỉ để xem). Trả về số bản ghi."""
    out = data_root / layout.out_path
    n = export_parquet(iter_records(spec, data_root, with_meta=False), out)
    logging.getLogger("vi_corpus").info("Đã xuất %d bản ghi ra %s", n, out)
    return n


def main(layout: PaperLayout, iter_records: Callable[..., Iterator[dict]]) -> int:
    """Đọc tham số của nguồn `layout` (iter_records sinh bản ghi để xuất Parquet), rồi in tiến độ (--status), xuất Parquet (--export-only) hoặc trích text + xuất Parquet. Trả về 1 nếu có PDF lỗi hoặc OCR hỏng."""
    p = argparse.ArgumentParser(description=f"Xử lý thô {layout.label}: chọn mẫu theo năm, trích text từng trang, xuất Parquet.")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")),
                   help="Thư mục gốc dữ liệu (mặc định: biến SEA_DATA_ROOT hoặc ./data).")
    p.add_argument("--root", type=Path, default=None, help="Thư mục PDF của nguồn (mặc định <data-root>/<đường dẫn raw trong registry>).")
    p.add_argument("--sample-docs", type=int, default=None,
                   help="Chỉ xử lý N bài chọn ngẫu nhiên chia đều theo năm (mặc định: mọi PDF). Tăng N rồi chạy lại thì chỉ làm phần mới.")
    p.add_argument("--seed", type=int, default=0, help="Hạt giống chọn mẫu (mọi tiến trình phải dùng cùng giá trị).")
    p.add_argument("--limit-files", type=int, default=None, help="Chỉ xử lý N file đầu của danh sách đã chọn (chạy thử).")
    p.add_argument("--retry-skipped", action="store_true", help="Làm lại cả file từng bị bỏ qua.")
    p.add_argument("--no-ocr", action="store_true", help="Không OCR: trang scan để needs_ocr (không cần GPU).")
    p.add_argument("--syllables", type=Path, default=None,
                   help="Tập âm tiết cho tín hiệu âm tiết lạ của bộ dò chữ hỏng (mặc định <data-root>/processed/ocr_bakeoff/syllables.json).")
    p.add_argument("--unaccented", type=Path, default=None,
                   help="Danh sách từ không dấu cho tín hiệu nodiac (mặc định <data-root>/processed/ocr_bakeoff/unaccented.json, sinh bằng scripts/build_unaccented.py; thiếu thì dùng danh sách tối thiểu).")
    p.add_argument("--font-groups", type=Path, default=None,
                   help="Danh sách nhóm font xấu (producer, creator) cho OCR cả bài (mặc định <data-root>/<layout.font_groups_rel>, VJOL: processed/vjol_font_canary/font_groups_bad.json do scripts/vjol/font_canary.py ghi; thiếu thì không nhóm nào, J-42 (4)).")
    p.add_argument("--no-rotate", action="store_true",
                   help="Không xoay thử trang scan nghi bị xoay (J-24); mặc định xoay thử khi có tập âm tiết.")
    p.add_argument("--no-export", action="store_true", help="Không xuất Parquet ở cuối (dùng khi nhiều tiến trình chạy cùng lúc).")
    p.add_argument("--export-only", action="store_true", help="Chỉ xuất Parquet từ checkpoint hiện có rồi thoát.")
    p.add_argument("--status", action="store_true", help="Chỉ in tiến độ rồi thoát.")
    g = p.add_argument_group("OCR trang scan (chuỗi engine VLM, O-20)")
    g.add_argument("--config", type=Path, default=DEFAULT_ENGINES, help="File cấu hình engine.")
    g.add_argument("--tiers", default=",".join(DEFAULT_TIERS),
                   help="Chuỗi engine, cách nhau dấu phẩy (mặc định %(default)s). Các engine VLM phải cùng mô hình.")
    g.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES cho server và engine baseline (vd 0).")
    g.add_argument("--serve", action="store_true", help="Tự dựng / tắt server vLLM (lúc gặp trang scan đầu tiên).")
    g.add_argument("--vllm-bin", default=None, help="Đường dẫn lệnh vllm (môi trường riêng); mặc định 'vllm'.")
    g.add_argument("--port", type=int, default=None, help="Cổng server (mặc định theo cấu hình; mỗi GPU một cổng khác nhau).")
    g.add_argument("--gpu-mem", type=float, default=0.85, help="--gpu-memory-utilization của server (chừa chỗ cho engine baseline).")
    g.add_argument("--concurrency", type=int, default=32, help="Số trang scan gửi đồng thời (mặc định 32, O-22).")
    g.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Thiết bị của engine baseline (mặc định cuda).")
    g.add_argument("--server-timeout", type=float, default=3600, help="Thời gian chờ server sẵn sàng (giây).")
    g.add_argument("--worker-name", default=None, help="Tên tiến trình, chống chạy trùng (mặc định gpu<N> hoặc main).")
    args = p.parse_args()
    spec = SOURCES[layout.key]
    root = args.root or args.data_root / spec.path
    if not root.is_dir() and not args.export_only:
        p.error(f"Không thấy thư mục {root}")

    if args.status:
        print(format_report(report(root, args.data_root, layout), layout.key))
        return 0
    if args.export_only:
        export(spec, args.data_root, layout, iter_records)
        return 0

    tiers = []
    if not args.no_ocr:
        try:
            tiers = build_tiers([t.strip() for t in args.tiers.split(",") if t.strip()], load_engines(args.config),
                                port=args.port, device=args.device, gpu_mem=args.gpu_mem)
        except ValueError as e:
            p.error(str(e))
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    worker = args.worker_name or (f"gpu{args.gpu}" if args.gpu is not None else "main")
    state = args.data_root / "state" / f"{layout.key}_extract" / worker
    _lock = acquire_run_lock(state)  # noqa: F841 — giữ khoá
    log = logging.getLogger("vi_corpus")
    setup_logging(args.data_root / "logs", f"{layout.key}_extract_{worker}")
    vocab = None if args.no_rotate else load_vocab(args.syllables or args.data_root / SYLLABLES_REL)  # J-24: có tập âm tiết thì xoay thử trang nghi ngờ
    if vocab is None and not args.no_rotate and not args.no_ocr:
        log.warning("Không thấy tập âm tiết: không xoay thử trang scan bị xoay 180 / 90 / 270 độ (J-24)")
    getter = CascadeOcrGetter(tiers, args.concurrency, enabled=not args.no_ocr, serve=args.serve, gpu=args.gpu,
                              vllm_bin=args.vllm_bin, state_dir=state, server_timeout=args.server_timeout, vocab=vocab)
    try:
        failed = run_extract(root, args.data_root, getter, ocr_enabled=not args.no_ocr, limit_files=args.limit_files,
                             retry_skipped=args.retry_skipped, sample_docs=args.sample_docs, seed=args.seed,
                             syllables=args.syllables, unaccented_path=args.unaccented, font_groups_path=args.font_groups, layout=layout)
    except (ServerDown, EngineUnavailable) as exc:
        # exc_info: ghi cả chuỗi nguyên nhân (vd DependencyError của paddlex nằm sau RuntimeError), nếu không log chỉ còn dòng bọc ngoài
        log.error("%s. Dừng cả lượt (không loại trang âm thầm); sửa server / môi trường rồi chạy lại đúng lệnh cũ.", exc, exc_info=True)
        return 1
    finally:
        getter.close()
    if not args.no_export:
        export(spec, args.data_root, layout, iter_records)
    log.info("%s", json.dumps(report(root, args.data_root, layout), ensure_ascii=False))
    print(format_report(report(root, args.data_root, layout), layout.key))
    return 1 if failed else 0
