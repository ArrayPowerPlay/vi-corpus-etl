"""
Phép thử "mỏ vàng" theo nhóm font của VJOL (J-42 (4), P2): với mỗi cặp (producer, creator) lấy 3 bài, vẽ 2 trang giữa bài, OCR bằng chuỗi engine VLM rồi so với lớp chữ nhúng. Cần vLLM + GPU (như extract.py);
logic ở vi_corpus/vjol/font_canary.py. CHẠY MẪU (các bài đã có checkpoint), chưa chạy cả kho.

Ghi <data-root>/processed/vjol_font_canary/canary.csv (+ canary.json). Với --labels (labeling_sample.csv đã gán nhãn) thêm hai cột số dòng nhãn sạch / nhãn lỗi font của từng nhóm để chọn ngưỡng.
Chưa có --threshold thì CHỈ đo (không ghi font_groups_bad.json): xem canary.csv, báo lại chủ dự án chọn ngưỡng, rồi chạy lại với --threshold X (dùng lại số đo, thêm --reuse để không OCR lại). extract.py đọc
font_groups_bad.json và OCR cả bài của các nhóm đó.
    uv run --no-sync python scripts/vjol/font_canary.py --data-root data --gpu 0 --serve --vllm-bin $(which vllm) [--labels <labeling_sample.csv>]
    uv run python scripts/vjol/font_canary.py --data-root data --reuse --threshold 0.8      # chỉ ghi font_groups_bad.json từ canary.json đã có
    uv run --no-sync python scripts/vjol/font_canary.py --data-root data --gpu 0 --serve --vllm-bin $(which vllm) --dump-below 0.7   # J-46: đo lại các nhóm < 0,7 và ghi canary_pages.jsonl
Với --dump-below X: chỉ đo lại các nhóm có độ giống trung bình < X trong canary.json đã có (cùng hạt giống nên đúng các bài đó), ghi cặp (lớp chữ, OCR) từng trang vào canary_pages.jsonl
để ĐỌC TAY (độ giống thấp có thể do thứ tự đọc nhiều cột chứ không phải lớp chữ hỏng); KHÔNG ghi đè canary.json / canary.csv.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])  # thay thư mục script bằng gốc repo (xem scripts/vjol/extract.py)

from vi_corpus.common.garbled import load_vocab  # noqa: E402
from vi_corpus.common.ocr_cascade import CascadeOcrGetter  # noqa: E402
from vi_corpus.common.state import acquire_run_lock, read_json, setup_logging  # noqa: E402
from vi_corpus.ocr_bakeoff.cascade import DEFAULT_TIERS, EngineUnavailable, ServerDown, build_tiers  # noqa: E402
from vi_corpus.ocr_bakeoff.engines import DEFAULT_ENGINES, load_engines  # noqa: E402
from vi_corpus.vjol.extract import SYLLABLES_REL, TEXT_DIR  # noqa: E402
from vi_corpus.vjol.font_canary import PAGES_PER_DOC, PER_GROUP, label_reference, read_labels, run_canary, write_canary  # noqa: E402
from vi_corpus.vjol.reocr import pdf_producer_creator  # noqa: E402

OUT_REL = "processed/vjol_font_canary"


def load_docs(data_root: Path) -> list[dict]:
    """Bài có checkpoint (không lỗi, không bị bỏ qua) kèm (producer, creator) đọc lại từ PDF gốc; bài có PDF không còn trên đĩa bị bỏ."""
    docs = []
    for ckpt in sorted((data_root / TEXT_DIR).glob("*.json")):
        done = read_json(ckpt)
        if not done or done.get("error") or done.get("skipped"):
            continue
        path = Path(done["paths"][0])
        path = path if path.is_file() else data_root / done["paths"][0]
        if path.is_file():
            producer, creator = pdf_producer_creator(path)
            docs.append({"sha256": done["sha256"], "path": str(path), "article_id": done.get("article_id"), "producer": producer, "creator": creator})
    return docs


def main() -> int:
    """Đọc tham số, chạy phép thử (hoặc dùng lại canary.json), ghi kết quả. Trả về 1 nếu OCR hỏng hoặc thiếu dữ liệu."""
    p = argparse.ArgumentParser(description="Phép thử nhóm font của VJOL: OCR so với lớp chữ (J-42 (4)).")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")))
    p.add_argument("--per-group", type=int, default=PER_GROUP, help="Số bài mỗi nhóm.")
    p.add_argument("--pages", type=int, default=PAGES_PER_DOC, help="Số trang giữa bài mỗi bài.")
    p.add_argument("--seed", type=int, default=0, help="Hạt giống chọn bài trong nhóm.")
    p.add_argument("--labels", type=Path, default=None, help="labeling_sample.csv đã gán nhãn, để đối chiếu nhóm với nhãn chuẩn.")
    p.add_argument("--threshold", type=float, default=None, help="Nhóm có độ giống trung bình dưới ngưỡng này được ghi vào font_groups_bad.json (không có thì chỉ đo).")
    p.add_argument("--dump-below", type=float, default=None, help="Đo lại nhóm có độ giống < X trong canary.json và ghi canary_pages.jsonl (cặp lớp chữ / OCR từng trang) để đọc tay.")
    p.add_argument("--reuse", action="store_true", help="Dùng lại canary.json đã có, không OCR lại (để đổi --threshold).")
    g = p.add_argument_group("OCR (chuỗi engine VLM, như extract.py)")
    g.add_argument("--config", type=Path, default=DEFAULT_ENGINES)
    g.add_argument("--tiers", default=",".join(DEFAULT_TIERS))
    g.add_argument("--gpu", default=None)
    g.add_argument("--serve", action="store_true")
    g.add_argument("--vllm-bin", default=None)
    g.add_argument("--port", type=int, default=None)
    g.add_argument("--gpu-mem", type=float, default=0.85)
    g.add_argument("--concurrency", type=int, default=32)
    g.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    g.add_argument("--server-timeout", type=float, default=3600)
    args = p.parse_args()
    out_dir = args.data_root / OUT_REL
    log = logging.getLogger("vi_corpus")
    setup_logging(args.data_root / "logs", "vjol_font_canary")
    docs = load_docs(args.data_root)
    if not docs:
        print("Không có checkpoint VJOL nào (chạy extract.py trước)", file=sys.stderr)
        return 1
    if args.reuse:
        groups = json.loads((out_dir / "canary.json").read_text(encoding="utf-8"))
    else:
        details: list[dict] = []
        try:
            tiers = build_tiers([t.strip() for t in args.tiers.split(",") if t.strip()], load_engines(args.config), port=args.port, device=args.device, gpu_mem=args.gpu_mem)
        except ValueError as e:
            p.error(str(e))
        if args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        state = args.data_root / "state" / "vjol_font_canary"
        _lock = acquire_run_lock(state)  # noqa: F841 — giữ khoá
        getter = CascadeOcrGetter(tiers, args.concurrency, serve=args.serve, gpu=args.gpu, vllm_bin=args.vllm_bin, state_dir=state, server_timeout=args.server_timeout,
                                  vocab=load_vocab(args.data_root / SYLLABLES_REL))
        try:
            only = None
            if args.dump_below is not None:
                prev = json.loads((out_dir / "canary.json").read_text(encoding="utf-8"))
                only = [(g["producer"], g["creator"]) for g in prev if g["similarity_mean"] is not None and g["similarity_mean"] < args.dump_below]
                log.info("Đo lại %d nhóm có độ giống < %.2f để lưu cặp lớp chữ / OCR", len(only), args.dump_below)
            groups = run_canary(docs, getter, args.per_group, args.pages, args.seed, only_groups=only, details=details)
        except (ServerDown, EngineUnavailable) as exc:
            log.error("%s. Dừng cả lượt; sửa server / môi trường rồi chạy lại.", exc, exc_info=True)
            return 1
        finally:
            getter.close()
    if args.dump_below is not None and not args.reuse:
        out = out_dir / "canary_pages.jsonl"
        out.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in details), encoding="utf-8")
        print(f"Ghi {len(details)} trang của {len(groups)} nhóm vào {out}")
        return 0
    if args.labels is not None:
        groups = label_reference(groups, docs, read_labels(args.labels))
    write_canary(groups, out_dir, args.threshold)
    measured = [g for g in groups if g["similarity_mean"] is not None]
    print(f"{len(groups)} nhóm (producer, creator), {len(measured)} đo được; thấp nhất:")
    for g in measured[:15]:
        print(f"  {g['similarity_mean']:.3f}  {g['producer']!r} / {g['creator']!r}  ({g['docs_tested']}/{g['docs_total']} bài, {g['pages']} trang)"
              + (f"  nhãn sạch {g['labeled_clean']}, nhãn lỗi font {g['labeled_bad']}, gạch nối thay ư {g['labeled_hyphen']}" if "labeled_clean" in g else ""))
    print(f"Ghi vào {out_dir}" + ("" if args.threshold is not None else " (chưa có --threshold nên chưa ghi font_groups_bad.json)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
