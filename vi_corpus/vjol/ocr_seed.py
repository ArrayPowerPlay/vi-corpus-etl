"""
Nạp bộ nhớ đệm OCR (vi_corpus.common.ocr_cache) từ các checkpoint trích text ĐÃ CÓ (J-25 Bước 2), để lần trích lại đầu tiên sau khi đổi EXTRACT_VERSION không phải OCR lại các trang đã OCR.

Mỗi trang method "ocr_vlm" trong checkpoint thành một mục cache: {text, method, tier, status[, rotation, rejected_text]}; cờ vẽ lại lấy từ việc trang có "reason" (lỗi font, vẽ lại cả trang) hay không
(trang scan lấy ảnh nhúng). An toàn: nếu chữ ký có xoay thử trang (":rot") thì BỎ QUA các trang mà lần chạy cũ có thể chưa kịp xoay thử (do engine tuần tự baseline đọc, hoặc bị loại), vì checkpoint không
ghi lại là lần chạy cũ đã xoay thử hay chưa; chúng sẽ được OCR lại (rất ít trang). Chữ ký phải khớp chuỗi engine mà lần chạy cũ đã dùng, nếu không cache sẽ chứa kết quả của chuỗi khác.
"""

from pathlib import Path

from vi_corpus.common.ocr_cache import OcrCache
from vi_corpus.common.state import read_json
from vi_corpus.vjol.extract import TEXT_DIR

SERIAL_TIERS = frozenset({"baseline"})  # engine tuần tự: trang do nó đọc là ứng viên xoay thử (CascadePageOcr._needs_rotation_check)
ENTRY_KEYS = ("text", "method", "tier", "status", "rotation", "rejected_text")


def seed_cache(data_root: Path, signature: str, serial_tiers: frozenset[str] = SERIAL_TIERS) -> dict:
    """
    Nạp cache OCR từ mọi checkpoint trong <data_root>/interim/vjol_text/.

    Args:
        data_root:    Thư mục gốc dữ liệu.
        signature:    Chữ ký bộ OCR (vi_corpus.common.ocr_cache.ocr_signature) gắn cho các mục nạp vào.
        serial_tiers: Tên engine tuần tự; trang do chúng đọc, hoặc trang flagged, bị bỏ qua nếu signature có xoay thử.

    Returns:
        {"checkpoints", "pages_seeded", "pages_skipped_rotation", "pdfs_with_cache"}.
    """
    rotate = signature.endswith(":rot")
    stats = {"checkpoints": 0, "pages_seeded": 0, "pages_skipped_rotation": 0, "pdfs_with_cache": 0}
    for path in sorted((data_root / TEXT_DIR).glob("*.json")):
        done = read_json(path)
        if not done or done.get("error") or done.get("skipped"):
            continue
        stats["checkpoints"] += 1
        cache = OcrCache.for_pdf(data_root, done["sha256"], signature)
        added = 0
        for i, pg in enumerate(done["pages"]):
            if pg.get("method") != "ocr_vlm":
                continue
            if rotate and (pg.get("status") == "flagged" or pg.get("tier") in serial_tiers):
                stats["pages_skipped_rotation"] += 1
                continue
            cache.put(i, bool(pg.get("reason")), {k: pg[k] for k in ENTRY_KEYS if k in pg})
            added += 1
        if added:
            cache.save()
            stats["pages_seeded"] += added
            stats["pdfs_with_cache"] += 1
    return stats
