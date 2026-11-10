"""
Bộ nhớ đệm kết quả OCR theo trang (J-25 Bước 2): đổi tín hiệu dò chữ hỏng hay làm sạch chỉ phải OCR thêm các trang MỚI bị gắn cờ, không OCR lại cả PDF.

Trước đây mỗi lần đổi EXTRACT_VERSION là làm lại mọi checkpoint, kể cả OCR lại các trang scan (~3 giờ đồng hồ trên 4 GPU cho 3.000 bài). Nay kết quả OCR của từng
trang (đúng dict mà CascadePageOcr.ocr_pages trả về: text, method, tier, status, rotation, rejected_text) được lưu cạnh checkpoint:

    interim/vjol_ocr_cache/<2 ký tự đầu sha256>/<sha256>.json   {"version", "signature", "pages": {"<chỉ số trang>:<e|r>": {...}}}

- Khoá trang gồm cờ vẽ lại: "e" = lấy ảnh nhúng (trang scan), "r" = vẽ lại cả trang (trang lỗi font); hai cách cho ảnh khác nhau nên kết quả khác nhau.
- `signature` mô tả bộ OCR đã tạo ra kết quả (chuỗi engine + có xoay thử hay không, xem ocr_signature). Khác chữ ký thì toàn bộ mục cũ bị bỏ qua (và bị ghi đè khi
  lưu), nên đổi chuỗi engine hay bật / tắt xoay trang không dùng nhầm kết quả cũ.
- Chỉ dùng cho bộ OCR có chữ ký (CascadePageOcr); bộ OCR cũ không có.
"""

from pathlib import Path

from vi_corpus.common.state import read_json, write_json_atomic

CACHE_DIR = "interim/vjol_ocr_cache"
CACHE_VERSION = 1  # đổi khi đổi định dạng file hoặc cách tạo ảnh đưa vào OCR


def ocr_signature(tier_names: list[str], rotate: bool) -> str:
    """Chữ ký của bộ OCR: tên các engine theo thứ tự, thêm ":rot" nếu có xoay thử trang nghi ngờ (J-24)."""
    return "|".join(tier_names) + (":rot" if rotate else "")


def page_key(index: int, render: bool) -> str:
    """Khoá trang trong file: "<chỉ số>:r" khi vẽ lại cả trang, "<chỉ số>:e" khi lấy ảnh nhúng."""
    return f"{index}:{'r' if render else 'e'}"


class OcrCache:
    """Kết quả OCR theo trang của MỘT PDF, đọc / ghi một file JSON nguyên tử."""

    def __init__(self, path: Path, signature: str) -> None:
        """
        Args:
            path:      File cache của PDF này (xem cache_path).
            signature: Chữ ký bộ OCR hiện tại; file cũ khác chữ ký bị coi là rỗng.
        """
        self.path = path
        self.signature = signature
        data = read_json(path)
        ok = isinstance(data, dict) and data.get("version") == CACHE_VERSION and data.get("signature") == signature
        self._pages: dict[str, dict] = dict(data["pages"]) if ok and isinstance(data.get("pages"), dict) else {}
        self._dirty = False

    @staticmethod
    def cache_path(data_root: Path, sha256: str) -> Path:
        """Đường dẫn file cache của PDF có băm nội dung `sha256`."""
        return data_root / CACHE_DIR / sha256[:2] / f"{sha256}.json"

    @classmethod
    def for_pdf(cls, data_root: Path, sha256: str, signature: str) -> "OcrCache":
        """Cache của PDF có băm `sha256` dưới `data_root`."""
        return cls(cls.cache_path(data_root, sha256), signature)

    def get(self, index: int, render: bool) -> dict | None:
        """Kết quả OCR đã lưu của trang `index` (bản sao), None nếu chưa có."""
        entry = self._pages.get(page_key(index, render))
        return dict(entry) if entry is not None else None

    def put(self, index: int, render: bool, entry: dict) -> None:
        """Ghi kết quả OCR của trang `index` (chỉ trong bộ nhớ; gọi save() để ghi xuống đĩa)."""
        self._pages[page_key(index, render)] = dict(entry)
        self._dirty = True

    def __len__(self) -> int:
        """Số trang đang có trong cache."""
        return len(self._pages)

    def save(self) -> None:
        """Ghi xuống đĩa nếu có thay đổi."""
        if self._dirty:
            write_json_atomic(self.path, {"version": CACHE_VERSION, "signature": self.signature, "pages": self._pages})
            self._dirty = False

