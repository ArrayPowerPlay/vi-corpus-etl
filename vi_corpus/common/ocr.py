"""
OCR tiếng Việt cho PDF dạng ảnh (không có lớp chữ), vd sách tải từ stbook.vn.

Hai bước cho mỗi trang:
1. PaddleOCR (PP-OCRv5_mobile_det) tìm các khung dòng chữ.
2. VietOCR (vgg_transformer) đọc chữ trong từng khung. Mô hình nhận dạng của PaddleOCR
   không đọc được dấu tiếng Việt (đã thử: "đánh du bưc ngot"), nên chỉ dùng phần detect.

Lớp PageOcr cần nhóm thư viện `ocr`: `uv sync --group ocr` (các hàm còn lại thì không). Lần chạy đầu tự tải trọng số mô hình
(~600 MB). Các khung được cắt rộng thêm PADDING px: nếu cắt sát, VietOCR hay bịa thêm chữ
ở đầu/cuối dòng.
"""

import fcntl
import io
import os
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pymupdf
from PIL import Image

PADDING = 6
DET_MODEL = "PP-OCRv5_mobile_det"
REC_MODEL = "vgg_transformer"


def group_lines(boxes: list[tuple[int, int, int, int]]) -> list[list[tuple[int, int, int, int]]]:
    """
    Gom các khung (x0, y0, x1, y1) thành dòng, theo thứ tự đọc: trên xuống, trái sang phải.

    Một khung thuộc dòng hiện tại nếu tâm dọc của nó nằm trong khoảng y của khung đầu dòng.
    """
    lines: list[list[tuple[int, int, int, int]]] = []
    for box in sorted(boxes, key=lambda b: (b[1], b[0])):
        center = (box[1] + box[3]) / 2
        if lines and lines[-1][0][1] <= center <= lines[-1][0][3]:
            lines[-1].append(box)
        else:
            lines.append([box])
    return [sorted(line) for line in lines]


def pdf_page_images(pdf_path: Path) -> Iterator[Image.Image]:
    """
    Sinh ảnh từng trang của PDF (xem page_image).

    Raises:
        ValueError: PDF bị cắt cụt / hỏng (vd crawler bị kill lúc đang ghi) mà PyMuPDF phải
                    tự sửa khi mở; nếu OCR tiếp sẽ chỉ được một phần sách.
    """
    with pymupdf.open(pdf_path) as doc:
        if doc.is_repaired:
            raise ValueError("PDF hỏng, cần xoá và tải lại bằng stbook-crawler")
        for page in doc:
            yield page_image(page)


def page_image(page: "pymupdf.Page") -> Image.Image:
    """
    Ảnh của một trang PDF để OCR.

    Trang đúng một ảnh (sách scan, PDF stbook) thì lấy thẳng ảnh gốc (không mất chất lượng);
    trang không có ảnh hoặc nhiều ảnh thì render ở 200 dpi.
    """
    images = page.get_images()
    if len(images) == 1:
        data = page.parent.extract_image(images[0][0])["image"]
    else:
        data = page.get_pixmap(dpi=200).tobytes("png")
    return Image.open(io.BytesIO(data)).convert("RGB")


def ensure_rec_weights(url: str) -> str:
    """
    Đảm bảo file trọng số VietOCR có đủ, nguyên vẹn trong thư mục tạm, rồi trả về đường dẫn.

    VietOCR tự tải vào /tmp nhưng không an toàn khi nhiều tiến trình cùng chạy: tiến trình sau thấy file đang tải dở
    là bỏ qua tải và nạp file hỏng ("failed finding central directory"). Ở đây việc tải được khoá bằng flock,
    ghi ra file .part rồi đổi tên nguyên tử, và file đã có nhưng hỏng (tải dở từ lần chạy trước) sẽ bị tải lại.

    Args:
        url: Địa chỉ file trọng số (cfg["weights"]).

    Returns:
        Đường dẫn file trọng số đã kiểm tra.
    """
    import requests

    path = Path(tempfile.gettempdir()) / url.rsplit("/", 1)[-1]
    with open(f"{path}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # tự nhả khi đóng file
        if path.exists() and zipfile.is_zipfile(path):
            return str(path)
        part = Path(f"{path}.part")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(part, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        if not zipfile.is_zipfile(part):
            part.unlink(missing_ok=True)
            raise RuntimeError(f"File trọng số tải về từ {url} bị hỏng")
        os.replace(part, path)
    return str(path)


class PageOcr:
    """Bộ OCR một trang: nạp mô hình detect + nhận dạng một lần, dùng lại cho mọi trang."""

    def __init__(self, device: str = "cuda"):
        """
        Nạp mô hình.

        Args:
            device: "cuda" hoặc "cpu" cho VietOCR. Phần detect chạy GPU chỉ khi đã cài
                    paddlepaddle-gpu (mặc định cài bản CPU), ngược lại chạy CPU.
        """
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        # vietocr còn gọi Image.ANTIALIAS, đã bị xoá từ Pillow 10.
        if not hasattr(Image, "ANTIALIAS"):
            Image.ANTIALIAS = Image.LANCZOS
        import paddle
        from paddleocr import TextDetection
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        if device == "cuda" and paddle.device.is_compiled_with_cuda():
            self.det = TextDetection(model_name=DET_MODEL, device="gpu:0")
        else:
            # oneDNN của paddle 3.3 lỗi với mô hình này (NotImplementedError ConvertPirAttribute...).
            self.det = TextDetection(model_name=DET_MODEL, device="cpu", enable_mkldnn=False)
        cfg = Cfg.load_config_from_name(REC_MODEL)
        cfg["device"] = device
        # Nhiều actor cùng nạp mô hình: tải trọng số một lần có khoá (xem ensure_rec_weights), và bỏ trọng số
        # ImageNet của VGG (sẽ bị load_state_dict ghi đè) để khỏi tải thừa vgg19_bn về cache dùng chung.
        cfg["weights"] = ensure_rec_weights(cfg["weights"])
        cfg["cnn"]["pretrained"] = False
        self.rec = Predictor(cfg)

    def page_text(self, image: Image.Image) -> str:
        """Trả về text của một trang, mỗi dòng in một dòng; trang trắng trả về ""."""
        import numpy as np

        polys = self.det.predict(np.array(image)[:, :, ::-1])[0]["dt_polys"]
        boxes = [
            (max(int(p[:, 0].min()) - PADDING, 0), max(int(p[:, 1].min()) - PADDING, 0),
             int(p[:, 0].max()) + PADDING, int(p[:, 1].max()) + PADDING)
            for p in polys
        ]
        lines = group_lines(boxes)
        flat = [box for line in lines for box in line]
        if not flat:
            return ""
        texts = iter(self.rec.predict_batch([image.crop(b) for b in flat]))
        return "\n".join(" ".join(next(texts) for _ in line) for line in lines)
