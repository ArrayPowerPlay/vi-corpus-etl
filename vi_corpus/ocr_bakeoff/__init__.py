"""
So sánh engine OCR để chọn cơ chế OCR mới (docs/DECISION_LOG.md, F-11).

Các bước (script ở scripts/ocr_bakeoff/, chạy trên server GPU):
    pages.py    : chọn trang. Bộ A = trang PDF giáo trình có lớp chữ hợp lệ, render thành ảnh cỡ trang stbook rồi làm
                  xấu nhẹ (có đáp án = lớp chữ). Bộ B = ảnh quét thật của sách stbook (không có đáp án).
    engines.py  : chạy từng engine (baseline Paddle + VietOCR, mô hình thị giác - ngôn ngữ qua vLLM, PaddleOCR-VL), lưu
                  đầu ra thô theo trang và thời điểm bắt đầu / xong để đo tốc độ.
    textnorm.py : adapter chung đưa đầu ra (markdown, HTML, JSON bố cục) về văn bản thuần.
    metrics.py  : CER, WER, F1 túi từ, recall ký tự đặc biệt, vòng lặp, bootstrap.
    syllables.py: tập âm tiết lập từ SEA-PILE (đo tỷ lệ âm tiết lạ ở bộ B).
    perplexity.py: perplexity Qwen3-0.6B của đầu ra (bộ B, đo thứ tự đọc khi không có đáp án).
    scoring.py  : chấm điểm, áp quy tắc chọn đã viết trước, xuất summary.json / per_page.csv / report.html.
    guard.py    : bộ kiểm từng trang cho OCR bằng VLM (O-18): nhận ra trang hỏng hẳn mà không cần đáp án.
    cascade.py  : chuỗi engine (O-20): quy tắc chọn pick_tier dùng chung cho chạy thật (CascadeOcr), mô phỏng và chấm điểm.
    vlm_bench.py: thử concurrency và tỷ lệ trang bị báo trên ~1.000 trang stbook thật (scripts/stbook/ocr_vlm_bench.py).
    gt_diag.py  : chẩn đoán đáp án bộ A (trang đôi, số trang lẫn giữa trang, tiêu đề / chân trang engine bỏ), CPU.
"""
