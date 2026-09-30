# Danh mục nguồn dữ liệu

> Đáp ứng KPI W1: mỗi nguồn phải có owner, provenance/license, domain. Các ô "chưa rõ" cần điền khi có thông tin.
> Đường dẫn dữ liệu cấu hình trong `configs/sources.json` (sẽ tạo ở bước ingest); VISTA dùng đường dẫn tạm; VJOL ở `data/raw/VJOL/`, cấu trúc bên trong chờ bổ sung.

| key | Nguồn | Owner | License | Domain | Định dạng | Đường dẫn | Trạng thái |
|---|---|---|---|---|---|---|---|
| `sea_instruct_2602` | `aisingapore/SEA-Instruct-2602` (gated) | AI Singapore | ODC-By 1.0 (cần xác nhận) | instruction đa lĩnh vực | parquet, ~2,7 GB | `data/raw/sea_vi/sea_instruct_2602/` | Đã tải |
| `sea_pile_v2` | `aisingapore/SEA-PILE-v2`, `vi/` | AI Singapore | ODC-By 1.0 + CommonCrawl ToU | web | parquet, ~132 GB | `data/raw/sea_vi/sea_pile_v2/` | Đã tải |
| `sea_lion_pile_v1` | `aisingapore/SEA-PILE-v1`, `sea-pile-mc4/vi/` | AI Singapore | ODC-By 1.0 + CommonCrawl ToU | web (mC4) | jsonl.gz, ~107 GB | `data/raw/sea_vi/sea_lion_pile_v1/` | Đã tải |
| `stbook` | stbook.vn, tải bằng repo `stbook_crawler` (sách miễn phí đọc online) | NXB Chính trị quốc gia Sự thật | bản quyền NXB, chưa rõ quyền tái sử dụng → quarantine | sách chính trị - xã hội | PDF dạng ảnh → OCR (PaddleOCR + VietOCR) | `data/raw/stbook/` → `data/interim/stbook_ocr/` | Đang tải; đã có bước OCR |
| `giao_trinh` | Google Drive "Tổng hợp giáo trình Đại học" (hầu hết là shortcut tới file người khác), tải bằng rclone | nhiều tác giả / trường / NXB | chưa rõ (có sách từ PDFDrive, z-lib) → quarantine | giáo trình đại học 16 ngành, có sách tiếng Anh | ≥1.498 file: PDF (đa số), pptx, ppt, docx | `data/raw/giao_trinh/` | Đã có chiến lược (`docs/GIAO_TRINH.md`) |
| `vista` | sti.vista.gov.vn | Cục Thông tin KH&CN quốc gia (cần xác nhận) | chưa rõ → quarantine | bài báo khoa học | PDF | chưa có (tạm) | Chờ dữ liệu |
| `vjol` | vjol.info.vn | các tạp chí (bản quyền thuộc journal/tác giả) | chưa rõ → quarantine | bài báo khoa học | PDF + metadata | `data/raw/VJOL/` (cấu trúc bên trong sẽ bổ sung) | Chờ dữ liệu |

Ghi chú: cấu trúc thư mục dữ liệu xem `PROJECT_ARCHITECTURE.md`. `SEA-PILE-v1` là tên repo của "SEA-LION-Pile v1"; trên HF chỉ có phần mC4.
