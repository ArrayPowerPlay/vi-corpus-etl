# Danh mục nguồn dữ liệu

> Đáp ứng KPI W1: mỗi nguồn phải có owner, provenance/license, domain. Các ô "chưa rõ" cần điền khi có thông tin.
> Đường dẫn dữ liệu cấu hình trong `configs/sources.json` (sẽ tạo ở bước ingest); VISTA/VJOL/Drive dùng đường dẫn tạm.

| key | Nguồn | Owner | License | Domain | Định dạng | Đường dẫn | Trạng thái |
|---|---|---|---|---|---|---|---|
| `sea_instruct_2602` | `aisingapore/SEA-Instruct-2602` (gated) | AI Singapore | ODC-By 1.0 (cần xác nhận) | instruction đa lĩnh vực | parquet, ~2,7 GB | `data/raw/sea_instruct_2602/` | Đã tải |
| `sea_pile_v2` | `aisingapore/SEA-PILE-v2`, `vi/` | AI Singapore | ODC-By 1.0 + CommonCrawl ToU | web | parquet, ~132 GB | `data/raw/sea_pile_v2/` | Đã tải |
| `sea_lion_pile_v1` | `aisingapore/SEA-PILE-v1`, `sea-pile-mc4/vi/` | AI Singapore | ODC-By 1.0 + CommonCrawl ToU | web (mC4) | jsonl.gz, ~107 GB | `data/raw/sea_lion_pile_v1/` | Đã tải |
| `stbook` | stbook.vn, tải bằng repo `stbook_crawler` (sách miễn phí đọc online) | NXB Chính trị quốc gia Sự thật | bản quyền NXB, chưa rõ quyền tái sử dụng → quarantine | sách chính trị - xã hội | PDF dạng ảnh → OCR (PaddleOCR + VietOCR) | `data/raw/stbook/` → `data/interim/stbook_ocr/` | Đang tải; đã có bước OCR |
| `drive_*` | Google Drive (folder do nhóm cập nhật) | chưa rõ | chưa rõ | chưa rõ | chưa rõ | chưa có (tạm) | Chờ đọc folder |
| `vista` | sti.vista.gov.vn | Cục Thông tin KH&CN quốc gia (cần xác nhận) | chưa rõ → quarantine | bài báo khoa học | PDF | chưa có (tạm) | Chờ dữ liệu |
| `vjol` | vjol.info.vn | các tạp chí (bản quyền thuộc journal/tác giả) | chưa rõ → quarantine | bài báo khoa học | PDF + metadata | chưa có (tạm) | Chờ dữ liệu |

Ghi chú: `SEA-PILE-v1` là tên repo của "SEA-LION-Pile v1"; trên HF chỉ có phần mC4.
