# Lộ trình (theo bảng mục tiêu tuần)

| Tuần | Trọng tâm | Mục tiêu chính | Việc trong repo |
|---|---|---|---|
| W1 | Kiểm kê nguồn + thiết kế ingestion | ≥20 nguồn ứng viên (sách/giáo trình, VJOL/VISTA, social, coding, standards); mỗi nguồn có owner, license, domain; định nghĩa clean corpus schema; thiết kế NeMo Curator; định nghĩa knowledge unit + instruction schema; ma trận CPT/SFT/Hybrid | `docs/SOURCES.md`, `docs/PIPELINE.md`, source registry |
| W2 | Ingestion PoC | ≥10 nguồn giá trị cao, ≥3 domain học thuật; ≥1B raw token; ≥300M clean token; parse → language → boilerplate → exact/fuzzy dedup; ≥100k knowledge candidates; audit chất lượng + contamination scan; ≥95% lineage | stage `ingest`…`dedup`, `package`, audit |

Thứ tự triển khai code: (1) đổi tên + tài liệu, (2) source registry + schema + adapter SEA, (3) language/normalize/quality/dedup, (4) adapter VISTA/VJOL (triage + parse), (5) adapter Drive + knowledge unit + dashboard.

Ghi chú: ảnh mục tiêu còn cột "Acceptance/KPI" bị cắt ở bên phải; cần bổ sung khi có bản đầy đủ.
