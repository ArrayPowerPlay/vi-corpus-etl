# Tiến độ so với bảng Milestone Tracker (W1-W8)

> File này đối chiếu từng mục của bảng mục tiêu (ảnh Milestone Tracker, phần việc xử lý dữ liệu) với code và tài liệu hiện có trong repo.
> Cập nhật tay mỗi khi hoàn thành một mục. Đánh giá dựa trên code, test và tài liệu; **các con số mục tiêu (token, số ứng viên, F1...) chưa được đo trên dữ liệu thật** vì pipeline mới chạy thử trên mẫu N bản ghi.
> Cột "Actual Result" ở ngoài cùng bên phải của bảng gốc bị cắt khỏi ảnh nên chưa được đưa vào đây.
> Cập nhật lần cuối: 2026-10-05.

Quy ước: ✅ xong | 🟡 mới làm một phần (bản tạm, chưa đạt yêu cầu mở rộng) | ⬜ chưa làm

## 1. Đang ở giai đoạn nào

**Đang ở W2 (Ingestion PoC), chưa đóng được W2.**

- **W1 (Kiểm kê nguồn + thiết kế)**: gần xong. Còn thiếu: danh mục mới có 7 nguồn (cần ≥20), chưa có nhóm social / coding / standards, nhiều ô license còn "chưa rõ".
- **W2 (Ingestion PoC)**: code của chuỗi `ingest → language → quality → dedup → knowledge unit → audit` đã có, chạy được trên mẫu. Chưa đạt mục tiêu định lượng: chưa có ≥300M clean token, chưa đo ≥1B raw token, chưa có ≥100k knowledge candidate, chưa quét rò rỉ benchmark, VJOL/VISTA chưa có dữ liệu.
- **W3 trở đi**: gần như chưa bắt đầu. Chỉ có vài mầm sớm: bản đồ embedding (của W4), heuristic chất lượng (của W3), `split_bucket` theo họ trùng (của W6).

| Tuần | Trọng tâm | Trạng thái |
|---|---|---|
| W1 | Kiểm kê nguồn + thiết kế ingestion | 🟡 gần xong |
| W2 | Ingestion PoC | 🟡 **đang làm (hiện tại)** |
| W3 | Filtering pipeline v1 | ⬜ (có mầm: heuristic chất lượng) |
| W4 | Quality model | ⬜ (có mầm: embedding map) |
| W5 | Knowledge atomization | ⬜ |
| W6 | Mini-model injection | ⬜ (có mầm: split theo họ trùng) |
| W7 | Scale experiment | ⬜ (có mầm: chạy song song Ray) |
| W8 | Release + decision | ⬜ |

## 2. Chi tiết từng tuần

### W1: Source inventory + ingestion design

| Mục (theo bảng) | Trạng thái | Hiện có / còn thiếu |
|---|---|---|
| ≥20 nguồn ứng viên (sách, VJOL/VISTA, social, coding, standards) | 🟡 | `docs/SOURCES.md` có 7 nguồn (3 SEA, stbook, giao_trinh, VISTA, VJOL). Thiếu: social, coding, standards và còn ~13 nguồn nữa. |
| Mỗi nguồn có owner, license, domain (KPI) | 🟡 | Đã có cột owner / license / domain, nhưng nhiều ô ghi "chưa rõ" / "cần xác nhận"; đường dẫn VISTA chỉ là tạm. |
| Raw: kiểm kê nền + ước lượng khối lượng nguồn | 🟡 | Có dung lượng SEA (~2,7 / 132 / 107 GB), giáo trình ≥1.498 file (`scripts/giao_trinh/survey.py`), `scripts/sea/count_rows.py`. Chưa quy ra số token. |
| Định nghĩa clean corpus schema | ✅ | `vi_corpus/common/schema.py`, `docs/PIPELINE.md` mục 3. |
| Thiết kế NeMo Curator | 🟡 | Kiến trúc lai đã chốt (`docs/PIPELINE.md`). Adapter `vi_corpus/pipeline/curator.py` mới bọc language, quality, embed, OCR; fuzzy dedup chưa có adapter. Ray mới kiểm trên CPU, chưa thử nhiều GPU thật. |
| Parse PDF / DOCX / HTML / EPUB, provenance / license | 🟡 | PDF (PyMuPDF, TCVN3, OCR) và pptx đã làm cho stbook, giáo trình. Chưa có: DOCX, ppt, djvu (bị bỏ qua `unsupported_format`), HTML, EPUB, parse bài báo VJOL/VISTA. Provenance có (`source_path`, `source_sha256`); license mặc định `unknown`. |
| Định nghĩa knowledge unit + instruction schema; ma trận CPT/SFT/Hybrid | 🟡 | `KU_SCHEMA` có CPT và SFT (`pipeline/knowledge.py`). Hybrid chưa sinh (dự kiến ghép ở bước huấn luyện); schema instruction theo độ sâu (W5) chưa có. |
| Thí nghiệm: liệt kê dữ liệu nguồn | ✅ | `docs/SOURCES.md`. |

### W2: Ingestion PoC (hiện tại)

| Mục (theo bảng) | Trạng thái | Hiện có / còn thiếu |
|---|---|---|
| ≥10 nguồn giá trị cao, ≥3 domain học thuật | 🟡 | Có dữ liệu cho 5 nguồn (3 SEA, stbook, giáo trình 16 ngành). Giáo trình có thể đủ ≥3 domain học thuật, nhưng chưa chạy thật. VJOL/VISTA chưa có dữ liệu. |
| ≥1B raw token | 🟡 | SEA đã tải ~240 GB nên nhiều khả năng đạt, nhưng **chưa đo** (hiện token_count mặc định = số từ, ước lượng thấp). |
| ≥300M clean token | ⬜ | Pipeline chỉ chạy trên mẫu N bản ghi. Mỗi stage nạp cả danh sách vào RAM nên chưa chạy được toàn bộ SEA-PILE (~132 GB). |
| Parse: PDF → text | 🟡 | stbook (OCR) và giáo trình (text + OCR) có; OCR thật chưa kiểm ở máy này (cần GPU server). VJOL/VISTA chưa có script. |
| Language id | 🟡 | `pipeline/language.py` dùng heuristic (dấu tiếng Việt + stopword), chưa dùng fastText. |
| Boilerplate | 🟡 | Mới là **kiểm tra + trừ điểm** ở stage quality cho nguồn web; chưa có bước xóa dòng boilerplate lặp trong văn bản. |
| Exact dedup | ✅ | sha256 text đã chuẩn hóa (`pipeline/dedup.py`). |
| Fuzzy dedup | 🟡 | MinHash-LSH bằng numpy, CPU, chạy trong RAM. Dùng được trên mẫu; chưa mở rộng được (chưa có bản Curator / phân tán). |
| ≥100k knowledge candidates | ⬜ | Có bộ sinh CPT/SFT nhưng chưa chạy ở quy mô; chưa đếm. |
| Audit chất lượng + contamination scan | 🟡 | `audit.json`: lineage, funnel, token, đếm PII. Contamination ghi "chưa chạy" (chưa có tập benchmark tiếng Việt). |
| ≥95% truy vết nguồn (KPI) | 🟡 | Đã đo trong audit (`lineage_rate_kept`, mục tiêu 0,95) nhưng mới trên mẫu, chưa trên corpus đầy đủ. |

### W3: Filtering pipeline v1

| Mục (theo bảng) | Trạng thái | Hiện có / còn thiếu |
|---|---|---|
| Mở rộng nguồn: sách + khoa học + kỹ thuật | ⬜ | Chưa có nguồn mới ngoài danh sách W1. |
| ≥30B raw / ≥15B clean tích lũy | ⬜ | Chưa đo. |
| Semantic dedup | ⬜ | Đã có embedding nhưng chỉ để vẽ bản đồ, chưa dùng để loại trùng ngữ nghĩa. |
| Contamination scan | ⬜ | Chưa có tập benchmark; audit ghi "chưa chạy". |
| Quality heuristics | 🟡 | `pipeline/quality.py`: điểm 0-100, band A/B/C/D, reason code, ngưỡng theo nguồn (`config.py`). **Ngưỡng chưa chỉnh** trên dữ liệu thật. |
| ≥500k train candidates | ⬜ | Chưa đo. |
| Đường cong retention / chất lượng sau lọc | 🟡 | `report.html` có funnel và phân phối band theo nguồn; chưa có đường cong theo ngưỡng. |
| Chốt trade-off chất lượng và tỷ lệ giữ lại | ⬜ | Cần chỉnh ngưỡng sau khi xem báo cáo trên dữ liệu thật. |

### W4: Quality model

| Mục (theo bảng) | Trạng thái | Hiện có / còn thiếu |
|---|---|---|
| ≥20 nguồn nữa; pool có nhãn | ⬜ | Chưa có. |
| Gán nhãn ≥50k, huấn luyện quality filter v1, chấm điểm corpus | ⬜ | Hiện chỉ có điểm heuristic, chưa có nhãn hay mô hình. |
| Embedding map v1 | 🟡 | Có `embed` + `reduce` (PCA, UMAP, HDBSCAN) + plotly (`pipeline/embed.py`, `reduce.py`, `viz.py`). Mới chạy trên mẫu; chưa fit trên mẫu phân tầng, chưa có Datashader cho >100k điểm. |
| ≥1,2M candidates | ⬜ | Chưa đo. |
| F1 / precision + độ phủ cluster / domain | ⬜ | Có `cluster_id` nhưng chưa phân tích theo domain. |
| Bộ lọc tương quan với đánh giá của judge; tìm domain thưa | ⬜ | Chưa có judge. |

### W5: Knowledge atomization

| Mục (theo bảng) | Trạng thái | Hiện có / còn thiếu |
|---|---|---|
| Bổ sung các domain học thuật còn thiếu; ≥7B raw / ≥2B clean | ⬜ | Chưa có. |
| Tách section → facts / concepts / relations; phân tích mật độ tri thức | ⬜ | Knowledge unit hiện chỉ là cả văn bản / cả đoạn sách (CPT) hoặc cặp hỏi đáp có sẵn của SEA-Instruct (SFT). Chưa có tách nguyên tử. |
| ≥2M instruction ở 3 mức độ sâu | ⬜ | Chưa có. |
| Audit tính nguyên tử, tính mới, căn cứ sự thật | ⬜ | Chưa có. |
| Ít dư thừa, provenance mạnh (KPI) | ⬜ | Provenance mức bản ghi đã có; mức từng fact thì chưa. |

### W6: Mini-model injection

| Mục (theo bảng) | Trạng thái | Hiện có / còn thiếu |
|---|---|---|
| Chốt bộ lọc + mixture builder | ⬜ | Chưa có bộ trộn tỷ lệ nguồn. `split_bucket` (băm theo `dedup_family_id`) đã sẵn để chia sau. |
| Dataset CPT vs SFT vs Hybrid | 🟡 | `knowledge_units.parquet` có CPT và SFT. Hybrid chưa có. |
| Mini model: VMLU / Lịch sử / Địa lý / Tổng hợp | ⬜ | Chưa có. |
| ≥3B clean | ⬜ | Chưa đo. |
| Tăng kiến thức, tụt năng lực chung ≤1 điểm (KPI) | ⬜ | Chưa có. |

### W7: Scale experiment

| Mục (theo bảng) | Trạng thái | Hiện có / còn thiếu |
|---|---|---|
| Chốt bộ nguồn chuẩn production | ⬜ | Chưa có. |
| ≥10B clean | ⬜ | Chưa đo. |
| Tối ưu throughput / chi phí | 🟡 | Có chạy song song nhiều CPU/GPU bằng Ray (`--executor`, `--num-gpus`). Mới kiểm trên CPU. Chưa tối ưu việc nạp cả dữ liệu vào RAM. |
| Versioning dataset | 🟡 | Mỗi lần chạy có `manifest.json` ghi cấu hình + thống kê. Chưa có phiên bản hóa bộ dữ liệu phát hành. |
| Gói huấn luyện lớn + replay mixture | ⬜ | Chưa có. |
| Ablation kích thước / tỷ lệ / giai đoạn | ⬜ | Chưa có. |

### W8: Release + decision

| Mục (theo bảng) | Trạng thái | Hiện có / còn thiếu |
|---|---|---|
| Danh mục nguồn cuối + phân tích khoảng trống | ⬜ | `docs/SOURCES.md` là bản đầu. |
| Clean release đã duyệt | ⬜ | Chưa có. |
| Đóng băng cấu hình + provenance + dataset có version | 🟡 | `manifest.json` ghi cấu hình từng lần chạy; chưa có quy trình đóng băng. |
| Gói Knowledge SFT cuối + công thức | ⬜ | Chưa có. |
| Phân tích knowledge gain + forgetting; go/no-go | ⬜ | Chưa có. |

## 3. Việc nên làm tiếp để đóng W2

1. **Chạy pipeline trên dữ liệu thật ở server** (có `data/raw`), ghi lại các số đo: raw token, clean token, số knowledge unit, lineage. Đây là cách duy nhất biết W2 đạt hay chưa.
2. **Sửa pipeline để xử lý theo luồng** (không nạp cả danh sách vào RAM), vì mục tiêu ≥300M clean token không chạy được bằng bản hiện tại.
3. **Đo token bằng tokenizer thật** (`--tokenizer`) thay vì đếm từ.
4. **Bổ sung nguồn**: VJOL/VISTA khi có dữ liệu (cần phân loại PDF và `cmap_healer` như ViLA), và các nguồn còn thiếu để đủ ≥10 nguồn / ≥20 ứng viên.
5. **Làm contamination scan**: cần có tập benchmark tiếng Việt (VMLU và các bộ khác).
6. Điền license / owner còn "chưa rõ" trong `docs/SOURCES.md`.
