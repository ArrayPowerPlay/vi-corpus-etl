# Tiến độ so với bảng Milestone Tracker (W1-W8)

> File này đối chiếu từng mục của bảng mục tiêu (ảnh Milestone Tracker, phần việc xử lý dữ liệu) với code và tài liệu hiện có trong repo.
> Cập nhật tay mỗi khi hoàn thành một mục. Đánh giá dựa trên code, test (747 test qua, 3 bỏ qua, `uv run pytest`, đo 2026-11-20), `docs/project/DECISION_LOG.md` (đến phiên 2026-11-17) và lịch sử git (đến 2026-11-19). **Các con số mục tiêu (token, số ứng viên, F1...) vẫn chưa được đo trên dữ liệu thật ở quy mô đầy đủ**: kết quả chạy trên server chỉ được ghi vào tài liệu khi chủ dự án báo lại (bảng cuối `docs/project/SERVER_TASK.md`), và bảng đó hiện còn trống.
> Cột "Actual Result" ở ngoài cùng bên phải của bảng gốc bị cắt khỏi ảnh nên chưa được đưa vào đây.
> Cập nhật lần cuối: 2026-11-19 (rà lại toàn bộ theo code và nhật ký quyết định; bản trước là 2026-10-08); 2026-11-20: cập nhật dòng giáo trình (GT-01) và số test.

Quy ước: ✅ xong | 🟡 mới làm một phần (bản tạm, chưa đạt yêu cầu mở rộng) | ⬜ chưa làm

## 1. Đang ở giai đoạn nào

**Vẫn ở W2 (Ingestion PoC), chưa đóng được W2; đã làm trước một phần W3 / W4 / W6.** Điểm khác so với 2026-10-08:

- **W1 (Kiểm kê nguồn + thiết kế)**: gần xong. `docs/sources/SOURCES.md` nay có **14 nguồn** (7 đã có code xử lý: 3 SEA, stbook, giáo trình, VJOL, VISTA; 7 đã chọn nhưng chưa có code: `wiki_vi`, `phap_luat`, `fineweb2_vi`, `anle`, `luanvan_vnu`, `finepdfs_edu_vi`, `finepdfs_vi`) cộng danh sách ứng viên đã khảo sát ở `docs/sources/SOURCE_CANDIDATES.md`. Còn thiếu: nhóm social / coding / standards (đã quyết chưa lấy code / The Stack, TCVN; social chưa có), nhiều ô license còn "chưa rõ". Thư xin phép 4 nguồn có bản quyền đã soạn nháp (`docs/sources/PERMISSION_EMAILS.md`), chờ duyệt (NS-08).
- **W2 (Ingestion PoC)**: chuỗi `ingest → prepare → language → quality → dedup → embed → reduce → finalize` chạy được trên mẫu (và, cho VJOL / VISTA, trên cả kho theo từng phần `--shard`). Đã có code xử lý thô cho cả 5 nguồn lớn (SEA, stbook, giáo trình, VJOL, VISTA). OCR stbook bằng chuỗi VLM đã chạy xong toàn bộ (2026-10-18). Với VJOL và VISTA, bước trích lớp chữ không OCR (`--no-ocr`) đã chạy trên cả kho (PP-01); **OCR GPU các trang `needs_ocr` là bước kế tiếp**. Chưa đạt mục tiêu định lượng: chưa có số ≥300M token sạch được đo, chưa đo ≥1B raw token, chưa đếm ≥100k knowledge candidate, quét nhiễm benchmark chưa chạy (danh sách benchmark rỗng).
- **W3 trở đi**: có các mầm khá xa: heuristic chất lượng đã khá dày cho VJOL (bộ dò chữ hỏng, lệch dấu, mất chữ...), bản đồ embedding, `cpt_segments` + `scripts/cpt_pack.py` (đóng gói 8.192 token), `split_bucket` theo họ trùng, chạy song song Ray, chia phần theo shard. Chưa có: bộ chấm chất lượng bằng mô hình, semantic dedup, atomization, mini-model.
- **Việc đang chờ**: các điểm mở của pipeline v2 (`docs/project/TASKS.md`: K-xx cần chốt, E-xx cần đo, L-xx cần code, hiện **chưa mục nào được chốt**), kết quả OCR GPU của VJOL / VISTA, chạy thật `make pipeline-cpu`, và xin phép các nguồn có bản quyền.

| Tuần | Trọng tâm | Trạng thái |
|---|---|---|
| W1 | Kiểm kê nguồn + thiết kế ingestion | 🟡 gần xong |
| W2 | Ingestion PoC | 🟡 **đang làm (hiện tại)** |
| W3 | Filtering pipeline v1 | 🟡 có heuristic chất lượng, quét nhiễm chưa có benchmark; semantic dedup chưa làm |
| W4 | Quality model | ⬜ (có mầm: embedding map, mẫu chấm LLM, perplexity tuỳ chọn) |
| W5 | Knowledge atomization | ⬜ (có thiết kế: `docs/training/SUGGESTION_v2.md` mục 10) |
| W6 | Mini-model injection | ⬜ (có mầm: `cpt_segments`, `cpt_pack`, split theo họ trùng) |
| W7 | Scale experiment | ⬜ (có mầm: Ray, chia phần theo shard, `make pipeline-cpu`) |
| W8 | Release + decision | ⬜ |

## 1b. Bảng Milestone Tracker đầy đủ (sheet `01_Knowledge_Injection`)

Nguồn: ảnh chụp bảng chủ dự án gửi 2026-10-08 (luồng: Raw sources → clean corpus → knowledge units → SFT injection; đo knowledge gain và capability retention). Chép lại nguyên văn tiếng Anh.

| Week | Focus / Milestone | Source & Collection Target | Raw Data Target | Clean Data Target | Technical / Pipeline Target | Train-ready Data Target | Experiment / Eval Target | Acceptance / KPI |
|---|---|---|---|---|---|---|---|---|
| W1 | Source inventory + ingestion design | ≥20 candidate sources; books/textbooks, VJOL/VISTA, social, coding, standards | Inventory baseline + source volume estimates | Define clean corpus schema | NeMo Curator design; PDF/DOCX/HTML/EPUB parsing; provenance/license | Define knowledge unit + instruction schema; CPT/SFT/Hybrid matrix | List sources data | Every source has owner, provenance/license, domain |
| W2 | Ingestion PoC | ≥10 high-value sources; ≥3 academic domains | ≥1B raw tokens or equivalent | ≥300M clean tokens | Parse → language → boilerplate → exact/fuzzy dedup PoC | ≥100k knowledge candidates | Clean quality audit + contamination scan | ≥95% traceable source lineage |
| W3 | Filtering pipeline v1 | Expand books + scientific + technical | ≥30B cumulative raw (*) | ≥15B cumulative clean (*) | Add semantic dedup + contamination + quality heuristics | ≥500k train candidates | Retention/quality curve after filtering | Clear quality vs retention trade-off |
| W4 | Quality model | ≥20 more sources; labeled pool | ≥5B cumulative raw | ≥1.5B cumulative clean | Label ≥50k; train quality filter v1; score corpus; embedding map v1 | ≥1.2M candidates | F1/precision; cluster/domain coverage | Filter correlates with judge; sparse domains identified |
| W5 | Knowledge atomization | Fill missing academic domains | ≥7B raw | ≥2B clean | Section → facts/concepts/relations; knowledge density analysis | ≥2M instructions across ≥3 depths | Atomicity/novelty/factual grounding audit | Low redundancy; strong provenance |
| W6 | Mini-model injection | High-depth academic sources | (trống) | ≥3B clean | Finalize filters + mixture builder | CPT vs SFT vs Hybrid datasets | Mini model: VMLU/History/Geography/General | Positive knowledge gain with ≤1 pt general drop target |
| W7 | Scale experiment | Freeze production-grade source set | (trống) | ≥10B clean | Throughput/cost optimization + dataset versioning | Large-scale train package + replay mixture | Data size/mix/stage ablations | Reproducible quality-cost frontier |
| W8 | Release + decision | Final source catalog + gap analysis | (trống) | Final approved clean release | Freeze configs + provenance + versioned dataset | Final Knowledge SFT package + recipe | Full knowledge gain + forgetting analysis | Go/no-go for production-scale injection |

(*) **Cần kiểm lại với bảng gốc**: số W3 (≥30B raw, ≥15B clean) đọc từ ảnh chụp nhỏ, không khớp dãy tăng dần của các tuần khác (raw: 1B → ? → 5B → 7B; clean: 0,3B → ? → 1,5B → 2B → 3B → 10B). Có thể là ≥3B / ≥1,5B hoặc số nhỏ hơn. Chưa dùng số W3 làm mục tiêu cho tới khi chủ dự án xác nhận (DECISION_LOG R-10).

**Cách hiểu các chỉ tiêu "candidate" / "instruction"** (DECISION_LOG R-09): candidate = một đoạn ~1.024 token (D-10) còn sống sau lọc và dedup; instruction = một cặp hỏi-đáp. Kiểm tính hợp lý: 1,2M candidate ≈ 1,2B token, khớp "≥1,5B clean" của W4.

**Lưu ý rút ra từ bảng**: mốc cuối ở W7 là **≥10B clean token**, nên tỉ lệ giữ lại sau lọc quan trọng (không chỉ chất lượng); W6 đánh giá bằng VMLU nên cần quét rò rỉ VMLU trước khi huấn luyện nếu muốn kết quả tin cậy (danh sách benchmark hiện để trống theo R-16).

## 2. Chi tiết từng tuần

### W1: Source inventory + ingestion design

| Mục (theo bảng) | Trạng thái | Hiện có / còn thiếu |
|---|---|---|
| ≥20 nguồn ứng viên (sách, VJOL/VISTA, social, coding, standards) | 🟡 | `docs/sources/SOURCES.md` có 14 nguồn đã chọn; `docs/sources/SOURCE_CANDIDATES.md` khảo sát thêm hàng chục ứng viên (đã chọn / không dùng / còn mở). Thiếu social; coding và standards đã quyết không lấy. |
| Mỗi nguồn có owner, license, domain (KPI) | 🟡 | Đủ cột owner / license / domain / nội dung, nhưng các nguồn PDF tự thu (stbook, giáo trình, VJOL, VISTA, luận văn) còn license "chưa rõ → quarantine". |
| Raw: kiểm kê nền + ước lượng khối lượng nguồn | 🟡 | Có dung lượng SEA (~2,7 / 132 / 107 GB), giáo trình ≥1.498 file, VJOL 177.031 PDF, VISTA 283.239 PDF (~234 GB), số trang cần OCR (`make count-ocr`). Chưa quy ra số token cho SEA / sách. |
| Định nghĩa clean corpus schema | ✅ | `vi_corpus/common/schema.py`, `docs/design/PIPELINE.md` mục 3. |
| Thiết kế NeMo Curator | 🟡 | Kiến trúc lai đã chốt. Adapter `vi_corpus/pipeline/curator.py` bọc language, quality, embed, OCR; fuzzy dedup chưa có adapter. Ray mới kiểm trên CPU, chưa thử nhiều GPU thật. |
| Parse PDF / DOCX / HTML / EPUB, provenance / license | 🟡 | PDF (PyMuPDF, TCVN3, OCR VLM), pptx, và theo D-05 docx / doc / ppt / djvu / html / epub (`vi_corpus/common/parsers/`, chưa thử trên file thật). VJOL / VISTA: bộ xử lý bài báo riêng. Provenance có (`source_path`, `source_sha256`); license mặc định `unknown`. |
| Định nghĩa knowledge unit + instruction schema; ma trận CPT/SFT/Hybrid | 🟡 | `KU_SFT` / `KU_CPT` trong `pipeline/knowledge.py`; `cpt_segments.parquet`. Hybrid chưa sinh. Đề xuất KU SFT dạng hội thoại nguyên vẹn (K-04, L-04) chưa chốt. |
| Thí nghiệm: liệt kê dữ liệu nguồn | ✅ | `docs/sources/SOURCES.md`. |

### W2: Ingestion PoC (hiện tại)

| Mục (theo bảng) | Trạng thái | Hiện có / còn thiếu |
|---|---|---|
| ≥10 nguồn giá trị cao, ≥3 domain học thuật | 🟡 | Có code và dữ liệu cho 7 nguồn (3 SEA, stbook, giáo trình 16 ngành, VJOL, VISTA); đã chọn thêm 7 nguồn chưa có code (chỉ `anle` đã tải về). Đủ ≥3 domain học thuật khi VJOL / VISTA / giáo trình / luận văn chạy xong. |
| ≥1B raw token | 🟡 | SEA ~240 GB và VISTA ~234 GB PDF nên nhiều khả năng đạt, nhưng **chưa đo** (`tokens.py` đã đếm bằng Qwen3, chưa chạy cả kho SEA). |
| ≥300M clean token | ⬜ | Chưa có số đo. Quy mô làm việc là hàng tỷ token (T-10), W2 chỉ là mức sàn. |
| Parse: PDF → text | 🟡 | stbook: OCR VLM đã chạy xong cả kho. Giáo trình: lớp chữ đã trích (1.663 file, còn 35.663 trang `needs_ocr`); OCR chuỗi VLM đã nối (GT-01), chưa chạy GPU (S-30). VJOL / VISTA: lớp chữ đã trích cả kho, OCR GPU các trang `needs_ocr` chờ chạy (số trang: `make count-ocr`). |
| Language id | ✅ | `pipeline/language.py`: fastText `lid.176` theo đoạn, `lang_mix` (D-02). Heuristic cũ vẫn chọn được. |
| Boilerplate | ✅ | `pipeline/lines.py`: xoá dòng lặp trong văn bản, qua trang, giữa văn bản (web); VJOL còn bỏ bìa / mục lục / tài liệu tham khảo / tiêu đề chạy (D-04, J-xx). Ngưỡng chưa chỉnh trên dữ liệu thật. |
| Exact dedup | ✅ | sha256 text đã chuẩn hóa (`pipeline/dedup.py`), theo văn bản + theo đoạn. |
| Fuzzy dedup | 🟡 | MinHash-LSH theo văn bản (CPU, vectơ hoá), L1 bao hàm, 2 vòng (`dedup_global.py`). Vòng 1 vẫn giữ cả run trong RAM: VJOL / VISTA gỡ bằng `--shard I/N`; chưa chạy cả SEA (~240 GB). Chưa có adapter Curator. |
| ≥100k knowledge candidates | ⬜ | Bộ sinh KU CPT / SFT có, manifest đếm `candidate` / `instruction`; chưa chạy ở quy mô, chưa đếm. |
| Audit chất lượng + contamination scan | 🟡 | `audit.json`: lineage, phễu, token, PII. Quét nhiễm 13-gram có code nhưng `configs/benchmarks/` rỗng; viết lại theo từng câu benchmark là L-01 (chưa làm, chờ K-01). Audit VJOL riêng: `scripts/vjol/audit_run.py`, `groundtruth.py`. |
| ≥95% truy vết nguồn (KPI) | 🟡 | Đo trong audit (`lineage_rate_kept`, mục tiêu 0,95) nhưng mới trên mẫu, chưa trên cả kho. |

### W3: Filtering pipeline v1

| Mục (theo bảng) | Trạng thái | Hiện có / còn thiếu |
|---|---|---|
| Mở rộng nguồn: sách + khoa học + kỹ thuật | 🟡 | Đã chọn FinePDFs-Edu / FinePDFs (nhóm giáo dục + nghiên cứu), luận văn ĐHQGHN (phần mở), pháp luật, án lệ; chưa tải (trừ án lệ) và chưa có code ingest. |
| ≥30B raw / ≥15B clean tích lũy | ⬜ | Số W3 chưa xác nhận (R-10, K-08); chưa đo. |
| Semantic dedup | ⬜ | Đã có embedding (chỉ để vẽ bản đồ) nhưng chưa dùng để loại trùng ngữ nghĩa (D-03 L3, chờ đo embedding). |
| Contamination scan | 🟡 | Có code; chưa có danh sách benchmark (K-01, L-01, L-11). |
| Quality heuristics | 🟡 | `pipeline/quality.py`: điểm 0-100, band A/B/C/D, reason code, ngưỡng theo nguồn; mã cứng `garbled_font`, `lost_letters`, `high_perplexity` (tuỳ chọn), bộ đo khiếm khuyết VJOL (`vi_corpus/common/defects.py`). **Ngưỡng chưa chốt hết** (HARD_DETECTOR_THRESHOLDS chờ số đo, PP-02). |
| ≥500k train candidates | ⬜ | Chưa đo. |
| Đường cong retention / chất lượng sau lọc | 🟡 | `report.html`: phễu và phân phối band theo nguồn; chưa có đường cong theo ngưỡng. |
| Chốt trade-off chất lượng và tỷ lệ giữ lại | ⬜ | K-05 (bỏ cứng hay xếp tầng) chưa chốt. |

### W4: Quality model

| Mục (theo bảng) | Trạng thái | Hiện có / còn thiếu |
|---|---|---|
| ≥20 nguồn nữa; pool có nhãn | ⬜ | Pool nhãn: `scripts/make_judge_sample.py` (mẫu 50k phân tầng) đã có, chưa có nhãn. Bộ đọc tay VJOL (`review_sample.py`, `groundtruth.py`) là nhãn riêng cho VJOL. |
| Gán nhãn ≥50k, huấn luyện quality filter v1, chấm điểm corpus | ⬜ | Chưa có judge (cần vLLM, S-7) và mô hình. Thiết kế: `docs/training/SUGGESTION_v2.md` mục 8, L-05. |
| Embedding map v1 | 🟡 | `embed` + `reduce` + plotly; đã chạy trên mẫu 10.000 đoạn VJOL (bản đồ trong `report.html`). Chưa fit trên mẫu phân tầng, chưa Datashader cho >100k điểm; bản full tắt embed (`--embedder none`, K-07 chờ chốt). |
| ≥1,2M candidates | ⬜ | Chưa đo. |
| F1 / precision + độ phủ cluster / domain | ⬜ | Có `cluster_id`, chưa phân tích theo domain. |
| Bộ lọc tương quan với judge; tìm domain thưa | ⬜ | Chưa có judge. |

### W5: Knowledge atomization

| Mục (theo bảng) | Trạng thái | Hiện có / còn thiếu |
|---|---|---|
| Bổ sung các domain học thuật còn thiếu; ≥7B raw / ≥2B clean | ⬜ | Chưa có. |
| Tách section → facts / concepts / relations; phân tích mật độ tri thức | ⬜ | KU hiện là cả văn bản / cả đoạn (CPT) hoặc hội thoại có sẵn của SEA-Instruct (SFT). Thiết kế SFT tổng hợp có căn cứ: `SUGGESTION_v2` mục 10.6, L-08. |
| ≥2M instruction ở 3 mức độ sâu | ⬜ | Cách đếm chưa chốt (K-27). |
| Audit tính nguyên tử, tính mới, căn cứ sự thật | ⬜ | Chưa có. |
| Ít dư thừa, provenance mạnh (KPI) | ⬜ | Provenance mức bản ghi đã có; mức từng fact thì chưa. |

### W6: Mini-model injection

| Mục (theo bảng) | Trạng thái | Hiện có / còn thiếu |
|---|---|---|
| Chốt bộ lọc + mixture builder | 🟡 | `cpt_segments.parquet` + `scripts/cpt_pack.py` (cắt 8.192 token, best-fit, EOS chỉ cuối văn bản thật; T-07..T-09) đã code. Bộ trộn tỉ lệ nguồn `cpt_mix` (L-06) chưa. `split_bucket` (băm theo `dedup_family_id`) sẵn để chia sau. |
| Dataset CPT vs SFT vs Hybrid | 🟡 | `knowledge_units.parquet` có CPT và SFT. Hybrid chưa có. |
| Mini model: VMLU / Lịch sử / Địa lý / Tổng hợp | ⬜ | Chưa có. Thí nghiệm E-06..E-11 chờ ngân sách GPU (K-14) và model (K-15). |
| ≥3B clean | ⬜ | Chưa đo. |
| Tăng kiến thức, tụt năng lực chung ≤1 điểm (KPI) | ⬜ | Ngưỡng cổng chưa chốt (K-16). |

### W7: Scale experiment

| Mục (theo bảng) | Trạng thái | Hiện có / còn thiếu |
|---|---|---|
| Chốt bộ nguồn chuẩn production | ⬜ | Chưa có. |
| ≥10B clean | ⬜ | Chưa đo (E-01). |
| Tối ưu throughput / chi phí | 🟡 | Ray (`--executor`, `--num-gpus`) kiểm trên CPU; `--shard I/N` + `--take-all` + `make pipeline-cpu` (128 phần song song, PP-03) + `make pipeline-progress`; manifest ghi `peak_rss_mb`. Chưa chạy thật cả kho, chưa có số đo RAM / giờ. |
| Versioning dataset | 🟡 | `manifest.json` mỗi lần chạy ghi cấu hình + thống kê; vân tay stage. Chưa có quy trình phát hành có version. |
| Gói huấn luyện lớn + replay mixture | ⬜ | Chưa có (L-06, K-11). |
| Ablation kích thước / tỷ lệ / giai đoạn | ⬜ | Thiết kế ở `SUGGESTION_v2` mục 12 (E-06..E-14). |

### W8: Release + decision

| Mục (theo bảng) | Trạng thái | Hiện có / còn thiếu |
|---|---|---|
| Danh mục nguồn cuối + phân tích khoảng trống | ⬜ | `docs/sources/SOURCES.md` + `SOURCE_CANDIDATES.md` là bản đầu. |
| Clean release đã duyệt | ⬜ | Chưa có; chính sách quyền cho bản phát hành chưa chốt (K-03). |
| Đóng băng cấu hình + provenance + dataset có version | 🟡 | `manifest.json` ghi cấu hình từng lần chạy; chưa có quy trình đóng băng. |
| Gói Knowledge SFT cuối + công thức | ⬜ | Chưa có. |
| Phân tích knowledge gain + forgetting; go/no-go | ⬜ | Chưa có. |

## 3. Việc nên làm tiếp

Các quyết định đã chốt nằm ở `docs/project/DECISION_LOG.md`; việc phải chạy trên server ở `docs/project/SERVER_TASK.md`; việc cần chốt / đo / code cho pipeline CPT + SFT ở `docs/project/TASKS.md`.

1. **Đóng W2 với VJOL + VISTA** (thứ tự theo PP-01 / PP-04): chạy thật `make pipeline-cpu` (S-28) để đo giờ + RAM; `font_canary` theo thiết kế (mẫu, GPU) và chốt ngưỡng; OCR GPU các trang `needs_ocr` (S-23 / S-24 cho VJOL, S-26 cho VISTA); rồi chạy lại pipeline từ ingest với `--perplexity-model` và `--hyphen-lm-model` bật.
2. **Chạy SEA qua pipeline đầy đủ** (từng subset, R-11) để có số token sạch thật (E-01) và quyết định có cần chia phần như VJOL / VISTA không.
3. **Đo / chốt ngay không tốn GPU** (`docs/project/TASKS.md` mục 0): K-14, K-15 (ngân sách, model), K-08 (số W3), K-01, K-02 (benchmark, nhóm giữ lại), K-03, K-04 (quyền, KU SFT); E-01..E-05.
4. **Thêm nguồn đã chọn**: viết adapter ingest + registry cho `phap_luat`, `anle` (đã tải), `fineweb2_vi`, `wiki_vi`, `finepdfs_edu_vi`; bước lọc chủ đề trước khi nạp `finepdfs_vi` (NS-20, NS-21); tải phần mở của `luanvan_vnu` (NS-12). Chờ duyệt thư xin phép cho VOER, Tia Sáng, NXB Giáo dục, bách khoa (NS-08).
5. **Chấm 50.000 mẫu bằng LLM** (R-11, R-12, K-06, K-12) trước khi chốt ngưỡng rule; cần dựng vLLM trên server (S-7).
6. **Contamination scan**: chọn danh sách benchmark (K-01) rồi viết lại `contamination.py` theo từng câu (L-01).
7. Điền license / owner còn "chưa rõ" trong `docs/sources/SOURCES.md`; chạy OCR GPU giáo trình (đã nối chuỗi VLM, S-30); OCR nội bộ VNPT (S-29, OI-01) nếu có khoá.
