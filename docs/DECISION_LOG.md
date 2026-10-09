# Nhật ký quyết định (DECISION_LOG)

> Ghi **mọi quyết định của chủ dự án** qua từng phiên làm việc: ngày, quyết định, lý do, trạng thái.
> Quyết định mới thêm vào cuối mục của phiên đó; quyết định bị thay thì không xoá, mà đánh dấu "thay bởi D-xx".
> Phần trước 2026-10-07 được ghi lại từ CLAUDE.md, tài liệu và lịch sử git (ngày lấy theo commit), không phải ghi tại chỗ.

Trạng thái: ✅ đã làm trong code | 🟡 đã chốt, chưa code | ❓ chưa quyết (đang chờ trả lời) | ⛔ đã bỏ / bị thay

## Các phiên trước 2026-10-07

| Mã | Ngày | Quyết định | Lý do / ghi chú | Trạng thái |
|---|---|---|---|---|
| P-01 | 2026-09-23 | Chỉ tải **phần tiếng Việt** của 3 bộ SEA (SEA-Instruct-2602, SEA-PILE-v2, SEA-PILE-v1/mC4), nguyên byte, không xử lý khi tải | Giữ raw nguyên gốc, xử lý tách riêng | ✅ |
| P-02 | 2026-09-23 | Quản lý thư viện **chỉ bằng uv** (`pyproject.toml` + `uv.lock`), không có requirements.txt | | ✅ |
| P-03 | 2026-09-28 | Đổi tên repo / gói thành `vi-corpus-etl` / `vi_corpus` (từ `sea-vi-crawler`) | Repo không còn chỉ là crawler SEA | ✅ |
| P-04 | 2026-09-28 | **Kiến trúc lai**: stage tự viết nối bằng Parquet trên đĩa + adapter NeMo Curator tùy chọn | Chạy được trên Jupyter không GPU, vẫn mở rộng được trên server | ✅ |
| P-05 | 2026-09-28 | **Không chia train/val/test ở tầng corpus**; chia sau ở tầng knowledge unit, theo `dedup_family_id` | Tránh rò rỉ giữa các bản gần trùng | ✅ (`split_bucket`) |
| P-06 | 2026-09-28 | Rights gate cấu hình trong source registry; `unknown` → quarantine (mặc định chỉ gắn nhãn, không loại) | Chất lượng và quyền là hai trục tách biệt | ✅ |
| P-07 | 2026-09-30 | Một thư mục script cho mỗi nguồn + một script tổng `scripts/run_all.py` | | ✅ |
| P-08 | 2026-09-30 | Giáo trình: **giữ sách tiếng Anh** (gắn nhãn ngôn ngữ, không loại) | Thêm token cho mục tiêu ≥300M | ✅ (`allowed_langs=("vi","en")`) |
| P-09 | 2026-09-30 | Giáo trình: quyền sử dụng **chưa quyết**, tạm `rights_status="unknown"` | Đã hỏi 2 lần, chưa có câu trả lời | ⛔ thay bởi Q10 (2026-10-08) |
| P-10 | 2026-09-30 | stbook: OCR bằng PaddleOCR (chỉ detect) + VietOCR (nhận dạng) | Recognizer của Paddle làm mất dấu tiếng Việt | ✅ |
| P-11 | 2026-10-06 | Pipeline dùng chung tham khảo **ViLA**: xem ViLA trước khi tự thiết kế | | ✅ |
| P-12 | trước 2026-10-07 | VJOL / VISTA: chỉ xử lý, không crawl; **không viết script cho tới khi biết cấu trúc dữ liệu** | | ❓ chờ dữ liệu |

## Phiên 2026-10-07: kế hoạch nâng cấp pipeline + W3/W4

Phiên này **chỉ ghi lại kế hoạch và quyết định, chưa viết code pipeline**; code làm ở phiên sau.
Bản kế hoạch chi tiết có sơ đồ: `.lavish/ke-hoach-w3-w4.html` (trang review Lavish, chưa đưa vào git).

### D-01 · Đếm token bằng tokenizer thật — ✅ đã code (2026-10-08)

- Module mới `vi_corpus/pipeline/tokens.py`, dùng thư viện `tokenizers` (Rust), chỉ nạp `tokenizer.json`, đếm theo lô (`encode_batch`).
- Tokenizer mặc định: **Qwen3** (`Qwen/Qwen3-0.6B`). Vẫn cho đổi bằng `--tokenizer` (`hf:<repo>`, `tiktoken:<enc>`, `words`).
- Lưu cả `word_count` và `token_count`; manifest / report ghi tên tokenizer và tỷ lệ token/từ theo nguồn. Đếm lại sau mọi bước làm đổi text. `knowledge.py` dùng chung bộ đếm.
- Lý do: số từ (số âm tiết) thấp hơn số token thật, không dùng để đo mục tiêu ≥300M được.

### D-02 · Nhận diện ngôn ngữ bằng fastText — ✅ đã code (2026-10-08)

- Mô hình **lid.176.bin** (Facebook, 176 ngôn ngữ), gói `fasttext-numpy2-wheel`. Phương án thay: GlotLID.
- Tỷ lệ chữ có dấu không còn quyết định ngôn ngữ, chuyển thành số đo chất lượng `vi_diacritic` (để bắt tiếng Việt không dấu).
- ~~Có bộ 500 mẫu gán nhãn tay để so heuristic cũ với fastText.~~ ⛔ thay bởi R-02 (nhãn lấy từ LLM lớn, không gán tay).

**Giải thích "dự đoán theo đoạn" và `min_lang_score`** (theo yêu cầu):

1. fastText nhận **một dòng** văn bản và trả về nhãn kèm độ tin cậy, ví dụ `vi 0,98`.
2. Một mẫu (vd một đoạn sách ~600 từ) có thể trộn ngôn ngữ: phần lớn tiếng Việt, xen một trích dẫn hay công thức tiếng Anh. Nếu đưa cả mẫu vào một lần thì chỉ nhận được một nhãn chung, không biết mẫu trộn bao nhiêu.
3. Vì vậy tách mẫu thành các đoạn văn (theo dòng trống), cho fastText đoán **từng đoạn**, rồi cộng lại có trọng số theo độ dài đoạn. Văn bản rất dài thì chỉ lấy tối đa ~50 đoạn rải đều để giữ chi phí cố định.
4. Ví dụ: 10 đoạn dài bằng nhau, 8 đoạn `vi` (tin cậy 0,98), 2 đoạn `en` (0,95).
   - `lang_mix = {"vi": 0,8, "en": 0,2}` (phần độ dài thuộc mỗi ngôn ngữ).
   - `language = "vi"` (ngôn ngữ chiếm nhiều nhất).
   - `lang_score` = độ dài × độ tin cậy của các đoạn `vi` / tổng độ dài = 0,8 × 0,98 ≈ **0,78**. Số này cao khi văn bản vừa **thuần** một ngôn ngữ vừa được đoán **chắc chắn**.
5. `min_lang_score` là ngưỡng tối thiểu cho `lang_score`, đặt riêng cho từng nguồn trong `SourceProfile` (khởi điểm 0,65). Mẫu ví dụ trên (0,78) qua ngưỡng. Một trang nửa Việt nửa Anh có `lang_score` ≈ 0,49, nên bị gắn reason code `low_lang_score` (lỗi cứng hoặc trừ điểm, chốt ở bước chỉnh ngưỡng W3). Giáo trình cho phép cả `vi` và `en` nên có thể đặt ngưỡng thấp hơn.

### D-03 · Dedup 4 tầng, mở rộng được — 🟡 một phần đã code (2026-10-08): L0, L1 (bao hàm), L2 theo văn bản trong RAM; L2 theo shard chờ D-06, L3 chờ G-08

- L0 exact: chỉ mục sha256 toàn cục trên đĩa (`state/dedup_index/exact/`), dùng lại giữa các lần chạy.
- L1 đoạn / dòng: đếm tần suất hash đoạn trên toàn nguồn (chung cơ chế với D-04 mục C).
- L2 fuzzy: MinHash-LSH theo shard, shuffle theo băng trên đĩa → cạnh ứng viên → connected components → chọn đại diện theo điểm chất lượng + ưu tiên nguồn. Kèm adapter NeMo Curator fuzzy dedup (GPU) cùng đầu vào / ra.
- L3 ngữ nghĩa (W3): SemDeDup trên embedding (k-means, cosine ≥ ngưỡng trong cụm). *Cập nhật theo D-11 / R-03:* embedding là Qwen3-Embedding (không còn e5); 0,92 chỉ là ngưỡng của e5, phải hiệu chỉnh lại theo số chiều và mô hình mới.
- `dup_kind` thêm `semantic`; `dedup_family_id` giữ nguyên ý nghĩa. Chữ ký MinHash cache theo sha256 text.
- Phạm vi: đã chốt ở Q4 (phiên 2026-10-08): toàn cục giữa mọi nguồn cho văn bản CPT, SEA-Instruct dedup riêng.

### D-04 · Xóa dòng lặp — ✅ đã code (2026-10-08)

Thứ tự trong `prepare`: normalize → clean_lines (trong văn bản) → clean_lines (liên văn bản) → chunk → đếm token.

- A. Trong một văn bản: gộp dòng giống hệt liền nhau; dòng ngắn (< 10 từ) lặp ≥ 3 lần thì giữ lần đầu; bỏ dòng chỉ là số trang. **Văn bản có code hoặc bảng được miễn mục A** (C-07, 2026-10-09).
- B. Header / footer sách: **đã có sẵn** trong `vi_corpus/common/pdf_text.clean_pages` (dòng ở 2 dòng đầu / cuối trang, thay số bằng `#`, lặp ≥ 3 trang thì bỏ; bỏ dòng chỉ có chữ số), đang dùng cho stbook và giáo trình. Giữ nguyên, chỉ thêm thống kê số dòng bị bỏ. (Bản kế hoạch đầu tiên ghi nhầm là phải viết mới.)
- C. Liên văn bản (chỉ nguồn web): lượt 1 đếm hash dòng theo nguồn, lượt 2 xóa dòng xuất hiện ở ≥ N văn bản (N = max(20; 0,01% số văn bản), chỉnh trong config).
- Ghi `lines_removed`, `chars_removed`; report liệt kê các dòng bị xóa nhiều nhất. `source_sha256` vẫn là hash của text gốc.

### D-05 · Parser DOCX, DOC/PPT, DJVU, HTML, EPUB — ✅ đã code (2026-10-08; chưa thử trên file thật)

python-docx; LibreOffice headless (đổi doc/ppt sang docx/pptx); djvulibre (`djvutxt`, trang không có chữ thì OCR); trafilatura; ebooklib. Gom vào `vi_corpus/common/parsers/`, nhận định dạng bằng magic bytes, thêm `parser_version` và `--retry-skipped`. Còn chờ Q5.

### D-06 · Chạy theo shard để gỡ bottleneck — 🟡 đã đồng ý

- Đơn vị = shard (~50k văn bản hoặc ~256 MB); mỗi file parquet / jsonl.gz SEA là một shard; sách gom theo nhóm.
- Gộp các stage xử lý từng văn bản thành **một lượt** đọc / ghi trên mỗi shard; stage toàn cục (đếm dòng, dedup, reduce) chạy map → shuffle trên đĩa → reduce.
- Checkpoint theo shard (`<stage>/part-*.parquet` + `_done/`); truyền dữ liệu bằng Arrow thay cho JSON `payload`; Ray Data streaming để CPU và GPU chạy cùng lúc.
- Bước đầu tiên: **đo tốc độ hiện tại** (docs/s, MB/s, RAM tối đa theo stage) trên 100k bản ghi ở server.
- Thứ tự làm: nền shard → (D-01, D-02, D-04 A, D-05 song song) → (D-04 C, D-03 L0 + L2, executor Ray) → W3 → W4.

### D-07 · Kế hoạch W3 + W4 — 🟡 đã đồng ý (Mục 7, 2026-10-08); thứ tự và cách lấy mẫu chỉnh bởi R-11, "review tay" thay bằng điểm LLM theo Q6

- W3: chạy full và đo token thật; đường cong retention theo ngưỡng + review tay để chốt ngưỡng; semantic dedup (D-03 L3); contamination scan 13-gram với benchmark tiếng Việt; thêm nguồn sách / khoa học / kỹ thuật (Q7); đếm ≥500k train candidate.
- W4: pool có nhãn 50k (lấy mẫu phân tầng, LLM judge chấm 0–5 + 500 mẫu người duyệt); so 3 bộ lọc (fastText classifier, đầu hồi quy trên embedding e5, bi-encoder nhỏ), chia theo `dedup_family_id`; cột `quality_model_score`; embedding map v1 (UMAP fit trên mẫu 200k + datashader); bảng độ phủ domain; tương quan với judge.

### D-08 · Tổ chức tài liệu — ✅ đã làm

- Mọi file `.md` trừ `CLAUDE.md` nằm trong `docs/` (đã chuyển `README.md`, `PROJECT_ARCHITECTURE.md`; `pyproject.toml` trỏ `readme = "docs/README.md"`).
- Tạo file này (`docs/DECISION_LOG.md`) để ghi mọi quyết định qua các phiên.
- Sơ đồ gom vào **một thư mục** `docs/diagrams/`: `00_pipeline.svg/png` (luồng chính) + `01..12_*.svg` (luồng con từng module, vẽ theo code hiện tại; khung nét đứt = phần đã chốt nhưng chưa code). Danh sách ở `docs/PIPELINE.md` mục 2.

### Câu hỏi còn mở (từ phiên 2026-10-07)

| Mã | Câu hỏi | Đề xuất | Trạng thái |
|---|---|---|---|
| Q1 | Tokenizer | Qwen3 | ✅ chốt qua D-01 (có thể đổi nếu chốt mô hình đích) |
| Q2 | Mô hình fastText | lid.176.bin | ✅ chốt qua D-02 |
| Q3 | Backend fuzzy dedup | tự viết CPU theo shard + adapter Curator GPU | ✅ chốt qua D-03 |
| Q4 | Phạm vi dedup | toàn cục giữa mọi nguồn | ✅ đã code (thứ tự ưu tiên thay bởi R-15) |
| Q5 | Cài LibreOffice + djvulibre trên server? | được cài | ✅ hướng dẫn / ✅ code D-05 |
| Q6 | LLM judge gán nhãn W4 | mô hình mở trên GPU server (vLLM) | ✅ chốt 2026-10-08: xếp tầng rule + mô hình nhỏ (xem dòng Q6 phiên 2026-10-08) |
| Q7 | Nguồn mới cho W3 | Wikipedia vi, pháp luật, FineWeb-2 | ✅ chốt 2026-10-08 |
| Q8 | Kích thước shard | SEA: 1 file = 1 shard; sách ~20.000 đoạn/shard | ✅ chốt mặc định kỹ thuật (R-16) |
| Q9 | Danh sách benchmark cho contamination scan (VMLU, UIT-ViQuAD, ViMMRC, XQuAD-vi, MLQA-vi, ...) | | ⏸ hoãn (2026-10-08: "để lại, chưa chốt") |

## Phiên 2026-10-08: trả lời các câu hỏi còn mở

Phiên này vẫn **chỉ ghi**, chưa viết code pipeline. Câu hỏi được trình bày lại dễ hiểu ở `.lavish/cau-hoi-mo.html`.

| Mã | Quyết định | Lý do / ghi chú | Trạng thái |
|---|---|---|---|
| Q4 | Dedup **toàn cục giữa mọi nguồn** cho văn bản thường (CPT). Khi trùng, giữ bản theo thứ tự ưu tiên stbook > giáo trình > SEA-PILE v2 > SEA-PILE v1, rồi mới xét điểm chất lượng. SEA-Instruct (hỏi-đáp, SFT) chỉ dedup trong chính nó. | Bỏ trùng giữa nguồn, chia train/test theo họ trùng không rò rỉ, không xoá nhầm cặp hỏi-đáp. Thứ tự ưu tiên nằm trong config. | ✅ đã code (thứ tự ưu tiên thay bởi R-15) |
| Q4-bổ sung | Chủ dự án sẽ **chạy từng nguồn một** (SEA trước, rồi stbook, ...), nên cần cách **dedup thêm một lần nữa sau khi chạy xong tất cả**. | Cách làm đang bàn: D-09. | ❓ |
| Q5 | **Cài luôn** LibreOffice + djvulibre trên server; hướng dẫn cài đã viết vào `docs/README.md` Phần E. | Đọc được .doc / .ppt / .djvu của giáo trình. | ✅ hướng dẫn / ✅ code D-05 |
| Q6 | Chủ dự án yêu cầu suy nghĩ kĩ: chấm chất lượng bằng **model hay bằng rule**. | Đang bàn, xem `.lavish/cau-hoi-mo.html`. | ❓ |
| Q8 | Chủ dự án chưa rõ khác nhau giữa **shard** và **sample** ("tại sao 100 cuốn một shard?"). | Đã giải thích lại, đang bàn. | ❓ |
| Q9 | Danh sách benchmark: **để lại, chưa chốt**. | | ⏸ hoãn |
| D-09 | **Dedup 2 vòng**. Vòng 1 (khi chạy từng nguồn): bỏ trùng trong nguồn và lưu "dấu vân tay" của mọi văn bản còn sống vào `state/dedup_index/<nguồn>/` (doc_id, sha256, chữ ký MinHash, điểm, mức ưu tiên; không có text, ~0,5 KB/văn bản). Vòng 2 (`scripts/dedup_global.py`, sau khi chạy xong các nguồn): chỉ đọc kho dấu vân tay của mọi nguồn, gộp họ trùng toàn cục, giữ bản theo ưu tiên Q4, ghi `verdict.parquet`. | Chủ dự án chạy từng nguồn một; vòng 2 không đọc lại text nên rẻ; thêm nguồn mới chỉ cần chạy vòng 1 cho nguồn đó rồi chạy lại vòng 2. Hệ quả: clean từng nguồn là bản tạm; knowledge unit và chia train/test làm **sau vòng 2**. | ✅ đã code (2026-10-08) |
| Mục 5 | Đồng ý parser DOCX (python-docx), HTML (trafilatura), EPUB (ebooklib); nhận dạng file theo nội dung đầu file; `--retry-skipped`. **Thêm thư viện bằng uv** (`uv add`, commit cả `pyproject.toml` + `uv.lock`), không dùng pip. | | ✅ đã code (2026-10-08) |
| Q7 | Nguồn mới cho W3: **Wikipedia tiếng Việt, văn bản pháp luật, FineWeb-2 (`vie_Latn`)**. Đã ghi vào `docs/SOURCES.md` (key `wiki_vi`, `phap_luat`, `fineweb2_vi`). Không chọn lúc này: Wikisource/Wikibooks, TCVN, code. | Quyền rõ nhất, dễ lấy; Wikipedia + pháp luật thêm kiến thức có cấu trúc, FineWeb-2 thêm khối lượng. | 🟡 |
| Q10 | Sách / giáo trình chưa rõ bản quyền: **xử lý đầy đủ, chỉ gắn nhãn quarantine**; quyết giữ hay loại lúc phát hành (W8) bằng `--rights-gate enforce`. Thay cho P-09. | Giữ đủ dữ liệu để thí nghiệm, đổi ý không phải chạy lại. | ✅ (đúng hành vi hiện tại) |
| Mục 7 | Đồng ý kế hoạch W3 / W4 (xem D-07). | | 🟡 |
| Q6 | **Chấm chất lượng xếp tầng**: rule hiện có chạy trước cho mọi nguồn; sau đó một **mô hình nhỏ** chấm "giá trị kiến thức". Nhãn để dạy mô hình nhỏ: **LLM mở chạy trên GPU server chấm 50.000 mẫu** theo một thang điểm 0–5 viết rõ (cách B). **Bỏ cách A** (lấy nguồn làm nhãn: sách = tốt, web = thường). **Không có bước người chấm 500 mẫu.** | Cách A học đường tắt (vd stbook toàn sách chính trị → "chính trị = tốt"), không định nghĩa được chất lượng. Chi phí cách B không tăng theo cỡ corpus: ước lượng ~2–3 GPU-giờ cho 50.000 mẫu, còn LLM chấm toàn bộ 30B token cần ~2.700 GPU-giờ nên không làm (ước lượng, cần đo trên 1.000 mẫu). Chủ dự án không có thời gian chấm tay. | 🟡 |
| Q6-thang | Thang điểm 0–5 (bản nháp, chưa duyệt): 0 không có thông tin (quảng cáo, spam, link, lỗi OCR nặng) · 1 ít thông tin, lẫn quảng cáo / chuyện vặt · 2 có thông tin nhưng rời rạc, hời hợt · 3 thông tin đúng, mạch lạc về một chủ đề · 4 giải thích có hệ thống, có khái niệm, ví dụ · 5 chất lượng giáo trình. | Ngưỡng giữ chốt ở W3 theo đường cong giữ lại từng nguồn. | ❓ chờ duyệt |
| Q8-bổ sung | Chủ dự án hỏi: khi train thì lấy theo sample hay shard, sample cỡ nào là hợp lý. | Đang bàn (xem dưới). | ❓ |
| Q8-đính chính | Lượt trước em nói "600 từ không ổn vì vượt giới hạn 512 token của mô hình embedding". **Rút lại**: cỡ đoạn nên do đơn vị quyết định giữ / loại và đơn vị chấm điểm quyết định, không do mô hình embedding (embedding là bước phụ, có thể tự thích nghi). Đo trên văn bản tiếng Việt trong docs/: Qwen3 ~1,60 token/từ, e5 ~1,51 token/từ → 600 từ ≈ 960 token Qwen3. | Chủ dự án hỏi lại: có nên cắt theo giới hạn của mô hình embedding không, khi embedding chủ yếu để vẽ 2D. | ✅ đã thay bằng D-10 |
| D-10 | **Cắt đoạn (sample) theo token Qwen3**, không theo số từ, không theo giới hạn của mô hình embedding: mục tiêu ~1.024 token, tối đa 2.048, đoạn cuối < 256 token gộp vào đoạn trước. **Thứ tự ưu tiên chỗ cắt**: dòng tiêu đề (`Chương 3`, `CHƯƠNG III`, `Bài 5`, `Mục 2.1`; chỉ coi là tiêu đề khi dòng < ~15 từ và không kết thúc bằng dấu chấm) → dòng trống (hết đoạn văn) → hết câu → cắt cứng. Bài web giữ nguyên, chỉ bài > 2.048 token mới cắt theo cùng quy tắc. Embedding tự thích nghi theo đoạn (bản đồ: 512 token đầu; semantic dedup: chốt ở W3). Khi xuất CPT: nhóm theo `parent_doc_id`, sắp `chunk_index`, nối các đoạn liền nhau còn giữ tới độ dài ngữ cảnh, gặp đoạn bị loại thì ngắt khối. | Cỡ đoạn do đơn vị chấm điểm / giữ-loại quyết định: LLM cần đủ ngữ cảnh để chấm, đoạn quá dài thì loại oan, nên cùng cỡ với bài web để điểm so sánh được; CPT không bị ảnh hưởng vì nối lại khi xuất. | ✅ đã code (2026-10-08); embedding theo đoạn chờ G-08 |
| Q6-kiểm | Kiểm LLM chấm mà không cần người chấm: chấm lại 1.000 mẫu lần hai với câu lệnh viết khác (đo độ ổn định); report in 5 ví dụ mỗi mức điểm để chủ dự án lướt đọc (~30 ví dụ, không phải chấm). | Chủ dự án không có thời gian chấm tay. | 🟡 |
| Q11 | Chủ dự án muốn đổi sang **mô hình embedding nhiều chiều hơn** (hiện `intfloat/multilingual-e5-base`, 768 chiều, đọc tối đa 512 token). | Đã bàn tác hại (đĩa, tốc độ, RAM gom cụm, phải chỉnh lại ngưỡng, sửa pooling / tiền tố) và lợi ích (đọc trọn đoạn 1.024–2.048 token theo D-10). | ✅ chốt bằng D-11 |
| D-11 | **Đổi mô hình embedding sang `Qwen/Qwen3-Embedding-0.6B`** (1.024 chiều, đọc tối đa 32K token, cùng họ tokenizer Qwen3 của D-01). Tính đủ 1.024 chiều, **lưu bản cắt 256 chiều** (Matryoshka) cho gom cụm / semantic dedup, chỉ lưu đủ chiều khi cần. Dự phòng: `BAAI/bge-m3` nếu Qwen3-Embedding không chạy được trên server. Không dùng bản 8B. Trước khi chốt hẳn: đo trên 2.000 đoạn ở server (tốc độ, độ tách cụm so với e5). | Đọc trọn đoạn → semantic dedup không nhầm vì chung phần đầu trang; cắt 256 chiều để đĩa (~15 GB fp16 cho 30 triệu đoạn) và RAM gom cụm không phình. Ước lượng ~30–60 GPU-giờ cho 30B token. Phải sửa `embed.py` (pooling token cuối, bỏ tiền tố `passage: `) và chỉnh lại ngưỡng cosine của semantic dedup. | 🟡 |

## Phiên 2026-10-08 (tiếp): rà soát toàn bộ chiến lược

Phiên này vẫn **chỉ chốt + bàn**, chưa viết code. Rà lại D-01..D-11 và Q1..Q11, tìm mâu thuẫn và chỗ chưa tối ưu. Đã sửa tại chỗ: D-02 (gạch bỏ 500 mẫu gán tay), D-03 (L3 đổi tham chiếu sang embedding Qwen3), `CLAUDE.md` (thêm mục Infrastructure).

| Mã | Quyết định | Lý do / ghi chú | Trạng thái |
|---|---|---|---|
| R-01 | **Rights gate để sau; hiện bypass hoàn toàn**: không loại, không chặn dedup / knowledge unit / chia split theo quyền. Vẫn giữ cột `rights_status` (chỉ là nhãn, không tốn gì). Dedup chọn bản giữ theo ưu tiên nguồn Q4 như cũ. | Chủ dự án: chưa cần gate. Rủi ro đã nhận: nếu sau này bật `--rights-gate enforce` mà nguồn được ưu tiên bị loại thì bản trùng ở nguồn khác đã bị đánh dấu trùng. Giảm nhẹ bằng điều kiện thiết kế: vòng 2 của D-09 chỉ **đánh dấu** (`verdict.parquet`), không xoá text ở `interim/`, nên chạy lại vòng 2 là rẻ. Q10 giữ nguyên ý nhưng "quarantine" hiện không có tác dụng loại. | ✅ đã code (chỉ gắn nhãn) |
| R-02 | Sửa D-02: **bỏ bộ 500 mẫu gán tay** (mâu thuẫn với Q6 "không có thời gian chấm tay"). Nhãn ngôn ngữ "đúng" lấy từ **LLM lớn** trong chính lượt chấm ~50.000 mẫu của Q6 (cách B): cùng một lần gọi, LLM trả thêm `language` / mức trộn ngôn ngữ; dùng để so fastText `lid.176` với GlotLID. Mô hình nhỏ học cách chấm từ 50.000 mẫu này chỉ áp dụng cho **chất lượng** (Q6); ngôn ngữ vẫn là fastText, không train thêm. | Cách hiểu của Claude về câu "LLM lớn chấm 50.000 sample rồi train model nhỏ"; chủ dự án xác nhận hiểu đúng (N-01, 2026-10-08). | 🟡 |
| R-03 | Sửa D-03 L3: semantic dedup dùng embedding Qwen3-Embedding (D-11), **ngưỡng cosine hiệu chỉnh lại** (0,92 là của e5). | | 🟡 |
| R-04 | Sửa `embed.py` theo D-10 / D-11: bỏ `MAX_CHARS = 6000` (cắt theo token, tới 2.048), pooling token cuối thay vì trung bình, bỏ tiền tố `passage: `. | Code ở phiên sau. | 🟡 |
| R-05 | **Thứ tự stage: làm rẻ trước, nhúng sau**: rule → fastText ngôn ngữ → exact + fuzzy dedup → embedding → semantic dedup (L3). GPU chỉ nhúng phần còn sống sau lọc rẻ. | Giảm số đoạn cần nhúng; kéo theo cỡ kho lưu embedding nhỏ lại. | 🟡 một phần: thứ tự stage đã là rule → ngôn ngữ → dedup → nhúng, thêm `--embed-scope kept`; L3 chưa có |
| R-06 | **Hạ tầng: 4× GPU A100 40GB** (đã ghi `CLAUDE.md`). Chủ dự án chạy được khoảng **30 giờ GPU** → chi phí GPU không còn là lý do bỏ L3 semantic dedup; **giữ L3** theo D-03. | Còn phải làm rõ "30 giờ" là tổng giờ-GPU hay giờ đồng hồ × 4 GPU, và chia cho OCR / judge / embedding / huấn luyện mô hình nhỏ W6: N-02. | ✅ ghi hạ tầng / ❓ phân bổ |
| R-07 | Chấp nhận giới hạn của Q6-kiểm: kiểm chỉ đo **độ ổn định** của LLM chấm (chấm lại bằng câu lệnh khác), **không đo độ đúng**. Report phải ghi rõ giới hạn này. | Chủ dự án không có thời gian chấm tay. | 🟡 |

### Câu hỏi đang chờ (sinh ra từ đợt rà soát này)

Chủ dự án muốn bàn các mục này để có đủ thông tin trước khi code. Đề xuất chi tiết nằm trong phản hồi của phiên, không lặp lại ở đây.

| Mã | Vấn đề | Đề xuất của Claude (tóm tắt) | Trạng thái |
|---|---|---|---|
| N-01 | Hiểu đúng R-02 chưa (LLM lớn gán cả nhãn ngôn ngữ, mô hình nhỏ chỉ cho chất lượng)? | Đúng như R-02 | ✅ chủ dự án xác nhận hiểu đúng |
| N-02 | "30 giờ GPU" là gì; phân bổ cho OCR / judge / embedding / W6. OCR có thể là khoản lớn nhất mà chưa đo. | Đo số trang + trang/giây trước, giữ dự phòng cho W6 | ✅ đồng ý đo trước (R-08, `docs/SERVER_TASK.md` S-1..S-3); ❓ "30 giờ" là tổng giờ-GPU hay giờ đồng hồ × 4 GPU thì chưa trả lời |
| N-03 | Thuật ngữ: *sample* (đoạn D-10), *knowledge unit*, *train candidate* đếm cái nào. | 1 đoạn còn sống = 1 candidate CPT; cặp hỏi-đáp = instruction SFT | ✅ chủ dự án giao Claude tự định nghĩa → R-09 |
| N-04 | Đơn vị dedup (văn bản hay đoạn) và dữ liệu L3: kho dấu vân tay của D-09 không có embedding nhưng L3 ở vòng 2 cần embedding. | Bản N-04 v2 | ✅ chủ dự án duyệt → R-19 |
| N-05 | Thứ tự W3/W4 là vòng tròn; chạy tập con trước; thiết kế mẫu 50.000. | Chấm 50.000 mẫu trước khi chốt ngưỡng; mẫu lấy trước lọc rule | ✅ đồng ý → R-11 |
| N-06 | Thang chấm, domain, LLM judge, SEA-Instruct có qua cascade không. | LLM trả một lượt: điểm, domain, ngôn ngữ, cờ lỗi | ✅ đồng ý đề xuất và mô hình judge → R-12; ❓ danh sách domain (chủ dự án cung cấp sau) |
| N-07 | Mô hình đích để huấn luyện (ảnh hưởng tokenizer, ngữ cảnh, W6). | Mặc định Qwen3 tới khi chốt | ❓ chủ dự án chưa hiểu câu hỏi → đã giải thích lại, chờ trả lời (R-13) |
| N-08 | D-05 parser: ppt / docx / djvu chỉ 27 file (~2%) giáo trình; HTML / EPUB hiện chưa có nguồn dùng. | Đề xuất làm gọn / hoãn | ✅ chủ dự án: giữ như cũ, vì sẽ thêm dataset khác → R-14 |
| N-09 | Nguồn mới: cách lấy `phap_luat`; thứ tự ưu tiên nguồn khi trùng; FineWeb-2 trùng nhiều. | | ✅ một phần → R-15; ❓ cách lấy `phap_luat`, thứ tự trong nhóm "các dataset khác" |
| N-10 | Chốt nốt Q8 (shard), Q9 (benchmark), P-12 (VJOL/VISTA). | | ✅ → R-16 (P-12 vẫn chờ cấu trúc VJOL) |
| N-11 | Phiên đo trên server trước khi viết code lớn. | | ✅ đồng ý; hướng dẫn chi tiết ở `docs/SERVER_TASK.md` → R-17 |

### Chốt lượt 3 (2026-10-08): trả lời N-01..N-11

Vẫn **chỉ chốt + bàn**, chưa code. Mục nào ✅/🟡 dưới đây là việc cho phiên code sau; mục ❓ còn bàn tiếp.

| Mã | Quyết định | Lý do / ghi chú | Trạng thái |
|---|---|---|---|
| R-08 | **Đo trước khi quyết ngân sách GPU** (N-02): đếm tổng số trang PDF (stbook, giáo trình), tỉ lệ trang scan, tốc độ OCR trang/giây trên 1 GPU, rồi quy ra giờ-GPU = trang cần OCR ÷ trang/giây ÷ 3600. Nếu OCR quá tốn thì xét OCR chọn lọc (bỏ sách chất lượng thấp trước khi OCR). Dành dự phòng giờ GPU cho W6 (huấn luyện mô hình nhỏ). | OCR có thể là khoản GPU lớn nhất mà chưa có số đo; `paddlepaddle` đang là bản CPU nên bước phát hiện chữ có thể là nút thắt (giả thuyết, cần đo). Còn mở: "30 giờ GPU" là tổng giờ-GPU hay giờ đồng hồ × 4 GPU. | ✅ đồng ý đo; ❓ nghĩa của "30 giờ" |
| R-09 | **Định nghĩa thuật ngữ** (Claude định nghĩa theo bảng Milestone chủ dự án gửi 2026-10-08): **mẫu / đoạn (chunk)** = đơn vị ~1.024 token của D-10, là đơn vị lọc, chấm điểm, nhúng, dedup L3. **Candidate (knowledge / train candidate)** = một đoạn CPT còn sống sau mọi bước lọc và dedup, có đủ provenance; các chỉ tiêu "≥100k knowledge candidates" (W2), "≥500k train candidates" (W3), "≥1,2M candidates" (W4) đều đếm đoạn này. **Instruction** (W5 "≥2M instructions across ≥3 depths") = một cặp hỏi-đáp sinh từ facts / concepts của đoạn (atomization) hoặc có sẵn (SEA-Instruct); *depth* = mức độ sâu của câu hỏi (gợi ý: nhớ lại → giải thích → suy luận), chốt ở W5. Trong code (`KU_SCHEMA`): KU loại CPT = candidate, KU loại SFT = instruction. | Kiểm tính hợp lý: 1,2M đoạn × ~1.000 token ≈ 1,2B token, khớp với "≥1,5B clean" của W4 (đọc từ ảnh bảng). | ✅ đã code (KU CPT = candidate, KU SFT = instruction; `cpt_blocks.parquet`) |
| R-10 | **Tin cậy số liệu bảng Milestone**: ảnh chụp nhỏ, số liệu W3 (raw ≥30B / clean ≥15B) đọc từ ảnh không khớp dãy tăng dần của các tuần khác (raw 1B → ? → 5B → 7B; clean 0,3B → ? → 1,5B → 2B → 3B → 10B), có thể là ≥3B / ≥1,5B hoặc nhỏ hơn. Đã ghi nguyên số đọc được vào `docs/PROGRESS.md` kèm cảnh báo. **Đính chính của Claude**: câu "300M token đã vượt xa nên nút thắt là chất lượng" chỉ đúng cho mốc W2; mốc cuối là **≥10B clean ở W7**, nên tỉ lệ giữ lại (retention) quan trọng và lọc quá chặt có thể làm hụt mốc. | Cần chủ dự án kiểm lại số W3 với bảng gốc. | ❓ chờ kiểm số W3 |
| R-11 | **Thứ tự việc** (N-05): (1) chạy từng **tập con (subset)** của từng nguồn; (2) lấy mẫu ~50.000, cho LLM lớn chấm; (3) dùng điểm LLM làm "đáp án" để chỉnh ngưỡng rule (rule loại nhầm bao nhiêu mẫu điểm 4–5, bỏ sót bao nhiêu mẫu điểm 0–1); (4) chạy toàn bộ. **Cách lấy mẫu 50.000**: lấy sau exact dedup và nhận diện ngôn ngữ, **trước** lọc rule; khoảng 70% mẫu qua rule, 30% mẫu bị rule loại (để đo rule sai ở đâu); phân tầng theo nguồn và độ dài. Chủ dự án: "mục tiêu là xử lý data trên từng subset một trước". | Gỡ vòng tròn W3 (chỉnh ngưỡng) cần đáp án của W4 (điểm LLM); bỏ "review tay" của D-07. | ✅ lấy mẫu (`scripts/make_judge_sample.py`) / 🟡 chấm + chỉnh ngưỡng |
| R-12 | **Chấm chất lượng** (N-06): duyệt **thang 0–5 bản nháp** của Q6-thang (chỉnh sau khi xem ~30 ví dụ). LLM trả **một lượt gồm điểm, domain, ngôn ngữ, cờ lỗi** (spam, OCR hỏng, quảng cáo...). **SEA-Instruct** (hỏi-đáp) đi qua cascade với **rubric riêng đơn giản** (đáp án đúng, đầy đủ, đúng tiếng Việt), khoảng 5.000 mẫu nằm trong ngân sách 50.000. **Không** so sánh hai mô hình khác cỡ với nhau. **Mô hình judge đề xuất**: `Qwen3.5-27B` (dense, bf16 ~54 GB, chạy 2 thẻ A100 40GB song song tensor, 2 bản trên 4 thẻ); **chủ dự án xác nhận 2026-10-08: dùng đúng mô hình này, không cần mô hình dự phòng, không so sánh các judge**. Tiêu chí chọn: tiếng Việt tốt, vừa 2 thẻ. Chưa có điểm Vietnamese cụ thể của riêng Qwen3.5 (xem phản hồi phiên). **Danh sách domain**: chưa có, chủ dự án sẽ cập nhật sau (mục tiêu hiện tại là xử lý từng subset). | | 🟡 (mô hình judge đã chốt) / ❓ danh sách domain |
| R-13 | **Mô hình đích** (N-07): chủ dự án chưa hiểu câu hỏi; đã giải thích lại bằng ví dụ may đồ. Hiện **mặc định Qwen3** (tokenizer Qwen3 đo token, mô hình nhỏ W6 chọn họ Qwen3) cho tới khi chủ dự án chốt. Không chặn việc code. | Bảng Milestone: W6 dùng "mini model" (VMLU / Lịch sử / Địa lý / Tổng hợp), W8 "production-scale injection" → có thể là hai mô hình khác nhau. | ❓ chờ trả lời |
| R-14 | **D-05 giữ nguyên như đã đồng ý ở Mục 5** (python-docx, trafilatura, ebooklib, LibreOffice cho doc / ppt, djvulibre cho djvu). Chủ dự án: "sau còn thêm các dataset khác nữa"; bảng Milestone W1 cũng ghi parse PDF / DOCX / HTML / EPUB. | Claude từng đề xuất hoãn HTML và bỏ EPUB vì chưa có nguồn dùng (chỉ 27 file ≈ 2% giáo trình là ppt / docx / djvu); chủ dự án chọn giữ. | ✅ đã code (2026-10-08) |
| R-15 | **Thứ tự ưu tiên khi trùng (thay Q4-thứ-tự cũ stbook > giáo trình > SEA-PILE v2 > v1)**: **VJOL → SEA → stbook → các dataset khác**. Dedup vẫn toàn cục (Q4), SEA-Instruct dedup riêng. **`phap_luat`: chưa có bộ dữ liệu sẵn**, cách lấy (crawl) chưa quyết. FineWeb-2 trùng nhiều với SEA: đo 1 shard trước khi tải hết (tùy chọn, `docs/SERVER_TASK.md` S-9). | Cách hiểu của Claude: "thứ tự ưu tiên" là thứ tự giữ bản khi trùng. Còn mở: (a) trong nhóm "SEA": v2 trước v1, SEA-Instruct không tính vì dedup riêng; (b) thứ tự trong nhóm "các dataset khác" (giáo trình, VISTA, wiki_vi, FineWeb-2...); (c) khi bản ưu tiên cao là OCR lỗi còn bản thấp hơn sạch, có xét điểm chất lượng trước không (đề xuất: xét điểm trước nếu chênh lớn). | ✅ đã code (`SOURCE_PRIORITY`, cấu hình được) / ❓ (a)(b)(c) |
| R-16 | **Chốt nốt**: Q8 shard: mặc định kỹ thuật (SEA: 1 file = 1 shard; sách ~20.000 đoạn / shard), không bàn thêm. Q9: **danh sách benchmark để trống, không lọc contamination ở giai đoạn này** (scan chạy được nhưng với danh sách rỗng; bảng Milestone W6 đánh giá bằng VMLU, nên nếu sau này thêm VMLU vào danh sách thì phải scan trước khi huấn luyện W6). P-12: chủ dự án sẽ cung cấp cấu trúc thư mục và dữ liệu VJOL sau; chưa viết script VJOL. | | ✅ (scan 13-gram đã code, danh sách rỗng) / ⏸ (P-12) |
| R-17 | **Việc chạy trên server** được ghi thành `docs/SERVER_TASK.md` (9 việc S-1..S-9, 5 việc chạy ngay bằng script có sẵn, 4 việc chờ code). | Chủ dự án cần biết cách làm. | ✅ file đã tạo |
| R-18 | **Cập nhật `docs/PROGRESS.md`** theo bảng Milestone đầy đủ (ảnh chủ dự án gửi 2026-10-08). | | ✅ |

### Câu hỏi còn mở sau lượt 3

| Mã | Vấn đề | Trạng thái |
|---|---|---|
| N-04 | Đơn vị dedup, luồng dữ liệu L3, chọn đại diện, family cho chia split: bản N-04 v2 đã phân tích trong phản hồi, chờ duyệt. | ❓ |
| N-02b | "30 giờ GPU" nghĩa là gì. | ❓ |
| N-06b | Xác nhận mô hình judge; danh sách domain (chủ dự án cung cấp sau). | ❓ |
| N-07 | Mô hình đích / mô hình nhỏ cho W6. | ⏸ gác lại theo R-23 |
| N-09b | Cách lấy `phap_luat`; thứ tự ưu tiên trong nhóm "các dataset khác" (VISTA, giáo trình, wiki_vi, FineWeb-2); luật chọn bản khi trùng (R-15 a, b, c). | ❓ |
| N-12 | Kiểm lại số liệu W3 với bảng gốc (R-10). | ❓ |

### Chốt lượt 4 (2026-10-08)

Vẫn **chỉ chốt + bàn**, chưa viết code pipeline.

| Mã | Quyết định | Lý do / ghi chú | Trạng thái |
|---|---|---|---|
| R-19 | **Duyệt N-04 v2**: (1) chọn bản giữ trong cùng nguồn bằng **điểm rule** (dedup chạy trước khi có điểm mô hình nhỏ; điểm mô hình chỉ dùng ở L3); (2) MinHash theo **văn bản**, trùng bao hàm (chương nằm trong sách khác) do hash đoạn (L1) và L3 bắt, chưa thêm cơ chế thứ ba, đo ở S-8 rồi mới quyết; (3) L3 chỉ bật **theo từng nguồn**, ngưỡng cosine bắt đầu rất chặt, kiểu SemDeDup (mỗi cụm con giữ một đại diện), không gộp mọi cặp giống nhau thành khối; nguồn văn bản theo mẫu (pháp luật, bài tập, hỏi-đáp) tắt L3 cho tới khi xem ví dụ; (4) **họ trùng** (`dedup_family_id`, dùng chia train/test) chỉ nối từ các cặp chắc (fuzzy và semantic ngưỡng cao), report in kích thước họ lớn nhất; (5) chưa kiểm trùng với nguồn đã xử lý ngay khi chạy nguồn sau, vòng 2 của D-09 vẫn gỡ trùng toàn cục. Embedding chỉ tính cho đoạn đã sống sót (R-05) nên kho 256 chiều khoảng 5–8 GB, 1024 chiều khoảng 20–31 GB cho 10–15 triệu đoạn: số chiều chốt sau thí nghiệm S-6. | Ước lượng số đoạn: 10B token ÷ ~1.000 token mỗi đoạn, chưa đo. | ✅ (1)(2)(4)(5) đã code (cỡ họ lớn nhất in ở report.html từ 2026-10-09, C-06) / 🟡 (3) L3 chờ G-08 |
| R-20 | **Ngôn ngữ (N-01)**: chọn **fastText `lid.176`** (giữ D-02), **không chạy so sánh với GlotLID**. Nhãn ngôn ngữ do LLM trả kèm trong lượt chấm 50.000 mẫu (R-02) vẫn được lưu, nên nếu muốn kiểm fastText thì chỉ cần đếm tỉ lệ khớp, không tốn lượt chạy mới. | Chủ dự án: chọn một, khỏi chạy thử. Claude chọn lid.176 vì đã chốt ở D-02, nhẹ (~126 MB), đủ cho nhiệm vụ Việt / Anh / khác. **Chưa kiểm chứng số liệu benchmark của hai mô hình trong phiên này**; GlotLID (dùng cho FineWeb-2) mạnh hơn ở ngôn ngữ ít tài nguyên và văn bản nhiễu, nên nếu về sau thấy fastText nhầm nhiều trên web nhiễu thì đổi. | ✅ đã code (2026-10-08) |
| R-21 | **Mô hình đích (N-07)**: chưa chốt, **có vẻ là một mô hình ~27B**. Tạm giữ tokenizer Qwen3 (D-01) cho tới khi chốt tên mô hình; khi chốt phải đếm lại token bằng tokenizer của mô hình đó. | **Cảnh báo ước lượng của Claude (chưa kiểm)**: huấn luyện đầy đủ tham số 27B trên 10B token cần khoảng 6 × 27e9 × 1e10 ≈ 1,6e21 FLOPs ≈ vài nghìn giờ-GPU A100, vượt xa ngân sách ~30 giờ GPU (N-02b chưa rõ nghĩa); với LoRA thì nhẹ hơn nhưng vẫn lớn. W6 (mô hình nhỏ) và W8 (production) có thể là hai mô hình khác nhau, cần hỏi lại. | ❓ chờ chốt tên mô hình và cách tiêm (full / LoRA) |
| R-22 | **VJOL**: chủ dự án sẽ đặt dữ liệu ở `data/raw/VJOL/` và cho xem cấu trúc qua **MinIO MCP**. Chưa xem được: MinIO MCP lỗi `MINIO_USE_SSL is required` (cấu hình thiếu biến môi trường này; `~/.claude.json` mới có `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`), và `data/raw/VJOL` chưa tồn tại trong repo local. Chưa viết script VJOL (P-12) và chưa ước lượng thời gian. | Cần chủ dự án thêm `MINIO_USE_SSL` (`true` hoặc `false`) rồi nối lại MCP, hoặc dán cây thư mục. | ❓ chờ dữ liệu |
| R-23 | **Phạm vi hiện tại: chỉ xử lý dữ liệu cho SFT + CPT.** Việc chọn mô hình đích và cách huấn luyện **chưa cần bận tâm** (chủ dự án, 2026-10-08). Thay R-21: bỏ cảnh báo chi phí huấn luyện 27B khỏi danh sách cần bàn; tokenizer Qwen3 (D-01) vẫn là mặc định để đếm token. | | ✅ |
| R-24 | **VJOL nằm ở MinIO**, đường dẫn `vjol.info.vn/` (endpoint của MCP minio), chủ dự án sẽ chép về `data/raw/VJOL/`. MCP đã nối lại (có `MINIO_USE_SSL`) nhưng **khóa truy cập không đủ quyền**: `list_buckets`, `get_bucket_versioning`, `get_bucket_tags` báo `Access Denied`; `list_bucket_contents` trên `vjol.info.vn` trả một dòng rỗng (giống hệt khi gọi một bucket không tồn tại, tức lỗi bị nuốt). Chưa xem được cấu trúc, chưa ước lượng thời gian, chưa viết script. | Cần: khóa có quyền `s3:ListBucket` + `s3:GetObject` trên bucket chứa VJOL, hoặc tên bucket đúng nếu `vjol.info.vn/` là thư mục con của bucket khác. | ❓ chờ quyền |
| R-25 | **VJOL: đã xem được cấu trúc qua MinIO MCP** (bucket `datacrawl`, thư mục `vjol.info.vn/`): thư mục năm `1970/`…`2026/` chứa PDF phẳng tên `<số>_<số>.pdf` + `metadata/` (4 file jsonl: `vjol_issues` 4 MB, `vjol_listing` 233 MB, `vjol_merged` 469 MB, `vjol_pdf_download_log` 34 MB). Số đo được: `1970/` có 1.535 PDF (median 195 KB; nhiều khả năng là bài thiếu ngày, mốc 0 Unix); `2020/` có ≥5.001 PDF (MCP dừng đếm ở đó; median 673 KB, trung bình 1,1 MB, lớn nhất 22 MB). **Chưa đọc được nội dung** metadata hay PDF: `download_object` của MCP ghi vào trong container Docker (không gắn thư mục máy), sau đó container mất kết nối tới MinIO (`connection refused` tới IP nội bộ). Việc đo còn lại chuyển cho chủ dự án: `docs/SERVER_TASK.md` S-10. ViLA đã có kế hoạch VJOL chi tiết (`/home/dell/workspace/Repo/ViLA/PAPER_CRAWL_SUGGEST.md`: 359 tạp chí, OJS, triage PDF, parse rẻ trước đắt sau) và bộ phân loại PDF (`packages/parser/triage.py`, `types.classify_pdf`, `cmap_healer.py`) để tham khảo khi code. | Ước lượng thời gian và số lần parse: xem R-26 (đề xuất). | ✅ đã xem cấu trúc / ❓ chờ S-10 |
| R-26 | **Chiến lược VJOL (chủ dự án duyệt 2026-10-08): tối đa 2 lần đọc mỗi trang.** Lần 1 (CPU) thư viện chuẩn cho mọi trang; lần 2 (GPU OCR) chỉ cho từng **trang** hỏng (gần như không có chữ, hoặc chữ rác do font sai mà `cmap_healer` không sửa được). Đổi TCVN3 / VNI và `cmap_healer` chỉ sửa trên chữ đã đọc, không tính là một lần đọc. Chi tiết: (1) **một lượt đọc PyMuPDF cho mọi PDF**, phân loại trang ngay trong lượt đó (text / tcvn3 / scan / rỗng / font hỏng, dùng lại `extract_pdf` + `clean_pages` của giáo trình, thêm `cmap_healer` của ViLA nếu S-10 thấy font hỏng); (2) **chỉ trang hỏng mới OCR** (lượt thứ hai, GPU, chỉ phần trăm nhỏ số trang); không chạy GROBID / Marker / Docling ở v1 (đắt, ~1–3 trang/giây/GPU, chỉ cần khi muốn giữ bảng, công thức, cấu trúc); (3) bỏ phần **Tài liệu tham khảo** và trang bìa / mục lục bằng heuristic tiêu đề; (4) metadata `vjol_merged.jsonl` nối vào bản ghi qua tên file (tiêu đề, tạp chí, năm, ngôn ngữ, DOI) → lineage, domain theo tạp chí; bài vừa có bản Việt vừa có bản Anh giữ cả hai (P-08); (5) ưu tiên giữ bản VJOL khi trùng (R-15); quyền bypass (R-01). | **Ước lượng thô của Claude, chưa đo**: nếu ~100–170 nghìn PDF (đoán từ cỡ file log tải 34 MB) × ~10 trang ≈ 1–1,7 triệu trang, ~100–170 GB: chép 0,5–3 giờ; lượt đọc PyMuPDF dưới 1 giờ trên nhiều CPU; OCR phụ thuộc tỉ lệ trang scan (10% ≈ 150 nghìn trang, vài giờ trên 4 GPU nếu đạt ~2 trang/giây/GPU); pipeline phía sau (chunk, ngôn ngữ, chất lượng, dedup, nhúng ~1–1,5B token) vài giờ. Tổng cỡ **một ngày** chạy máy nếu trang scan ít; nếu dùng parser bố cục nặng cho mọi trang thì lên **2–3 ngày** GPU. Mọi số chốt lại sau S-10. | 🟡 đã chốt / ❓ số liệu chờ S-10 |

## Phiên 2026-10-08 (tiếp 2): rà bản đồ embedding + đối chiếu ViLA

Phiên này **chỉ ghi lại việc cần sửa, chưa sửa code pipeline** (phiên sau làm). Bằng chứng: `report.html` của run `run_10000_seed42` (10.000 điểm: 57 cụm, 3.065 điểm nhiễu = 30,7%, `cluster_id` tính trên UMAP 2D) và mã nguồn ViLA tại `/home/dell/workspace/Repo/ViLA` (`packages/reducer/stage.py`, `packages/embedder/*`, `packages/visualizer/scatter.py`).

### Việc cần sửa (chủ dự án yêu cầu ghi để phiên sau làm)

| Mã | Quyết định | Lý do / ghi chú | Trạng thái |
|---|---|---|---|
| R-27 | Thêm `numpy>=1.26` vào `dependencies` của `pyproject.toml`, cập nhật `uv.lock` (commit `1b9012e`, đã push). Trên server dùng `uv sync --locked --group viz` (bản đồ embedding cần plotly ở group `viz`). | Server báo `ModuleNotFoundError: numpy`: `dedup.py`, `embed.py`, `reduce.py` import numpy ngay đầu file, nhưng numpy chỉ vào gián tiếp qua group `ocr` / `curator` / `viz`. | ✅ |
| R-28 | **Gom cụm (HDBSCAN) chạy trên ma trận embedding gốc, không chạy trên toạ độ UMAP 2D**; UMAP 2D và PCA 2D chỉ để vẽ, cùng tô theo một nhãn cụm. **Sau khi làm D-11**: gom cụm trên bản cắt 256 chiều (Matryoshka) đã được lưu sẵn cho mục đích này; trước đó (e5 768 chiều) dùng thẳng vector gốc. Nếu HDBSCAN trên vector nhiều chiều quá chậm / quá nặng bộ nhớ thì thêm bước trung gian (PCA 50 chiều hoặc UMAP 10 chiều) rồi mới gom cụm; 2D vẫn chỉ để vẽ. Sửa `reduce.py:75` (hiện `cluster(umap_xy if umap_xy is not None else pca)`). | Cách ViLA làm: `packages/reducer/stage.py` gọi `_cluster(matrix, ...)` trên ma trận gốc, độc lập với mọi phép chiếu. UMAP làm méo mật độ nên HDBSCAN trên 2D có thể tạo hoặc xé cụm giả. **Giới hạn cần nhớ**: `sklearn.cluster.HDBSCAN` trên vector nhiều chiều không scale tới hàng triệu đoạn (R-19: 10–15 triệu đoạn) → khi chạy lớn phải fit trên mẫu phân tầng rồi gán nhãn cho phần còn lại, hoặc dùng cuML HDBSCAN trên GPU (ViLA ưu tiên cuML khi có). | ✅ đã code (bản cắt 256 chiều chờ D-11) |
| R-29 | `min_cluster_size` và `min_samples` cấu hình được (cờ CLI + `RunConfig`), ghi vào manifest cùng số chiều dùng để gom cụm, số cụm, tỉ lệ nhiễu. Mặc định theo ViLA: `max(2, min(20, n // 10))` (hiện cố định 25 ở `reduce.py:56`); `min_samples` chỉ truyền khi đặt rõ. | ViLA: `hdbscan_min_cluster_size` / `hdbscan_min_samples` trong cấu hình (một site ViLA đặt 50 và 10). Run hiện tại có 30,7% nhiễu; không thể so các lần chạy nếu tham số không được ghi lại. | ✅ đã code (2026-10-08) |
| R-30 | Đổi nhãn trên bản đồ: nút tô theo cụm ghi rõ "Cụm (HDBSCAN trên embedding gốc, tham số ...)"; ghi chú rằng `pca-cluster` là toạ độ PCA tô theo nhãn tính ở không gian gốc. Sửa `viz.py` (`COLOR_BY`, `ALGOS`) và `tests/test_pipeline.py::test_embed_reduce_viz_tfidf`. Chạy lại bằng `--force-from reduce`. | PCA 2D trông trộn lẫn là do PCA (phép chiếu tuyến tính lên 2 hướng phương sai lớn nhất của vector 768 chiều), không phải do nhãn UMAP. Chưa tính phương sai giải thích của 2 trục PCA trong run này. | ✅ đã code (2026-10-08) |
| R-31 | Cập nhật tài liệu sau khi sửa: `docs/PIPELINE.md` mục 5 (điểm điều chỉnh so với ViLA), `docs/diagrams/07_embed_reduce.svg`, đoạn "Repo tham khảo: ViLA" trong `CLAUDE.md` (hiện ghi "đã lấy theo ViLA" nhưng cluster đang lệch). | | ✅ đã sửa tài liệu (2026-10-08) |
| R-32 | Chưa duyệt, chờ chủ dự án chọn: (a) bảng đếm nguồn × cụm trong report; (b) từ khoá chủ đề gợi ý cho mỗi cụm lớn; (c) đổi tên hoặc ẩn nút `pca-cluster`. | Kết quả đọc run 10.000 điểm: cụm 15 (1.021 điểm) và cụm 3 (415) gần như toàn `sea_instruct_2602`; cụm 1, 48, 23, 20, 4 gần như toàn `stbook`; `sea_pile_v2` và `sea_lion_pile_v1` trộn đều trong các cụm web; chỉ 34% điểm nằm trong cụm có hơn 90% cùng một nguồn. 150 trong 221 mẫu bị loại `too_short` nằm trong cụm, 102 mẫu ở cụm 53 (toàn `sea_lion_pile_v1`). | ❓ |

### Kết quả đối chiếu code với ViLA (đề xuất, chưa phải quyết định)

Đối chiếu `vi_corpus/pipeline/{embed,reduce,viz,curator,runner,normalize,chunk}.py`, `vi_corpus/common/pdf_text.py` với `packages/{embedder,reducer,visualizer,pipeline,parser,extractor}` của ViLA. Chưa chạy thử code ViLA, chỉ đọc mã. "Yếu hơn" nghĩa là so với ViLA, theo mã đọc được, chưa đo.

| Mã | Loại | Chênh lệch | Đánh giá | Đề xuất | Trạng thái |
|---|---|---|---|---|---|
| V-01 | Khác (sai) | Cụm tính trên UMAP 2D thay vì ma trận gốc | **Yếu hơn** về độ đúng; nhanh hơn | Xem R-28 (đầu vào gom cụm ưu tiên là bản cắt 256 chiều của D-11) | ✅ (R-28) |
| V-02 | Code chưa theo quyết định | Code `embed.py` còn `multilingual-e5-base`, cắt 6.000 ký tự / 512 token, tiền tố `passage: `, trung bình token. Nhật ký **đã chốt khác**: D-10 (cắt đoạn ~1.024 token Qwen3, tối đa 2.048; bản đồ chỉ dùng 512 token đầu **có chủ đích**), D-11 (Qwen3-Embedding-0.6B, 1.024 chiều, đọc tới 32K token, lưu bản cắt 256 chiều), R-04 (bỏ `MAX_CHARS`, pooling token cuối, bỏ tiền tố). **Số đo 2026-10-08 (10.000 đoạn trích thật, tokenizer XLM-R của e5):** 97% mẫu stbook, 76% SEA-Instruct, 63% SEA-PILE v2 và 43% SEA-PILE v1 dài hơn 512 token; embedding e5 chỉ thấy trung bình 62–82% văn bản. | **Không cần cửa sổ + mean pool kiểu ViLA** nếu thực hiện R-04 / D-11 (cả đoạn 2.048 token nằm gọn trong ngữ cảnh 32K). Việc cần làm là đưa code về đúng quyết định. | Làm R-04 trước khi bật L3 semantic dedup | 🟡 (R-04) |
| V-03 | Thiếu | ViLA lưu `embedding_model_id`, `embedding_text_hash`, `embedding_chunks_used`; bỏ qua doc đã xong theo từng doc (`SkipExistingParquetFilter`). Ta: runner bỏ qua stage nếu file kết quả đã tồn tại, không so `embed_spec` / text (`runner.py`, nhánh `emb_path.exists()`). | **Yếu hơn, rủi ro im lặng.** Đổi `--embedder` hoặc sửa text mà không `--force-from embed` thì dùng lại embedding cũ, không báo. `manifest["config"]` chỉ được ghi ở `finalize`. | Ghi `spec`, `dim`, hash của danh sách `doc_id` + text vào manifest từng stage; so sánh khi bỏ qua; cảnh báo hoặc tự xoá nếu lệch | ✅ (G-02) |
| V-04 | Thiếu | Resume theo doc / shard (ViLA ghi một parquet mỗi doc). Ta ghi một file cho cả stage (`06_embeddings.parquet` ghi ở cuối) | **Yếu hơn ở quy mô lớn**: crash giữa chừng mất cả lần nhúng. Với 10.000 mẫu không thấy vấn đề. Đã có kế hoạch D-06 (shard) nhưng chưa code. | Làm D-06 trước khi chạy corpus thật | 🟡 (D-06) |
| V-05 | Khác | ViLA dùng reader/writer Curator, driver không giữ corpus. Ta bọc `RowsStage` với JSON `payload`, và `runner` giữ `rows: list[dict]` toàn bộ trong RAM; `read_embeddings` gọi `to_pylist()` rồi `np.array` trên cả cột | **Yếu hơn ở quy mô lớn** (10–15 triệu đoạn, R-19: hết RAM). **Mạnh hơn** ở an toàn kiểu: JSON giữ nguyên int / None, tránh pandas đổi int thành float (đã ghi ở CLAUDE.md). | Khi làm shard (D-06), đọc theo `iter_batches`, không gom hết vào list | 🟡 (D-06) |
| V-06 | Giống | Cách nhúng bằng HF: ViLA với `runtime=hf` cũng dùng stage tự viết (`NimEmbedderStage` + `HuggingFaceEmbedder`, trung bình hoá token + chuẩn hoá L2, tiền tố `passage: `), giống `HfEmbedder` của ta. `EmbeddingCreatorStage` của Curator (tokenizer và mô hình là hai stage tách) chỉ dùng khi `runtime=curator-hf`, không site ViLA nào dùng. | Không phải điểm yếu so với ViLA. Chỉ còn việc: throughput thật trên 4 A100 chưa đo (R-06); `HfEmbedder` không sắp văn bản theo độ dài nên padding có thể tốn (chưa đo). | Đo throughput trước, chỉ tối ưu nếu chậm | ❓ |
| V-07 | Thiếu | ViLA cấu hình reducer: `n_components` 2 hoặc 3, danh sách `methods` gồm t-SNE, tham số HDBSCAN. Ta: cố định 2D, PCA + UMAP, `min_cluster_size=25` | Thiếu tham số HDBSCAN là cái cần (R-29). 3D và t-SNE không cần cho mục tiêu hiện tại. | Chỉ làm R-29 | ✅ (R-29) |
| V-08 | Khác | Executor: ViLA đọc `cfg.executor.*` (mode, autoscale, `ignore_failures`, `ignore_head_node`) và `stage_overrides`. Ta hard-code giá trị trong `build_executor` | Nhẹ; chưa ảnh hưởng. Cần khi chạy nhiều node. | Hoãn tới lúc cần | ❓ |
| V-09 | Khác | Trực quan: ViLA tạo một HTML plotly cho mỗi cặp (tô theo, thuật toán), nạp plotly.js từ CDN. Ta nhúng một bản đồ vào `report.html`, `scattergl`, plotly.js nhúng một lần, chạy offline | **Mạnh hơn** cho dùng nội bộ / offline. **Nhược**: toàn bộ điểm và 240 ký tự văn bản nằm trong JSON của file (file báo cáo 9,6 MB cho 10.000 điểm) → khó mở khi trên khoảng 100.000 điểm; docs đã ghi chưa làm Datashader / lấy mẫu. | Khi vượt 100k điểm: vẽ mẫu phân tầng, hoặc bỏ phần văn bản khỏi JSON | ❓ |
| V-10 | Thiếu | Parser PDF: ViLA phân loại PDF thành 8 loại (`native_digital`, `cmap_repairable`, `font_corrupted`, `scanned_image`, `mixed_pages`, `office_document`, `encrypted`, `corrupted`) bằng triage không cần GPU, và `cmap_healer.py` sửa bảng ToUnicode hỏng (ViLA đo ~3–5% PDF luật bị lỗi "đấu" thành "đ u"). Ta (`pdf_text.py`): phân loại theo trang (`text` / `tcvn3` / `ocr` / `needs_ocr` / `empty`); không có `cmap_healer`, không phát hiện font hỏng, không tách PDF mã hoá / hỏng | **Thiếu**, quan trọng cho VJOL / VISTA (R-26 đã dự kiến thêm `cmap_healer` nếu S-10 thấy font hỏng). Ngược lại ta **mạnh hơn** ViLA ở TCVN3 (ViLA không có) và slide mất dấu cách. | Làm theo R-26 sau S-10; tham khảo `triage.py` và `cmap_healer.py` | 🟡 (R-26) |
| V-11 | Thiếu | `llm_ocr_fix.py` của ViLA: LLM sửa lỗi chính tả OCR, kèm guardrail (cùng số token, tên riêng và chữ hoa giữ nguyên, không đụng số, tối đa 5% ký tự, tối đa 30 loại sửa mỗi doc) | Ta không có; text OCR stbook là thô. Đụng nguyên tắc "không sửa nội dung" của `normalize.py` nên **cần chủ dự án quyết**. Guardrail đáng tham khảo nếu làm. | Chỉ ghi nhận, chưa làm | ❓ |
| V-12 | Khác | Chuẩn hoá tiếng Việt: ViLA đổi vị trí dấu thanh kiểu cũ sang mới (hoà → hòa), thống nhất biến thể chính tả (công ti → công ty; bộ luật lấy từ `undertheseanlp/text_normalization`, **GPL-3.0**), sửa khoảng trắng trong từ, ftfy xử lý mojibake qua Curator. Ta (`normalize.py`) chỉ NFC, bỏ ký tự vô hình, gộp khoảng trắng; **cố ý** giữ nguyên dấu thanh (docstring). | Không phải yếu hơn mà là chọn khác. Nhưng corpus trộn "hoà/hòa" làm tokenizer thấy hai dạng, và **mojibake chưa được xử lý** (chưa kiểm xem SEA mC4 có không). Không chép rules GPL-3.0 vào repo khi chưa xét giấy phép. | Hỏi chủ dự án: có chuẩn hoá dấu thanh không; thêm ftfy cho mojibake | ❓ |
| V-13 | Không có ở ViLA | Chấm điểm chất lượng, nhận diện ngôn ngữ, dedup do ta tự viết (đã ghi `docs/PIPELINE.md` mục 5). Ngôn ngữ đang là heuristic ký tự + stopword; đã chốt fastText `lid.176` (D-02, R-20) nhưng chưa code. Report 10.000 mẫu: band A 9.612 (96%), D 220; chỉ 1 mẫu bị loại vì trùng | Không so được với ViLA. **Nghi ngờ ngưỡng chất lượng lỏng** (96% band A) hoặc mẫu quá sạch: chưa kiểm chứng, cần xem tay mẫu band A. | Làm fastText (D-02) và hiệu chỉnh ngưỡng bằng mẫu có nhãn LLM (R-02) | 🟡 fastText đã code; ngưỡng chờ điểm LLM |

### Phiên sau làm theo thứ tự đề xuất

1. R-28, R-29, R-30, R-31 (một lần sửa nhỏ, `reduce.py` + `viz.py` + test + tài liệu).
2. V-03 (bỏ qua stage im lặng khi đổi cấu hình) — sửa nhỏ, rủi ro cao nếu không làm.
3. V-02 (cắt cửa sổ + mean pool) trước khi bật L3 semantic dedup.
4. V-04, V-05 cùng D-06 (shard) trước khi chạy corpus thật.

### Số đo và đề xuất chi tiết (2026-10-08, trang `.lavish/de-xuat-pipeline.html`)

Chủ dự án yêu cầu phân tích chi tiết và đề xuất đầy đủ hơn. Mục này ghi **số đo thật** và **tám gói việc** của trang đó. Chủ dự án mới yêu cầu "ghi lại các phần đã chốt"; chưa trả lời Q-A…Q-D trên trang, nên các gói mới và các câu hỏi dưới đây **chưa phải quyết định**. Ghi nhận: tokenizer Qwen3 (D-01), fastText (D-02), cắt đoạn theo token (D-10), Qwen3-Embedding (D-11) và shard (D-06) **đã chốt từ trước**; vấn đề là code chưa theo (V-02, bảng ở mục 0 của trang). Các script đo nằm ở thư mục tạm của phiên, **chưa đưa vào repo** (P5 bước 1 sẽ đưa phần dedup vào `tests/`).

#### Số đo (10.000 đoạn trích 238 ký tự đầu của `report (1).html`; tải tokenizer và `lid.176.bin` thật)

| Mã | Số đo | Kết luận / giới hạn |
|---|---|---|
| M-01 | Token / từ (từ = âm tiết): Qwen3 **1,366**, GPT-4o 1,373, Gemma-2 1,301, Llama-3 1,300, XLM-R (e5) 1,229; theo nguồn, stbook cao nhất (Qwen3 1,451). 6.618.201 từ giữ lại ≈ **9,05 triệu token Qwen3** (Llama-3 8,62; XLM-R 8,15). 300M token Qwen3 ≈ 219 triệu từ. | Báo cáo hiện ghi số từ là "token". KPI phụ thuộc tokenizer: Llama-3 thấp hơn Qwen3 4,8%, XLM-R thấp hơn 10%. Giữ Qwen3 đúng D-01. Giới hạn: chỉ 238 ký tự đầu, chưa đo toàn văn. |
| M-02 | Với e5 (512 token): vượt 512 token ở stbook 97%, SEA-Instruct 76%, SEA-PILE v2 63%, v1 43%; phần văn bản được nhìn thấy trung bình 62–82%. | Suy ra từ số từ × token / từ, không đếm lại từng văn bản. Giải quyết bằng D-10 / D-11 / R-04, **không** bằng cửa sổ + mean pool (sửa V-02). |
| M-03 | fastText `lid.176.bin` (131.266.198 byte) khớp heuristic **9.949 / 10.000 (99,5%)**. 24 đoạn stbook mở đầu bằng tiếng Anh (độ tin cậy 0,86–0,97) đều đang `kept`; 14 mẫu SEA-Instruct bị gọi `en` vì mở đầu `system: You are an AI assistant…`. | Lợi ích của fastText nằm ở nhận diện **theo đoạn** và ở việc **bỏ tiền tố vai / lượt system** của SEA-Instruct, không ở quyết định vi / không vi cho cả mẫu. 24 là cận dưới. |
| M-04 | `dedup.py` hiện tại (shingle 5 từ, 16×8, ≥0,8) trên 59 văn bản × 400 từ cấy biến thể: bắt 100% bản sao, sửa 1%; 76% sửa 2%; **0% sửa 5%**; **44% cắt 10% đầu + 10% cuối**; **0% khi văn bản nằm trong bản lớn hơn** (còn 50% / 25%). Cấu hình C (shingle 3, 32×4, ≥0,7): 86% sửa 5%, 100% cắt đầu cuối. 0 dương tính giả ở cả 5 cấu hình. | Giới hạn: văn bản là tài liệu kỹ thuật của repo, không phải corpus; "0 dương tính giả" trên 59 văn bản là bằng chứng yếu. Chứa nhau cần L1 (hash đoạn), không phải MinHash. |
| M-05 | Trong 220 mẫu `rejected:quality`, 202 (92%) là `too_short`, 13 `lang_not_allowed`. 453 / 9.779 mẫu `kept` có mã trừ điểm nhẹ (4,6%). Band A 96%. | Chưa biết ngưỡng lỏng hay dữ liệu vốn sạch. **Không chỉnh ngưỡng trên run 10.000**; chờ điểm LLM (R-11). |

#### Tám gói việc đề xuất (chưa duyệt từng gói)

| Mã | Gói | Căn cứ | Trạng thái |
|---|---|---|---|
| G-01 (P1) | Gom cụm trên không gian gốc (e5 768 chiều; sau G-08 dùng bản cắt 256 chiều); `min_cluster_size` / `min_samples` cấu hình được; chạy lưới {10, 20, 50, 100} × {embedding gốc, UMAP 10 chiều} trên run 10.000 trước khi chốt mặc định | R-28…R-31 | ✅ đã code / 🟡 lưới tham số chạy trên run 10.000 ở server |
| G-02 (P2) | Vân tay (`fingerprint`) cho mỗi stage: tham số + vân tay đầu vào; lệch thì chạy lại stage đó và các stage sau; `--keep-stale` để cố ý bỏ qua; run cũ không có vân tay thì cảnh báo một lần | V-03 (mới); nên làm trước G-03, G-04, G-08 | ✅ đã code (2026-10-08) |
| G-03 (P3) | `tokens.py` + `Qwen/Qwen3-0.6B` (`tokenizers`, `encode_batch`); `RunConfig.tokenizer` mặc định `hf:Qwen/Qwen3-0.6B`; cắt đoạn theo token (mục tiêu 1.024, tối đa 2.048, gộp đoạn cuối < 256); báo cáo ghi token / từ theo nguồn và chênh lệch giữa các tokenizer | D-01, D-10 | ✅ đã code (2026-10-08) |
| G-04 (P4) | fastText theo đoạn + `lang_mix`; SEA-Instruct bỏ lượt `system` và tiền tố vai; mã `mixed_language` | D-02, R-20 | ✅ đã code (2026-10-08) |
| G-05 (P5) | Bước 1: `tests/test_dedup_recall.py` cấy bản trùng. Bước 2: đọc 50 cặp ở vùng Jaccard [0,7; 0,8) trên mẫu 100k ở server rồi mới đổi tham số. Bước 3: L1 hash đoạn / dòng cho trường hợp văn bản nằm trong bản lớn | D-03, R-19; bước 2 chờ kết quả S-11 (R-34) | ✅ bước 1, 3 đã code / ❓ bước 2 (S-11) |
| G-06 (P6) | `scripts/make_judge_sample.py`: 50.000 mẫu, ~70% qua rule, ~30% bị rule loại, phân tầng nguồn × độ dài | R-11, R-12 | ✅ đã code (2026-10-08) |
| G-07 (P7) | Shard + luồng dữ liệu, Arrow thay JSON `payload`, `_done` theo shard; **bước 0 đo trước** (docs/s, MB/s, RSS theo stage trên 100k bản ghi) | D-06, V-04, V-05 | 🟡 chờ số đo |
| G-08 (P8) | Qwen3-Embedding-0.6B: bỏ `MAX_CHARS` và `passage: `, cắt theo token tới 2.048, sắp lô theo độ dài, lưu 1.024 và bản cắt 256 chiều. Đã kiểm `config.json` (28 lớp, 1024, 32768, bf16); **chưa kiểm** pooling token cuối, padding trái, Matryoshka (đọc model card trước khi code) | D-11, R-04 | 🟡 |

Thứ tự đề xuất: G-02 → (G-03, G-04) → (G-08, G-06, G-05 bước 2) → G-07, với G-01 và G-05 bước 1 làm ngay không cần server. Chi tiết, sơ đồ phụ thuộc và rủi ro: trang Lavish.

#### Câu hỏi chờ chủ dự án (trên trang Lavish)

| Mã | Câu hỏi | Đề xuất của Claude | Trạng thái |
|---|---|---|---|
| N-13 (Q-A) | Đoạn stbook song ngữ (≥24 / 2.500 đoạn trích mở đầu bằng tiếng Anh, đang được giữ): gắn nhãn `lang_mix` và giữ / bỏ đoạn không phải tiếng Việt / loại cả đoạn nếu vượt ngưỡng | Gắn nhãn và giữ (hợp P-08), quyết lại sau khi có phân phối `lang_mix` của cả stbook | ✅ chốt A (R-33), đã code |
| N-14 (Q-B) | Tham số fuzzy dedup: giữ A (shingle 5, 16×8, ≥0,8) / chuyển sang C (shingle 3, 32×4, ≥0,7) sau khi đọc 50 cặp thật / chuyển sang E (shingle 2) | C sau khi kiểm dương tính giả trên dữ liệu thật | 🟡 đo trước trên server (R-34, S-11) |
| N-15 (Q-C) | Tokenizer chính thức đếm KPI ≥300M: Qwen3 / Llama-3 / khác | Qwen3 (đúng D-01) | ✅ Qwen3 (R-35) |
| N-16 (Q-D) | Gói việc cho phiên code tiếp theo | G-01, G-02, G-03, G-04, G-05 bước 1 (không cần server) | ✅ duyệt (R-36), đã code |

### Chốt lượt 5 (2026-10-08): trả lời N-13…N-16

Vẫn **chỉ ghi**, chưa viết code pipeline; code làm ở phiên sau theo R-36.

| Mã | Quyết định | Lý do / ghi chú | Trạng thái |
|---|---|---|---|
| R-33 | **N-13 = A**: đoạn song ngữ (stbook) chỉ **gắn nhãn `lang_mix` và giữ nguyên**; chưa bỏ đoạn tiếng Anh, chưa loại đoạn. Quyết lại sau khi có phân phối `lang_mix` của cả stbook (ngưỡng loại, nếu có, sẽ chọn lúc đó). | Hợp P-08 (giữ sách tiếng Anh); ít nhất 24 / 2.500 đoạn trích stbook mở đầu bằng tiếng Anh (M-03), chưa biết tỉ lệ trên toàn sách. Thực hiện trong G-04. | ✅ đã code (2026-10-08) |
| R-34 | **N-14: chưa đổi tham số fuzzy dedup; đo trên server trước**, việc đo ghi ở `docs/SERVER_TASK.md` mục **S-11**. Cấu hình A (shingle 5 từ, 16×8, ≥0,8) tiếp tục là mặc định cho tới khi có kết quả. Đề xuất tiêu chí của Claude (≥ 90% cặp trong nhóm Jaccard [0,7; 0,8) là trùng thật thì chuyển sang C) **chưa được duyệt**. | Thí nghiệm 2026-10-08 chỉ có 59 văn bản kỹ thuật, 0 dương tính giả không đủ tin (M-04). Xoá nhầm dữ liệu tốt đắt hơn bỏ sót một bản trùng. | 🟡 chờ S-11 |
| R-35 | **N-15 = Qwen3**: `Qwen/Qwen3-0.6B` là tokenizer chính thức để đếm KPI ≥300M token và cắt đoạn (D-01, D-10, R-13 giữ nguyên). Khi chốt mô hình đích khác họ Qwen thì đếm lại (Llama-3 thấp hơn khoảng 4,8%, XLM-R khoảng 10%, theo M-01). | Cùng tokenizer với Qwen3-Embedding (D-11); đo thực tế 1,366 token / từ. | ✅ (code: G-03) |
| R-36 | **N-16 = duyệt**: phiên code tiếp theo làm **G-01, G-02, G-03, G-04 và G-05 bước 1**. G-02 (vân tay stage) làm trước G-03 và G-04. G-05 bước 2–3, G-06, G-07, G-08 chưa làm ở phiên đó (cần server hoặc số đo). Sau khi G-03 / G-04 xong, run 10.000 mẫu hiện tại phải chạy lại hoàn toàn (đổi tokenizer, cỡ đoạn, ngôn ngữ làm chỉ số cũ không so được). | Đề xuất của Claude được chấp nhận nguyên văn (trả lời "ok"). | ✅ đã code (2026-10-08) |


## Phiên 2026-10-08 (code): code các quyết định không cần chạy server

Yêu cầu của chủ dự án: "code tất cả những gì đã chốt trong DECISION_LOG.md … chỉ code các phần không cần điều kiện tiên quyết là phải chạy thử trên server", xong thì gọi reviewer. Chi tiết thay đổi theo stage: `docs/PIPELINE.md` mục 6; lệnh mới: `docs/README.md` Phần D.

| Mã | Quyết định / ghi chú | Lý do | Trạng thái |
|---|---|---|---|
| C-01 | **Đã code**: G-02 (vân tay stage, `--keep-stale`), G-03 / D-01 / R-35 (`tokens.py`, Qwen3 mặc định), D-10 (cắt đoạn theo token, `cpt_blocks.parquet`), D-04 (`lines.py`), G-04 / D-02 / R-20 / R-33 (fastText theo đoạn, `lang_mix`, mã `low_lang_score` / `mixed_language` / `low_diacritic`), D-03 L0 + L1 + L2 theo văn bản, Q4 / R-15 / R-19 (nhóm cpt / sft, ưu tiên nguồn), D-09 (kho dấu vân tay + `scripts/dedup_global.py` + `--global-verdict`), R-01 (rights gate chỉ gắn nhãn), G-01 / R-28…R-31 (HDBSCAN trên không gian gốc, tham số cấu hình được), R-09 (đếm candidate / instruction), R-16 (quét 13-gram, danh sách rỗng ở `configs/benchmarks/`), D-05 / R-14 (`vi_corpus/common/parsers/`, `--retry-skipped`), G-05 bước 1 + 3 (`tests/test_dedup_recall.py`, L1 bao hàm), G-06 / R-11 (`scripts/make_judge_sample.py`). Thư viện thêm bằng `uv add`: `tokenizers`, `fasttext-numpy2-wheel`, `python-docx`, `trafilatura`, `ebooklib`. | Theo các quyết định đã chốt ở trên. | ✅ |
| C-02 | **Không code ở phiên này (cần server / số đo trước)**: G-05 bước 2 (S-11, R-34), G-07 / D-06 (shard, Arrow; "đo trước"), G-08 / D-11 / R-04 (Qwen3-Embedding: đọc model card, S-6), L3 semantic dedup (R-03, sau G-08), script LLM chấm điểm (thang Q6 chưa duyệt), VJOL (P-12, R-26, chờ S-10), lưới tham số HDBSCAN của G-01 (chạy trên run 10.000 ở server). | Đúng giới hạn chủ dự án đặt. | 🟡 |
| C-03 | Cách hiểu của Claude, **chưa được duyệt**: (a) L1 (D-03) làm thành phát hiện **bao hàm**: văn bản có ≥ 80% đoạn văn dài (≥ 20 từ) nằm trong một văn bản lớn hơn thì bị loại `contained`, giữ bản lớn bất kể ưu tiên nguồn; chỉ trong phạm vi một lần chạy (kho vòng 2 chưa có hash đoạn văn). (b) R-15 (a) còn mở nên SEA-PILE v2, v1 và SEA-Instruct cùng mức ưu tiên 1; giáo trình và nguồn khác mức 3. (c) `min_lang_score` 0,65 (giáo trình 0,5) làm lỗi mềm −10, `low_diacritic` −15 khi tỉ lệ chữ có dấu < 5% (văn bản tiếng Việt ≥ 20 từ). (d) Mẫu chấm LLM bỏ trùng chính xác bằng sha256 text trong chính bộ mẫu (đọc `04_quality.parquet`, trước dedup như R-11). (e) Chunk-level exact dedup chạy sau dedup theo văn bản. | Các điểm này không có số cụ thể trong quyết định gốc; đều nằm trong `RunConfig` / hằng số nên đổi được. | ❓ chờ chủ dự án xem |
| C-04 | Số đo recall MinHash cấu hình A bằng bản cấy (40 văn bản âm tiết ngẫu nhiên × 7 biến thể, seed 2026): chép nguyên 1,0; sửa 1% 1,0; sửa 2% 0,875; sửa 5% 0,0; cắt 10% hai đầu 0,5; nằm trong bản lớn gấp 2 / 4 lần 0,0 (do L1 bắt); 0 dương tính giả giữa các bản gốc. | Khớp M-04; mức sàn ghi trong test để phát hiện khi đổi cấu hình. | ✅ đo |
| C-06 | **Sửa theo reviewer (2026-10-09)**: (1) kho dấu vân tay ghi cả văn bản bị loại (bia mộ `alive=False`) + `written_at` + `rights_status`; vòng 2 lấy dòng mới nhất theo `written_at`, không theo tên file; mỗi lần chạy ghi đè file của mọi nguồn trong mix và xoá file cùng tên của nguồn đã bỏ (trước đây văn bản bị loại ở lần chạy mới vẫn sống trong kho cũ và có thể làm loại oan bản khác). (2) Vòng 2 tính lại ưu tiên từ `source_key` (`--priority`) và có `--rights-gate enforce`, nên đúng câu "đổi ưu tiên / bật rights gate chỉ cần chạy lại vòng 2". (3) MinHash băm cả văn bản (trước: 20.000 từ đầu, hai sách chung phần đầu bị coi là trùng). (4) Vân tay stage chỉ băm trường hồ sơ nguồn stage đọc. (5) Report in cỡ họ trùng lớn nhất (R-19 mục 4). (6) Test thêm: `--retry-skipped`, pptx / epub / `.doc` thiếu LibreOffice, `--global-verdict` qua runner, `--compare-tokenizers`. Phụ: `make_judge_sample.py` lấy `keep_bands` từ manifest (`--keep-bands`); parser HTML / EPUB để BeautifulSoup / trafilatura tự dò bảng mã, giữ dòng trống giữa đoạn. | Reviewer trả BLOCKED với các điểm này. | ✅ đã code |
| C-07 | **Chủ dự án chốt (2026-10-09): miễn xoá dòng cho văn bản có code hoặc bảng.** Luật A của `lines.py` xoá dòng ngắn lặp ≥ 3 lần và dòng chỉ là số, nên `}` trong code hay ô số trong bảng bị xoá. Cách làm (Claude tự chọn, chưa duyệt): `has_code_or_table` coi văn bản có code khi có **khối** ≥ 3 dòng code **liền nhau** (dòng trống không ngắt), trong đó ít nhất một dòng "chắc" (hàng rào ```, dòng chỉ có ngoặc, dòng kết thúc `{`, câu lệnh import, dòng chỉ là lời gọi hàm `f(x)`, câu lệnh gọi hàm kết thúc `;`, dòng bắt đầu bằng từ khoá chữ thường và kết thúc `:` `;` `{`, phép gán `=` / `:=` kết thúc `;`, dòng SQL `SELECT` / `FROM` / `WHERE`...), các dòng còn lại có thể là dòng "yếu" (phép gán không có `;`, `return x`, `else`, `pass`, `begin`, `end;`...; chỉ toàn dòng yếu như công thức toán thì không tính); có bảng khi có dòng kẻ markdown `\|---\|`, hoặc ≥ 3 dòng liền nhau có cùng số dấu `\|` (≥ 2), hoặc một dải ≥ 6 dòng ô ngắn (< 4 từ) liền nhau mà ít nhất nửa có chữ số. Dòng menu / breadcrumb `Trang chủ \| Tin tức` lẻ và văn xuôi tiếng Anh trích PDF xuống dòng giữa câu ("for ... (see Section 3.2)") không bị coi là code / bảng (có test). Văn bản như vậy bỏ qua mục A, chỉ còn bỏ dòng nhãn số trang chắc chắn ("Trang 12", "- 12 -", không bỏ số trơn); mục C (web) vẫn chạy nhưng không xoá dòng trông như code hay dòng không có chữ cái trong văn bản đó. Report có cột "Văn bản miễn A". `LINES_VERSION` 3 nên vân tay tự chạy lại từ prepare. | Reviewer nêu; chủ dự án trả lời "có miễn có văn bản có code hoặc bảng nhé". Tỉ lệ văn bản được miễn trên dữ liệu thật chưa đo (máy local chỉ có dữ liệu giả): xem cột report ở run server. | ✅ đã code / 🟡 đo tỉ lệ miễn ở server |
| C-08 | **Sửa theo reviewer lần 2 (2026-10-09)**: (1) kho dấu vân tay tìm file của lần chạy bằng đường dẫn chính xác, không glob (tên `--run-name` có `[`, `*`, `?` từng khớp và xoá nhầm file của lần chạy khác). (2) Kho gắn với vân tay của `05_dedup.parquet` (`manifest["dedup_index"]`): dedup chạy lại với `--no-dedup-index` thì xoá kho của lần chạy; `05_dedup.parquet` bị xoá (đổi tham số, `--force-from`, kể cả khi dừng bằng `--until quality`) thì xoá kho; kho lệch vân tay thì ghi lại từ `05_dedup.parquet`. (3) Văn bản mà lần chạy mới nhất đã loại ra status `dropped` trong verdict; áp verdict vào lần chạy cũ hơn còn giữ văn bản đó thì đoạn giữ nguyên, đếm `superseded` và cảnh báo (chỉ nên finalize lần chạy mới nhất của mỗi nguồn). (4) `--priority` sai dạng báo lỗi argparse. | Reviewer trả BLOCKED lần 2 với (1), (2); (3), (4) là ghi chú không chặn. | ✅ đã code |
| C-09 | **Chủ dự án chốt (2026-10-09): giữ thụt lề cho văn bản có code. Đã code (2026-10-09).** Trước đây `normalize.py` (chạy trước xoá dòng) gộp khoảng trắng và bỏ khoảng trắng đầu dòng, nên code Python trong giáo trình / web mất thụt lề (sai nghĩa với Python). Cách làm: `normalize_text` bỏ ký tự vô hình, đổi khoảng trắng Unicode về dấu cách, rồi gọi `lines.has_code` (chỉ phần nhận diện **khối code** của C-07, không tính bảng) trên text **còn thụt lề**; văn bản có code giữ nguyên khoảng trắng đầu dòng (dấu cách / tab như gốc), chỉ gộp khoảng trắng giữa dòng và bỏ khoảng trắng cuối dòng; văn bản khác như cũ. Mục C của `lines.py` và `chunk_text` không còn `strip()` làm mất thụt lề dòng đầu đoạn văn. Manifest prepare thêm `indent_kept_docs`. `NORMALIZER_VERSION` 2, `LINES_VERSION` 4, `CHUNK_VERSION` 3 nên vân tay prepare đổi, run cũ tự chạy lại từ prepare. Giới hạn: code trích PDF / OCR đã mất thụt lề từ trước thì không khôi phục được. | Phát hiện khi sửa C-07; chủ dự án trả lời "tôi đồng ý giữ thụt lề, lưu vào DECISION_LOG lượt sau làm". Tỉ lệ văn bản giữ thụt lề trên dữ liệu thật chưa đo. | ✅ đã code / 🟡 đo `indent_kept_docs` ở server |
| C-05 | **Chưa kiểm**: tokenizer Qwen3 thật (tải từ HF bị hết giờ trên máy local, test dùng tokenizer `words` / WordLevel tự dựng), fastText thật chỉ test khi có `lid.176.bin`, parser doc / ppt / djvu chưa thử với LibreOffice / djvulibre thật, mọi đường chạy GPU. Run 10.000 mẫu cũ phải chạy lại hoàn toàn (vân tay sẽ tự xoá các stage lệch). | Môi trường local không có mạng ổn định / GPU. | 🟡 chờ server |
