# Ứng viên nguồn dữ liệu mới (ưu tiên STEM, kiến thức dày)

> Trạng thái: **khảo sát xong; chủ dự án đã trả lời 5 câu (NS-06..NS-10), chốt sau khi xem kết quả kiểm chất lượng ở mục 6**. Chưa ghi vào `docs/SOURCES.md`.
> Khảo sát ngày 2026-11-10 bằng 6 agent song song (web corpus đã lọc, toán, SGK/OER, học thuật, dữ liệu dịch/tổng hợp, bách khoa/tra cứu).
> Chỉ dùng request nhẹ (API Hugging Face, robots.txt, OAI-PMH Identify/đếm, vài trang). Không tải file dữ liệu.
> Cột "Kiểm chứng": **đo** = đọc được số từ API/README; **một phần** = thấy nguồn tồn tại, số liệu từ tin tức/tìm kiếm; **chưa** = chưa kiểm chứng.
> Không xét lại các nguồn đã có trong `docs/SOURCES.md` và `docs/LUAN_AN.md`.

## 1. Kết luận ngắn

- Nguồn **mới đáng giá nhất**: **FinePDFs tiếng Việt** (HF `HuggingFaceFW/finepdfs`, config `vie_Latn`): 1.229.330 văn bản, **7,72 tỷ token** (README), 13,0 GB parquet (đo), ODC-By. Tập con lọc giáo dục **FinePDFs-Edu** (`HuggingFaceFW/finepdfs-edu`, `vie_Latn`): 2,99 GB parquet (đo), số token chưa có trong README (ước tính 1,5–2 tỷ theo tỉ lệ dung lượng). Cùng miền PDF với VJOL/VISTA/giáo trình/luận án → phải dedup chéo (D-09).
- Kho giáo dục mở tiếng Việt (SGK, OER, toán bản địa, bách khoa) **nhỏ**: cộng lại cỡ vài chục đến vài trăm triệu token (ước tính), phần lớn có bản quyền hoặc robots.txt cấm AI. Là lớp bổ sung chất lượng cao, không phải nguồn khối lượng.
- Không có bản tiếng Việt sẵn của FineWeb-Edu, OpenWebMath, FineMath, NuminaMath, OpenR1-Math, s1K. Muốn có STEM khối lượng lớn bằng tiếng Việt thì phải **tự dịch** (OpenWebMath 14,7 tỷ token, FineMath, cả hai ODC-By).
- Hầu hết tạp chí đại học/viện (Cần Thơ, Huế, VAST, ĐHQG TP.HCM, Y học VN, TNU) **đã nằm trên VJOL**. Giá trị riêng: lấp ~1/3 bài VJOL không có PDF qua OAI-PMH.

## 2. Bảng xếp hạng

### Nhóm A: mở, lớn hoặc rẻ, nên làm trước

| key đề xuất | Nguồn | Quy mô | Giấy phép | STEM | Công sức | Rủi ro / ghi chú | Kiểm chứng |
|---|---|---|---|---|---|---|---|
| `finepdfs_edu_vi` | HF `HuggingFaceFW/finepdfs-edu` (`vie_Latn`) | 3 file, 2,99 GB | ODC-By | 5 | thấp (parquet, không gated) | Lọc bằng classifier giáo dục (nhãn Qwen3-235B, mỗi ngôn ngữ một classifier). Trùng PDF với VJOL/VISTA/luận án/giáo trình | đo |
| `finepdfs_vi` | HF `HuggingFaceFW/finepdfs` (`vie_Latn`) | 1,23 triệu văn bản, 7,72 tỷ token | ODC-By | 4 | thấp | Tập cha của dòng trên; lấy cả hai thì dedup theo id | đo |
| (nhãn lọc) | HF `epfml/FineWeb2-HQ` (`vie_Latn`) | 4,0 triệu văn bản, 63 GB | ODC-By | 3–4 | thấp | Là top 10% của FineWeb-2 (đã chọn) theo classifier "giàu kiến thức". Chỉ cần lấy danh sách id làm cờ chất lượng / trọng số, không cần text mới | đo |
| `wikimedia_vi_misc` | dump Wikibooks (8,7 MB bz2), Wiktionary (63 MB), Wikivoyage (4,6 MB) tiếng Việt | nhỏ | CC BY-SA 4.0 | 2–3 | rất thấp, dùng chung parser với `wikisource_vi` | Wiktionary phần lớn là định nghĩa ngắn, nhiều từ tiếng Anh. Wikiquote, Wikinews không đáng | đo |
| `hust_dspace` | dlib.hust.edu.vn (DSpace 7.4, như kho ĐHQGHN) | 33.202 mục, 31.335 có file (luận văn 13.435, đồ án 2.898, khóa luận 1.822, bài báo ~2.100) | chưa rõ → quarantine | 4–5 (kỹ thuật) | trung bình, dùng lại kế hoạch kho ĐHQGHN (`docs/LUAN_AN.md` mục 2) | REST tìm kiếm + OAI-PMH ẩn danh; `core/items` trả 401, tỉ lệ file tải ẩn danh chưa đo. Có thể trùng luận án NLV | đo |

### Nhóm B: dùng được nhưng có điều kiện

| key đề xuất | Nguồn | Quy mô | Giấy phép | STEM | Điều kiện / ghi chú | Kiểm chứng |
|---|---|---|---|---|---|---|
| `cosmopedia_vi` | HF `thangvip/cosmopedia_vi_math` (320k), `_khanacademy` (24k), `_openstax` (1k), cột `vi_text` | ~345k văn bản | không ghi (Cosmopedia gốc Apache-2.0, do Mixtral sinh) | 4 | Văn bản tổng hợp dịch máy, không rõ công cụ dịch. Kiểm mẫu chất lượng dịch và LaTeX trước; gắn nhãn `synthetic` | đo |
| `vi_pubmed` | HF `VietAI/vi_pubmed` | 20,1 triệu abstract PubMed dịch máy (cột `en` + `vi`), 44 GB | thẻ "cc", gốc theo điều khoản NLM → chưa rõ | 3 (y sinh) | Khối lượng lớn (ước tính vài tỷ token) nhưng dịch máy, câu ngắn. Nếu dùng thì đặt trọng số thấp, gắn nhãn `machine_translated` | đo (cỡ), chưa (license) |
| `hplt3_vi` | HPLT 3.0 `vie_Latn` (data.hplt-project.org, jsonl.zst) | chưa đo | CC0 | 3 | Web khác crawl (có Internet Archive), có điểm chất lượng WDS và nhãn register; chỉ lấy bin cao + register "informational/how-to". Trùng một phần với SEA-PILE/FineWeb-2 | một phần |
| `byt_phac_do` | Hướng dẫn chẩn đoán điều trị Bộ Y tế, kcb.vn/phac-do | vài trăm PDF (ước tính) | ban hành kèm Quyết định → có thể thuộc Điều 15 Luật SHTT (phần phụ lục chuyên môn là vùng xám) | 4 (y khoa) | Nhỏ nhưng dày kiến thức; dùng bộ trích PDF sẵn có | một phần |
| (lấp VJOL) | OAI-PMH của tạp chí OJS: Y học VN (15.870 bản ghi, `Crawl-delay: 20`), ĐH Cần Thơ (5.274), TNU (6.762), Y học cộng đồng (5.398), js.vnu.edu.vn các series (MPS 604, tiếng Anh) | vài chục nghìn bài | chỉ ghi "Copyright", không CC → quarantine như VJOL | 4 | Chỉ để lấy bài VJOL thiếu PDF, không phải nguồn riêng. Mức trùng cấp bài chưa đo | đo (số bản ghi) |
| (SFT toán) | `5CD-AI/Vietnamese-395k-meta-math-MetaMathQA-gg-translated`, `5CD-AI/...orca-math...200k...`, `5CD-AI/...OpenMathInstruct-1-50k...`, `VoVanPhuc/Vi_MathInstruct` (gated), `vntc2/Vi_Reasoning_ColdStartSFT` (Apache-2.0, 20k), `hllj/vi_math_problem_crawl` | vài trăm nghìn cặp QA | phần lớn không ghi; gốc sinh bởi GPT-3.5/4 (điều khoản OpenAI) | 3 | Cho giai đoạn SFT, không phải CPT. Google Translate làm hỏng LaTeX (`\times` → "\ lần") → cần bước sửa công thức. Phải chạy contamination (MetaMath sinh từ GSM8K/MATH train) | đo |
| (tin tức, tuỳ chọn) | HF `bkai-foundation-models/BKAINewsCorpus` (16,8 triệu bài, 56 GB), `tmnam20/Vietnamese-News` (2,4 triệu bài) | lớn | không ghi | 2 | Ít STEM; chỉ nếu cần thêm văn bản chung chất lượng khá | đo |

### Nhóm C: chất lượng cao nhưng phải xin phép trước

| Nguồn | Quy mô | Vướng mắc | Đề xuất |
|---|---|---|---|
| VOER, voer.edu.vn | 22.343 tài liệu (KH&CN 9.612, Toán-Thống kê 582) | Ghi CC BY 3.0 nhưng robots.txt có `Content-Signal: ai-train=no` và chặn bot AI | Gửi email VNF/VOER xin xác nhận bằng văn bản |
| SGK NXB Giáo dục, taphuan.nxbgd.vn | SGK lớp 1–12 | Bản quyền NXBGDVN, NXB tuyên bố cấm đăng tải/tải bản điện tử khi chưa phép; hanhtrangso đã ngừng | Xin phép; dạng flipbook ảnh → OCR |
| Tia Sáng, tiasang.com.vn | chưa đếm | "Toàn bộ bản quyền thuộc Tia Sáng" (nay thuộc VnExpress); robots cho phép | Xin phép (tiasang@vnexpress.net) |
| Bách khoa toàn thư / Từ điển Bách khoa VN (bachkhoatoanthu.gov.vn, .vass.gov.vn) | ~40k mục từ | Bản quyền Nhà nước/NXB, yêu cầu văn bản cho phép; site không truy cập được từ máy này | Xin phép Viện Hàn lâm KHXH VN |
| diendantoanhoc.org | diễn đàn lớn, dùng LaTeX | robots.txt và trang chủ trả 403 (chặn bot), license nội dung người dùng không rõ | Nguồn toán bản địa tốt nhất; chỉ làm khi chủ dự án đồng ý và có phép |
| Bài giảng toán có mã LaTeX (vd. Huỳnh Quang Vũ, sites.google.com/view/hqvu/teaching) | vài giáo trình | Không ghi license | Xin tác giả; nhỏ nhưng chất lượng rất cao |
| VAST journals, vjs.ac.vn | chưa đếm | robots.txt `ai-train=no`; phần lớn đã có trên VJOL | Không crawl riêng |
| bktt.vn (bách khoa cộng đồng) | rất nhỏ (~194 URL trong sitemap) | License "tương thích Wikipedia" chưa xác nhận | Kiểm license rồi lấy qua MediaWiki API |

### Nhóm D: tự tạo dữ liệu (dự án riêng, tốn GPU)

| Việc | Đầu vào | Ghi chú |
|---|---|---|
| Dịch STEM sang tiếng Việt, giữ nguyên LaTeX | OpenWebMath (14,7 tỷ token, ODC-By), FineMath-4plus (6,7 triệu dòng, ODC-By), FineWeb-Edu | Cách duy nhất để có STEM tiếng Việt cỡ tỷ token. Tránh MathPile (CC BY-NC-SA). Cần quyết model dịch, ngân sách GPU, cách đánh giá chất lượng dịch |

## 3. Benchmark phải loại khỏi train (đưa vào contamination scan)

`vi_corpus/pipeline/contamination.py` hiện để trống danh sách benchmark (R-16). Các bộ dưới đây là đề thi/benchmark; **không đưa vào train**, nên thả vào `configs/benchmarks/` trước khi dựng mix:

VMLU (`ZaloAI/VMLU`, 10.880 câu, 58 môn), ViExam, VNHSGE (THPT QG), `Intelligent-Internet/Vietnamese-Entrance-Exam`, `ura-hcmut/Vietnamese-MATH` (5.000 câu test), `stair-lab/MATH_vi`, `hllj/vi_gsm8k` / `namfam/gsm8k-vietnamese`, `nhn309261/MathVerse-Vietnamese`, ViMath-Bench, `tmnam20/ViMedAQA`, `tarudesu/ViHealthQA`, `vilm/vlsp_arc_vi`, `vilm/vlsp_hellaswag_vi`, `vilm/vlsp_truthfulqa_vi`, `danganhdat/university-stem-quiz-en-vi-translated`.

## 4. Đã xem và không đề xuất

| Nguồn | Lý do |
|---|---|
| CulturaX vi (`uonlp/CulturaX`), MADLAD-400 vi, `VTSNLP/vietnamese_curated_dataset` | Gần như trùng hết SEA-PILE v1 (mC4) + FineWeb-2 |
| FineWiki viwiki (`HuggingFaceFW/finewiki`) | Trùng Wikipedia vi đã chọn (chỉ đáng so cách parse) |
| HPLT 2.0 cleaned vi | Web đã làm sạch nhưng không lọc giáo dục; HPLT 3.0 tốt hơn |
| Khan Academy tiếng Việt | CC BY-NC-SA, chủ yếu video, chặn bot |
| `5CD-AI/Viet-Doc-VQA*`, `5CD-AI/...tiny-webtext...`, `5CD-AI/...kaist-CoT...unrefined` | Văn bản do Gemini/GPT sinh, nhiễu, ít STEM |
| `LuminarAI/vietnamese-ms-hs-textbook-math-300K` | Nguồn gốc không rõ, mẫu có trường hỏng |
| toanmath.com, loigiaihay.com, vietjack.com, olm.vn | Bản quyền thương mại; toanmath chặn thư mục PDF/Word trong robots.txt |
| PhoMT, MTet | Câu ngắn, miền TED/phụ đề/tin tức, license chỉ cho nghiên cứu |
| Project Gutenberg vi, data.gov.vn, Tạp chí Cộng sản | Không có sách tiếng Việt / chủ yếu bảng số / bản quyền tạp chí, ít STEM |
| Sáng chế (IP Vietnam), CORE, BASE | Chưa xác nhận có toàn văn tải được; để sau |
| `vilm/tinystories-envi`, `bkai-foundation-models/vi-alpaca` | Không có giá trị STEM |
| Các repo không tồn tại / private: `vilm/CulturaX-vi`, `5CD-AI/Viet-fineweb-edu`, `HuggingFaceFW/fineweb-2-edu`, `binhvq/news-corpus`, `vinai/PhoGPT-pretrain` | API trả lỗi, không có |

## 5. Việc cần chủ dự án quyết

1. Lấy **FinePDFs-Edu vi** (đề xuất: có), và có lấy cả **FinePDFs vi** đầy đủ không. Khi trùng với PDF tự trích (VJOL/VISTA/giáo trình), giữ bản nào (đề xuất: bản tự trích, vì có metadata và OCR cascade; FinePDFs xếp sau trong ưu tiên Q4).
2. Dùng **FineWeb2-HQ** làm cờ chất lượng cho FineWeb-2 không (đề xuất: có, không tải text).
3. Có nhận **dữ liệu dịch máy / tổng hợp** vào CPT không (`cosmopedia_vi`, `vi_pubmed`), và nếu có thì trần tỉ lệ bao nhiêu.
4. Có gửi email xin phép VOER, Tia Sáng, NXB Giáo dục, Bách khoa toàn thư không.
5. Có mở dự án **tự dịch OpenWebMath/FineMath** không (cần ngân sách GPU riêng).
6. Thêm **HUST DSpace** vào kế hoạch crawl chung với kho ĐHQGHN không.

## 6. Kết quả kiểm chất lượng trên mẫu (2026-11-10)

> Chốt sau khi xem kết quả (DECISION_LOG NS-13..NS-18): **lấy FinePDFs-Edu**; **FinePDFs bản đầy đủ chỉ lấy nhóm giáo dục + nghiên cứu (NS-18)**; **không lấy** HPLT 3.0, FineWeb2-HQ, vi_pubmed, MetaMathQA vi, Orca-Math vi, Cosmopedia dịch (NS-19). Còn mở: cách lọc FinePDFs bản đầy đủ.

Cách làm: `scripts/source_probe/probe_hf.py` lấy ~300 bản ghi mỗi nguồn (row group ngẫu nhiên của parquet trên HF, API datasets-server cho bộ JSON, phần đầu shard cho HPLT), đo bằng đúng stage language + quality của pipeline (fastText lid.176, điểm/band/reason code), đếm token Qwen3, thêm số đo LaTeX hỏng và dịch máy. Kết quả: `data/processed/source_probe/` (`summary.json`, `<nguồn>/measured.jsonl`). Thêm: đọc tay vài chục đoạn trích, đếm từ khoá spam cờ bạc / người lớn, đếm chỗ thiếu khoảng trắng sau dấu câu ("lại.Chúng").

**Phát hiện chung:** bộ lọc quy tắc của pipeline xếp gần như mọi văn bản web vào band A (FineWeb-2 ngẫu nhiên 98% A, HPLT bin 10 98% A) nhưng **không bắt được spam cờ bạc và văn bản xào (spin)**. Band A ở đây chỉ nghĩa là "hình thức sạch", không nói gì về hàm lượng kiến thức. Muốn so kiến thức phải đọc nội dung hoặc dùng classifier (điểm edu).

| Nguồn | Mẫu | Token/bản (trung vị) | Ước tính tổng | Band A / D | Vấn đề đo được | Đánh giá |
|---|---|---|---|---|---|---|
| FinePDFs-Edu vi | 300 | 2.219 | **~1,2 tỷ** (131.190 văn bản × 9.051 token trung bình) | 81% / 15% | 10% tiếng Anh/song ngữ; 21% `is_truncated` (PDF dài bị cắt); LaTeX hỏng chỉ 7%; classifier edu vẫn cho qua tài liệu tôn giáo, biểu mẫu, bài phát biểu | **Tốt nhất trong các nguồn mới.** Nhiều lời giải SGK, đề thi, tài liệu hướng dẫn, báo cáo khí tượng... |
| FinePDFs vi (đầy đủ) | 300 | 1.538 | **~7,4 tỷ** (1.229.248 × 6.028) | 68% / 20% | 24% `repetitive` (bảng biểu); nhiều báo cáo tài chính, biên bản họp cổ đông (static2.vietstock.vn đứng đầu), văn bản hành chính; chỉ 7% có điểm edu ≥ 2 | Khối lượng lớn nhưng kiến thức trung bình thấp. Lấy, nhưng đặt trọng số theo điểm edu có sẵn (`fw_edu_scores`) |
| FineWeb2-HQ vi | 300 | 444 | ~3,6 tỷ (tập con của FineWeb-2 đã chọn) | 99% / 0% | Phần lớn là **tin tức, giải trí** (VOA, vietbao, Thanh Niên, thể thao); spam 2% | "HQ" = viết tốt, không phải STEM. Hơn FineWeb-2 ngẫu nhiên (3–10% bản ghi có từ khoá cờ bạc tuỳ cách đếm, nhiều trang quảng cáo dịch vụ). Dùng làm cờ / trọng số, không phải nguồn kiến thức |
| HPLT 3.0 vi, bin WDS 10 | 300 | 3.493 | tổng HPLT vi ~260 GB nén (chưa ước token) | 98% / 0% | **35% có từ khoá cờ bạc** (V9Bet, nhà cái, soi cầu), văn bản xào từ, tán gẫu diễn đàn; 39% văn bản dính câu (mất xuống dòng) | **Không đề xuất.** Điểm WDS đo hình thức trang, không đo nội dung. Bin 5 ít spam hơn (11%) nhưng ngắn và vụn |
| Cosmopedia vi (math + Khan) | 300 + 240 | 764 / 967 | ~0,27 tỷ | 93% / 6% (math) | **6–13% bản dịch rỗng**; LaTeX hỏng ở **80%** bài có công thức (`\ frac`, `\ cdot`); 84–91% bài thiếu khoảng trắng sau dấu chấm; thuật ngữ dịch sai, viết hoa sai ("F (q)") | Chỉ dùng sau khi sửa: bỏ bài rỗng, chèn lại khoảng trắng, sửa `\ lệnh` → `\lệnh`; bài toán còn LaTeX hỏng thì bỏ |
| vi_pubmed | 300 | 137 | ~2,7 tỷ (20 triệu × 137, đã tính phần bị cắt) | 93% / 7% | **59% bản dịch bị cắt giữa câu** (bản Anh gần như không bị: 0,3%), bản Việt chỉ dài 0,65 lần bản Anh; 4% có ký tự "⁇" (token lạ, vd chữ μ) | Mô hình dịch có trần độ dài. Nếu dùng: cắt về câu trọn vẹn cuối cùng, bỏ bản có "⁇"; giá trị y sinh vừa phải, câu ngắn |
| MetaMathQA vi (SFT) | 300 | 229 | ~0,09 tỷ | 97% / 2% | Dịch trôi chảy; LaTeX hỏng 7% bài có công thức; thuật ngữ sai ("sự khác biệt chung" thay cho "công sai"); câu hỏi bị lặp lại | Dùng cho SFT, sau khi sửa LaTeX |
| Orca-Math vi (SFT) | 300 | 369 | ~0,07 tỷ | 96% / 0% | LaTeX hỏng 25% bài có công thức | Dùng cho SFT, sau khi sửa LaTeX |

Trùng với nguồn tự trích: 0/600 URL mẫu của FinePDFs đến từ vjol.info.vn hay sti.vista.gov.vn; ~18% đến từ tên miền `.edu.vn` / tạp chí; 2/300 bản ghi FinePDFs-Edu là PDF của **voer.edu.vn** (một phần VOER đã có sẵn trong FinePDFs qua CommonCrawl).

**Nguồn chưa đo được:**
- **HUST DSpace** (dlib.hust.edu.vn): ngày 2026-11-10 mọi đường dẫn (trang chủ, REST, OAI) trả **403** cho máy này, dù agent khảo sát vài giờ trước vẫn vào được. Không tìm cách vượt chặn. Cần thử từ server GPU, hoặc hỏi thư viện ĐHBK.
- **Phác đồ Bộ Y tế** (kcb.vn): robots.txt cho phép `/phac-do` và `/upload`. Thử 1 PDF (QĐ 1857/QĐ-BYT, suy tim cấp): 27 trang, có lớp chữ sạch (trung vị 1.695 ký tự/trang, không trang nào trống), không cần OCR. Số lượng nhỏ (vài chục đến vài trăm văn bản).
- Wikibooks/Wiktionary/Wikivoyage: chưa đo (nhỏ, giấy phép rõ, chất lượng như Wikipedia).

**Lỗ hổng của pipeline lộ ra:** stage quality chưa có mã cho spam cờ bạc / người lớn / văn bản xào. FineWeb-2 ngẫu nhiên có 3–10% bản ghi chứa từ khoá loại này (10/300 có "cá cược", 9/300 "nhà cái"; đếm thô bằng từ khoá, có báo nhầm, vd "sex" trong tài liệu y tế), SEA-PILE có lẽ cũng vậy (chưa đo). Nên thêm một mã chặn (cần chủ dự án đồng ý, chưa code).
