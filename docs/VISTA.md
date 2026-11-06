# VISTA: khảo sát dữ liệu và kế hoạch xử lý

> Lập 2026-11-05 (VS-01 trong `docs/DECISION_LOG.md`); chủ dự án đã trả lời đủ 6 câu hỏi (VS-02, VS-03, mục 5). **Trạng thái 2026-11-06: V1 (code) xong, V0 (chép về server, `docs/SERVER_TASK.md` S-25) chờ chạy, GPU chờ lượt full VJOL.**
> Nguyên tắc: xử lý VISTA **giống hệt VJOL** (trích text 2 lượt PyMuPDF -> chuỗi VLM, bộ dò chữ hỏng, làm sạch lúc sinh bản ghi, nối metadata, pipeline dùng chung), chỉ viết phần khác nhau.

## 1. Dữ liệu trên MinIO (đã xem 2026-11-05)

Nguồn: `sti.vista.gov.vn` (Cục Thông tin KH&CN quốc gia, cơ sở dữ liệu bài báo khoa học Việt Nam). Bucket `datacrawl`, thư mục `sti.vista.gov.vn/`.
Đường public `storage-ailab.icenter.ai` chạy được với rclone (`sign_accept_encoding=false`); MCP minio không trả kết quả hôm đó.

```
sti.vista.gov.vn/
├── 1998/ ... 2026/<publication_id>.pdf      # tên file = publication_id (KHÁC VJOL: <article_id>_<galley_id>.pdf)
├── unknown/<sha256>.pdf                     # 19 file, tên = sha256 nội dung; không có trong log tải (bản thử cũ, Advances in Natural Sciences 2006, tiếng Anh)
└── metadata/
    ├── vista_detail.jsonl      643 MB   323.513 dòng, mỗi dòng một bài (khoá publication_id, không trùng)
    ├── vista_listing.jsonl     406 MB   danh sách từ trang tìm kiếm (slug, record_hash, source_parts, query_year/cat)
    ├── pdf_download_log.jsonl   53 MB   284.743 dòng: publication_id, status, path, sha256, bytes
    └── vista_partitions.json    36 KB   174 phân vùng (năm × 6 lĩnh vực), ước 399.540 bài
```

Trường của `vista_detail.jsonl`: `publication_id`, `title_vi`, `title_en`, `abstract_vi`, `abstract_en`, `authors`, `authors_raw`, `keywords`, `keywords_en`,
`source_name` (tên tạp chí), `issn`, `issue_year`, `issue_number`, `pages` ("179-185"), `research_field_code` / `research_field_name` (mã lĩnh vực 7 chữ số),
`document_type` ("" hoặc "BB"), `shelf_mark`, `fulltext_urls`, `has_fulltext`, `detail_url`. **Không có** trường ngôn ngữ, không có DOI.

### Số liệu chính

| Mục | Số |
|---|---|
| Bài có metadata chi tiết | 323.513 (đều `has_fulltext=true`), 2.774 tên tạp chí |
| Có tóm tắt tiếng Việt / tiếng Anh | 153.795 / 104.206 |
| Lượt tải PDF: thành công / lỗi 502 / PDF hỏng / 404 | 283.239 / 1.314 / 182 / 7 |
| Dung lượng PDF tải được | 233,7 GB; trung vị 341 KB, p99 9,2 MB, lớn nhất 151 MB |
| PDF khác nhau (theo sha256) | 273.140 → **9.708 nhóm PDF dùng chung cho nhiều bài** (19.807 bài, nhóm lớn nhất 15 bài) |
| Bài có metadata nhưng **chưa tải** | 40.988, dồn ở 1998–2001 (1998: 4.516/4.640, 1999: 5.009/5.025, 2000: 5.439, 2001: 6.774) và 2003, 2014, 2018–2020 |
| PDF tải được nhưng không có metadata chi tiết | 2.194 (thiếu năm) |
| Phân vùng bị cắt ở trang 501 (thiếu bài trong listing) | 3: Khoa học xã hội 2021, 2022, 2023 (~4.700 bài) |
| Trùng tiêu đề nguyên văn với VJOL | ~64.700 tiêu đề (~20%), chủ yếu 2020–2026 |

Kiểm tra khớp: thư mục `2026/` trên MinIO có 8.323 file, log có 8.315 bài 2026 đã tải; 2.000 file đầu khớp cả kích thước. Nhưng `2001/` chỉ có 163 file
trong khi log ghi 180 đã tải (2 file mẫu thiếu) → **bản chép lên MinIO có thể chưa đủ** hoặc việc tải còn đang chạy (log sửa lần cuối 2026-11-04).
Liệt kê cả kho qua đường public rất chậm (~3,5 phút cho 8.300 file), cả kho ước ~2 giờ, nên chưa đếm hết.

### Khảo sát 85 PDF mẫu (3 bài / năm, chạy bộ trích text VJOL không OCR)

| Nhóm | Bài | Ghi chú |
|---|---|---|
| Ảnh scan hoàn toàn | 50 | 1998–2008 gần như 100% (producer `Adobe PDF Library 5.0` bọc ảnh scan); vài bài 2016–2021 |
| Có lớp chữ tốt | 29 | chủ yếu 2017–2026 (Word, Adobe 15/17, Pdftools); có bài tiếng Anh |
| Lẫn | 6 | |

**Phát hiện quan trọng: lớp chữ OCR đời cũ không dấu.** 2009–2019 nhiều PDF có lớp chữ do phần mềm OCR cũ tạo (producer `OCR Manager`, `ORPALIS PDF Reducer`, `GdPicture`),
mất hết dấu và sai chữ: "Nang cao chat lifgrng kiem toan noi bo", "KHOA HOC VA CONG NGHE MO". Bộ dò chữ hỏng của VJOL (`garbled`) đã bắt phần lớn
(trang `garbled_text_layer` / `garbled_doc`), nhưng **bỏ sót 3/27 bài** khi chạy local (local không có `syllables.json`, nên tín hiệu âm tiết lạ tắt; trên server có thể bắt thêm).
Đây giống trường hợp `ABBYY FineReader 11 ... ScanGate` của VJOL (J-29) → đề xuất luật "producer OCR cũ thì OCR cả bài", **đo trước rồi mới bật** (mục 3, bước V2).

Trang khổ nhỏ (295×440 pt) là trang tạp chí in 2 cột, scan; tạp chí phổ thông / chính trị ("Cộng sản", "Lý luận chính trị") cũng có trong kho, không chỉ bài nghiên cứu.

### Ước lượng khối lượng (nhân tỉ lệ mẫu theo năm với số PDF từng năm, sai số lớn vì mẫu nhỏ)

- ~273.000 PDF, trung bình ~5,8 trang/bài → **~1,65 triệu trang**.
- Trang phải OCR: **~0,7 triệu** (luật hiện tại của VJOL) đến **~0,95 triệu** (thêm luật producer OCR cũ).
- Theo tốc độ bench stbook (~1,55 trang/giây/GPU, O-22): 0,95 triệu trang / (4 × 1,55) ≈ **42 giờ đồng hồ trên 4 GPU**. Phải đo lại trên VISTA ở bước V3.

## 2. Khác nhau giữa VISTA và VJOL (chỉ những chỗ phải code)

| Điểm | VJOL | VISTA | Cách làm |
|---|---|---|---|
| Tên file | `<năm>/<article_id>_<galley_id>.pdf` | `<năm>/<publication_id>.pdf`; `unknown/<sha256>.pdf` | hàm `path_meta` riêng |
| Nối metadata | `vjol_merged.jsonl` theo `article_id` | `vista_detail.jsonl` theo `publication_id` | đọc theo luồng một lượt, chỉ giữ dòng cần (như J-03) |
| Một PDF ↔ nhiều bài | hiếm | 9.708 nhóm, tới 15 bài / PDF (PDF cả số tạp chí hoặc bản trùng) | checkpoint theo sha256 nên chỉ trích một lần; **một bản ghi / PDF**, `meta.publication_ids` liệt kê mọi bài; đánh dấu `meta.whole_issue` khi số trang PDF lớn hơn nhiều so với trường `pages` |
| Lớp chữ OCR cũ không dấu | ít (27 bài ScanGate) | rất nhiều, 2009–2019 | đo rồi thêm `OCR_ALL_PRODUCERS` riêng của VISTA |
| Tỉ lệ scan | thấp | cao (gần hết trước 2009) | cùng chuỗi VLM; khối GPU lớn hơn nhiều |
| Tóm tắt trong metadata | không có | 153.795 tóm tắt tiếng Việt | (a) **thước đo chất lượng OCR miễn phí**: độ trùng từ giữa tóm tắt và text trang đầu (giống `font_canary.word_overlap`); (b) cặp tóm tắt ↔ thân bài cho SFT sau này |
| Nhóm font xấu, bảng lệch dấu | học trên VJOL | chưa có | chạy lại `font_canary` / `learn_tone_shift` trên mẫu VISTA, ghi file riêng |
| Trùng với nguồn khác | ưu tiên 0 | ~20% tiêu đề trùng VJOL | để dedup toàn cục xử lý (VJOL giữ bản), không bỏ trước khi trích |

Phần **dùng lại nguyên** của VJOL: `pdf_text.extract_pdf` (detect_garbled, rescue_empty, ocr_cache), `ocr_cascade`, `batch_extract.run_batch` (khoá flock, J-61, J-62),
`vjol/clean.py` (bìa, mục lục, tài liệu tham khảo, tiêu đề tạp chí theo `source_name`), `hyphen_fix`, `reocr` (defect_doc), `defects.py`, `text_md` (T-06),
`audit_run.py`, `groundtruth.py`, `compare_runs.py`, pipeline `PROFILES` (chunked, vi + en, odd_words, perplexity).

## 3. Kế hoạch

**Cách code (đã chọn, VS-02; làm ở V1, xem mục 6):** tách phần "bài báo PDF" của `vi_corpus/vjol/` thành hàm nhận tham số nguồn (thư mục checkpoint, thư mục claim, `path_meta`, đọc metadata,
`OCR_ALL_PRODUCERS`), VJOL gọi với tham số cũ. Thêm `vi_corpus/vista/` mỏng (`layout.py`, `records.py`) và `scripts/vista/` (extract, run_extract_vlm.sh).
Điều kiện bắt buộc: **checkpoint VJOL đang chạy full (S-23 / S-24) không bị làm lại** (giữ nguyên `EXTRACT_VERSION` và chuỗi `extract_version` của VJOL),
test VJOL cũ + golden `tests/data/vjol_golden.jsonl` vẫn qua. Cách B (chép cả `vjol/` sang `vista/`) nhanh hơn nhưng mỗi lần sửa lỗi phải sửa hai nơi, không đề xuất.

| Bước | Việc | Máy | Ước lượng |
|---|---|---|---|
| **V0** | Chép `sti.vista.gov.vn/` về server thành `raw/vista/` (chữ thường như J-02): `rclone copy --fast-list --transfers 16`. Kiểm kê: đếm file theo năm so với log, so sha256 với log (bắt file chép dở), liệt kê PDF không có trong log / không có metadata | server, CPU | 234 GB, vài giờ tuỳ đường mạng |
| **V1** | Code: tách phần dùng chung (cách A), `vista/layout.py` + `records.py` (nối `vista_detail.jsonl`, `publication_ids`, `whole_issue`), registry `vista` → `raw/vista`, `run_all.py` thêm phần `vista`, `PROFILES["vista"]` = như vjol, ingest, `SOURCE_PRIORITY` (VJOL trước VISTA). Test không mạng (PDF tổng hợp nhỏ + metadata giả) | local | 1 phiên |
| **V2** | Đo trên CPU trước khi tốn GPU (mẫu 3.000 PDF chia đều theo năm): `check_garbled.py` cho VISTA, bảng producer × tỉ lệ trang bị gắn cờ, độ trùng tóm tắt ↔ trang đầu của lớp chữ hiện có, đếm PDF cả số tạp chí. Chủ dự án chọn có bật luật producer OCR cũ không | server, CPU | < 1 giờ |
| **V3** | Trích mẫu 3.000 PDF trên 4 GPU (**sau khi lượt full VJOL xong**), đo trang/giây thật, tỉ lệ trang OCR, cờ guard. Chạy `font_canary` + `learn_tone_shift` cho VISTA. Pipeline `vista_10k` (10.000 đoạn), `audit_run.py`, đọc tay 30 bài, so độ trùng tóm tắt sau OCR | server, 4 GPU | vài giờ + pipeline ~15 phút |
| **V4** | Trích text cả kho (~273.000 PDF), theo dõi 1–2 giờ đầu như J-60 | server, 4 GPU | ~2 ngày (ước, đo ở V3) |
| **V5** | Pipeline cả kho + dedup vòng 2 chung với VJOL (`dedup_global.py`), báo cáo phần trùng VJOL | server | tuỳ F-04 / F-05 (RAM) |

Mỗi bước sẽ có mục `S-xx` trong `docs/SERVER_TASK.md` kèm lệnh chính xác khi code xong.

## 4. Rủi ro

- Bản trên MinIO có thể chưa đủ (2001: 163 file / 180 trong log) và ~41.000 bài chưa tải (chủ yếu 1998–2001). Trích trước phần đang có vẫn được vì checkpoint theo file; phần tải thêm sau chỉ làm phần mới.
- GPU đang bận lượt full VJOL; VISTA cần thêm ~2 ngày GPU.
- Lớp chữ OCR cũ không dấu mà bộ dò bỏ sót sẽ lọt vào corpus dưới dạng chữ không dấu: cần V2 trước khi chạy full.
- PDF cả số tạp chí (tới 151 MB) gán cho nhiều bài: lineage theo bài không chính xác; giữ text, gắn nhãn.
- Quyền: chưa rõ → `rights_status=unknown`, chỉ gắn nhãn quarantine (R-01) như VJOL.

## 5. Câu hỏi cho chủ dự án

Đã trả lời (2026-11-06, VS-02):
1. Tải VISTA đã xong: không tải thêm. V0 vẫn đếm lại sau khi chép về server.
2. Code theo cách chung (cách A), tận dụng phần đã code của VJOL.
3. Chờ lượt full VJOL xong mới chạy VISTA trên GPU. Viết code và đo trên CPU không cần chờ.

Chốt nốt (2026-11-06, VS-03):
4. Luật "PDF do phần mềm OCR đời cũ tạo thì OCR lại cả bài": ban đầu "đo ở V2 rồi mới quyết" (VS-03), sau đó chủ dự án quyết **bật ngay** (VS-05). Danh sách producer: `OLD_OCR_PRODUCERS` ở `vi_corpus/vista/layout.py`; đối chiếu bằng `scripts/vista/producer_survey.py`.
5. PDF dùng chung cho nhiều bài: một bản ghi cho mỗi PDF, kèm `meta.publication_ids` (cách A).
6. 19 file `unknown/`: giữ, bản ghi thiếu metadata.

## 6. Đã code ở V1 (2026-11-06, VS-04)

Dùng lại bộ xử lý bài báo của VJOL, không chép code. Thêm một `PaperLayout` (`vi_corpus/vjol/layout.py`) mô tả phần khác nhau của từng nguồn và một `RecordAdapter` (`vi_corpus/vjol/records.py`) cho phần nối metadata:

| Việc | VJOL | VISTA |
|---|---|---|
| Bố cục / đường dẫn | `VJOL_LAYOUT` ở `vi_corpus/vjol/extract.py` (giữ nguyên thư mục checkpoint `interim/vjol_text`, khoá `state/vjol_extract/claims`, `extract_version`) | `vi_corpus/vista/layout.py`: `interim/vista_text`, `state/vista_extract/claims`, `processed/vista/vista.parquet`, nhóm font xấu riêng, chưa có luật producer OCR cũ (VS-03) |
| Bản ghi | `VJOL_ADAPTER` (kết quả y như trước) | `vi_corpus/vista/records.py`: một bản ghi / PDF, `meta.publication_ids`, `meta.shared_pdf`, metadata lấy từ bài mã nhỏ nhất có dòng trong `vista_detail.jsonl`; `unknown/` giữ, thiếu metadata |
| Dòng lệnh | `scripts/vjol/extract.py` (nay chỉ gọi `vi_corpus/vjol/cli.py`) | `scripts/vista/extract.py`, `scripts/vista/run_extract_vlm.sh` (4 GPU), `scripts/vista/check_raw.py` (kiểm bản chép) |
| Nối vào pipeline | `PROFILES["vjol"]` | registry `vista` -> `raw/vista`, `PROFILES["vista"]` = hồ sơ VJOL, nhánh `vista` trong `ingest.py`, phần `vista` trong `scripts/run_all.py` (chạy sau `vjol`) |

Thứ tự ưu tiên khi trùng: VISTA nằm nhóm "các dataset khác" (R-15) nên VJOL (mức 0) được giữ trước. Test: `tests/test_vista.py` (8 test), toàn bộ 379 test qua, VJOL không đổi (cùng checkpoint, cùng `meta`).

Chưa làm (đợi V2/V3): nhóm font xấu và bảng lệch dấu riêng của VISTA, đo độ trùng tóm tắt với trang đầu, đánh dấu PDF cả số tạp chí (`whole_issue`), cờ `PROFILES["vista"]` riêng. `scripts/vjol/run_extract_vlm.sh` giữ nguyên (không sửa file shell đang có thể được bash đọc dở); sau khi lượt full VJOL xong có thể gộp hai file shell làm một.
