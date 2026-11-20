# Chiến lược xử lý giáo trình (Google Drive "Tổng hợp giáo trình Đại học")

> Trạng thái: **xử lý thô đã viết** (mục 2-4: kiểm kê, trích text, làm sạch trang, `iter_records`, xuất Parquet), dựa trên khảo sát
> ngày 30/09/2026. Chạy: `uv run --no-sync python scripts/giao_trinh/extract.py --data-root <root> [--no-ocr] [--gpu G --serve --vllm-bin <vllm>] [--limit-files N] [--retry-skipped] [--multi-process] [--export-only] [--status]`; 4 GPU: `DATA_ROOT=<root> VLLM_BIN=<vllm> ./scripts/giao_trinh/run_extract_vlm.sh`
> (xuất `processed/giao_trinh/giao_trinh.parquet` tự động ở cuối mỗi lần chạy). Code: `vi_corpus/giao_trinh/` (`extract.py`, `records.py`),
> `vi_corpus/common/pdf_text.py` (`extract_pdf`, `clean_pages`). Đã thử E2E với dữ liệu tổng hợp (kill -9 rồi chạy lại đúng lệnh cũ);
> **Cập nhật 2026-11-20 (GT-01, GT-02)**: lượt trích lớp chữ đã chạy trên server (không OCR): 1.687 file, 1.663 nội dung duy nhất đã có checkpoint
> (1 lỗi: PDF bị cắt cụt, bị loại có chủ ý; 20 file bỏ qua do thiếu LibreOffice / djvulibre đã làm lại sau khi cài, GT-04), 1.569 bản ghi trong
> `giao_trinh.parquet`. Trang theo method: `text` 144.475, `pptx` 3.319, `djvu` 820, `tcvn3` 2.524, `empty` 294, `docx` 1, **`needs_ocr` 35.663** (chưa vào bản ghi). OCR trang scan **đã nối vào chuỗi VLM**
> (O-20: `qwen3vl_8b_mt4k` -> `rp110` -> baseline -> bỏ trang) qua `CascadeOcrGetter`, kèm cache OCR theo trang (`interim/giao_trinh_ocr_cache/`) và
> `scripts/giao_trinh/run_extract_vlm.sh` (4 GPU, chia file bằng khoá); **chưa chạy thật trên GPU** (việc S-30 trong `docs/project/SERVER_TASK.md`).
> Nhận diện font TCVN3 nay quyết ở mức trang (`page_has_tcvn3_evidence`): một span `.Vn*` đạt ngưỡng thì mọi span `.Vn*` của trang được đổi; font có
> tên `VNI-` không bật chế độ đổi cả tài liệu. Pipeline chung mới chạy thử 100 chunk (`giao_trinh_cpu100_20261119`, đến stage `quality`).
> Chưa làm (theo yêu cầu "chỉ xử lý thô"): phân loại `doc_type`/`language`
> (mục 5), quality, dedup ngoài sha256 file (mục 6). **2026-10-08 (D-05, R-14)**: đã thêm parser docx, doc / ppt (LibreOffice), djvu
> (djvulibre), html, epub ở `vi_corpus/common/parsers/`, nhận định dạng theo nội dung đầu file; máy thiếu LibreOffice / djvulibre thì file bị
> `skipped` (`needs_libreoffice` / `needs_djvulibre`), cài xong chạy lại với `--retry-skipped`. Parser mới chưa thử trên file thật của Drive.
>
> Khác thiết kế: file lỗi (PDF hỏng/mật khẩu) được ghi lỗi vào checkpoint nhưng **thử lại ở lần chạy sau**; checkpoint còn trang
> `needs_ocr` cũng được làm lại khi chạy có OCR. `clean_pages` ghép cả các trang thành một văn bản (đoạn nối được qua ranh giới trang);
> slide pptx thì giữ mọi dòng (không bỏ tiêu đề chạy, mỗi dòng một đoạn) vì slide lặp cấu trúc dễ bị bỏ nhầm. Trang `needs_ocr` không đưa vào bản ghi.
> Checkpoint OCR là theo file (crash giữa một sách scan dài mất cả cuốn đó).
> Quyết định của người dùng: **giữ cả sách tiếng Anh** (gắn nhãn ngôn ngữ, không loại).

## 1. Dữ liệu thực tế (đã kiểm tra)

**Cấu trúc**: `<ngành>/<môn>/<file>`, 16 ngành. Liệt kê được ≥ 1.498 file (1.368 PDF, 59 pptx, 25 ppt, 1 docx, 1 djvu, 44 mục là
shortcut tới thư mục). Con số thật lớn hơn: 3 thư mục (Lý thuyết điều khiển tuyến tính, Mạch tuyến tính, Nguyên lý trường điện từ)
bị Google cắt ở đúng 50 mục khi liệt kê không đăng nhập. Kỹ thuật điện (≥ 628) và Điện - Điện tử (≥ 353) chiếm 2/3.

**Gần như mọi file là shortcut** trỏ tới file/thư mục của người khác (vd `Bài giảng sản phụ khoa tập 1 2006.PDF` thực chất là
shortcut tới một *thư mục*). Hệ quả:
- Tải bằng `rclone` (đi theo shortcut, tải file gốc). `gdown` không dùng được (Access denied, giới hạn 50 file/thư mục).
- Cùng một file gốc có thể xuất hiện ở nhiều môn (vd `Lecture1_Cuong.ppt` ở 2 thư mục) → dedup theo sha256 file.
- File gốc của người khác có thể bị xoá / giới hạn quota → một phần file có thể tải thất bại; ghi lại, không chặn cả lô.

**Mẫu đã chạy thật** (tải về, qua `extract_pdf`):

| File | Loại | Kết quả |
|---|---|---|
| `Mật mã học.pdf` (289 tr) | sách, Unicode | 289/289 trang có text, sạch. Công thức (font Symbol) vỡ: `C CCCCCCCC =`, số mũ mất (`TE mod N` = T^E). |
| `Giao Trinh Mat Ma Hoc Hè đọc.pdf` (220 tr) | sách, Unicode + **bìa gõ font TCVN3** | Thân sách sạch; bìa `VnArialH` ra `NGUYÔN THÞ THU Hμ` → **đã sửa**, nay ra `NGUYỄN THỊ THU HÀ`. |
| `dien_tu_cong_suat_doan_quang_vinh_97.pdf` (197 tr) | slide, trang xoay dọc | 185 trang text + 12 trang cần OCR. Mất dấu cách `tựnhiên`, `mởthyristor` → **đã sửa** (đo khoảng trống theo hướng dòng). |
| `Session 4_EinRide.pptx` | slide tiếng Anh | 13 slide, chỉ 3,6k ký tự (câu hỏi thảo luận). Giá trị thấp. |
| `3.Socket_APIs_slide.pdf` | slide Anh-Việt trộn | Chỉ tiêu đề + gạch đầu dòng + code C. |

Đặc điểm chung: text nằm từng dòng (câu bị ngắt giữa chừng); mỗi trang lặp tiêu đề chạy (tên sách / "Chương n: ...") và số trang;
đoạn toàn ký hiệu (công thức, nhãn hình vẽ) chiếm ~3% (sách thường) đến ~13% (sách toán, slide điện tử) số đoạn.

## 2. Luồng xử lý

```
rclone copy ─► raw/giao_trinh/<ngành>/<môn>/<file>      (giữ nguyên, không sửa)
   │
   ├─ 1. inventory   : sha256 từng file, ngành/môn từ đường dẫn, định dạng; trùng sha256 → chỉ xử lý 1 bản
   ├─ 2. extract     : theo định dạng (mục 3) → interim/giao_trinh_text/<sha256>.json  (từng trang + method)
   │                   checkpoint theo file như bước OCR stbook; trang scan → OCR (PaddleOCR detect + VietOCR)
   ├─ 3. clean trang : bỏ header/footer lặp, số trang; ghép dòng thành đoạn (mục 4)
   ├─ 4. phân loại   : doc_type = book | slide ; language = vi | en | mixed (theo đoạn → tài liệu)
   └─ 5. iter_records: mỗi tài liệu 1 bản ghi CORPUS_SCHEMA → các stage chung (quality, dedup, knowledge unit)
```

Khoá của interim là **sha256 file** (không phải đường dẫn): tự dedup exact, đổi tên/chuyển thư mục không phải chạy lại.
`source_path` vẫn giữ đường dẫn đầu tiên; các đường dẫn trùng khác ghi vào manifest để truy vết.

## 3. Trích text theo định dạng

| Định dạng | Số file | Cách xử lý |
|---|---|---|
| PDF có lớp chữ | đa số | `extract_pdf` (PyMuPDF `rawdict`): chèn lại dấu cách mất, đổi span font TCVN3 theo tên font, cả tài liệu TCVN3 theo tỉ lệ ký tự. |
| PDF scan / trang scan | chưa rõ (chạy `profile.py`) | Trang < 30 ký tự mà có ảnh → OCR. Dùng lại `vi_corpus.common.ocr` (GPU). Sách y học (share từ testyhoc) khả năng cao là scan. |
| pptx | 59 | `python-pptx` (đã có trong deps): text frame + bảng + ghi chú, mỗi slide 1 "trang". |
| ppt, docx, djvu | 27 (~2%) | ~~Bỏ qua ở v1~~ Đã có parser (2026-10-08): docx bằng python-docx, doc / ppt đổi sang docx / pptx bằng LibreOffice headless, djvu bằng djvulibre (trang không chữ thì OCR). Thiếu công cụ thì `skipped`, chạy lại bằng `--retry-skipped`. |

PDF có mật khẩu / hỏng: ghi lỗi vào checkpoint, không dừng cả lô (giống stbook).

## 4. Làm sạch theo trang (trước khi ghép thành 1 văn bản)

Phải làm khi **còn ranh giới trang**, nên thuộc bước này chứ không phải stage normalize chung. Đã thử trên 3 PDF thật ở trên:

1. **Header/footer**: dòng nằm trong 2 dòng đầu / 2 dòng cuối trang, sau khi thay chữ số bằng `#`, lặp ở ≥ 3 trang → bỏ.
   Bắt được `Giáo trình mật mã học và hệ thống thông tin an toàn`, `Chương #: ...`, số trang. Bỏ ~9-18% số dòng.
2. **Dòng chỉ có số** (số trang lẻ) → bỏ.
3. **Ghép dòng**: nối dòng vào đoạn trước; chỉ xuống đoạn khi dòng trước kết thúc bằng `.:?!;` và dòng sau bắt đầu bằng chữ hoa
   hoặc gạch đầu dòng; từ tiếng Anh bị gạch nối cuối dòng (`exam-`/`ple`) → nối liền.
4. **Không sửa nội dung** (theo PIPELINE): công thức vỡ không cố khôi phục; chỉ gắn cờ ở bước quality.

Không dùng `get_text(sort=True)`: một số PDF là 2 trang sách trên 1 trang giấy ngang, sắp theo toạ độ sẽ trộn 2 cột.

## 5. Phân loại và chất lượng

- **doc_type**: `slide` nếu trung bình < 500 ký tự/trang (slide mẫu ~190, sách ~1.100-1.500); ngược lại `book`.
  Không dùng tỉ lệ khung trang: slide xuất PDF có thể xoay thành trang dọc.
- **language**: tỉ lệ chữ cái riêng của tiếng Việt (`ạ ầ ế ơ ư đ ...`) trên tổng chữ cái: sách Việt đo được 21-27%, tiếng Anh ~0%.
  Theo đoạn: ≥ 5% → `vi`, < 1% → `en`; tài liệu là `mixed` nếu cả hai chiếm ≥ 20%. Không cần thư viện ngoài.
  Sách tiếng Anh **giữ lại** (quyết định của người dùng), nhãn `en` để token accounting tách riêng.
- **quality** (reason code, không xoá cứng):
  - `symbol_heavy`: đoạn có < 50% chữ cái (công thức vỡ, nhãn hình vẽ) → loại khỏi text CPT, giữ ở bản thô.
  - `slide_low_text`: slide < 1.000 ký tự/tài liệu → band D.
  - `ocr_text`: tài liệu > 50% trang là OCR → band thấp hơn 1 bậc.
  - band gợi ý: sách `vi` sạch → A; sách `en` → B; slide → C; còn lại D.
- **rights_status**: tài liệu tổng hợp từ nhiều nguồn (PDFDrive, z-lib, sách NXB, bài giảng trường) → `unknown` theo mặc định của
  registry (quarantine ở tầng xuất bản). Không cản bước xử lý; cần người dùng/nhóm quyết khi xuất corpus.

## 6. Dedup
- Exact file: sha256 file (mục 2), bắt các shortcut trùng.
- Exact text: `source_sha256` của text đã làm sạch (bản PDF khác nhau nhưng cùng nội dung).
- Near-dup: MinHash ở stage dedup chung (PIPELINE 6b). Giáo trình hay có nhiều bản của cùng một môn
  (bản đầy đủ + từng chương tách lẻ, vd `Digital Logic Circuit Analysis and Design.pdf` và `Digital_Logic_...pdf`).
  Chương lẻ nằm trọn trong sách đầy đủ → cần so theo **đoạn** (containment), không chỉ theo cả tài liệu.

## 7. Việc cần làm, theo thứ tự
1. Tải: `rclone copy gdrive: data/raw/giao_trinh --drive-root-folder-id 1DRUL5cFHIusB3PoLXJ6tcpuSNJjfZUTt ...` (hướng dẫn trong phiên chat).
2. Khảo sát: `uv run --group ocr python scripts/giao_trinh/survey.py --data-root ./data` → tỉ lệ scan/TCVN3/tiếng Việt thật theo ngành.
   Kết quả này quyết định có cần chạy OCR GPU hay không và trong bao lâu.
3. (Xong) Viết `vi_corpus/giao_trinh/` (inventory + extract có checkpoint + `iter_records`) và `scripts/giao_trinh/extract.py`;
   bước clean trang đặt trong `vi_corpus/common/pdf_text.py` để VISTA/VJOL dùng lại.
4. (Xong) Thêm `giao_trinh` vào `vi_corpus/common/registry.py`.
5. (Xong, D-05) ppt/docx qua LibreOffice, djvu, html, epub. OCR giáo trình bằng chuỗi VLM: đã code (GT-01), chờ chạy GPU (S-30). Để sau: công thức toán (Nougat/Marker trên GPU) cho nhóm Toán nếu cần chất lượng cao.
