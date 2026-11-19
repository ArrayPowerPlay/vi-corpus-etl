# Kế hoạch crawl luận án / luận văn (Thư viện Quốc gia, ĐHQGHN)

> Trạng thái: **kế hoạch, chưa chọn, chưa tải gì**. Chưa ghi vào `docs/SOURCES.md` (theo yêu cầu ngày 2026-11-10).
> Số liệu đo ngày 2026-11-10 bằng vài chục request nhẹ (sitemap, trang metadata, API đếm). Không tải ảnh trang, không tải hàng loạt file.
> Chỗ nào là ước tính chứ chưa đo thì ghi rõ "ước tính".

## 1. Luận án tiến sĩ, Thư viện Quốc gia Việt Nam (luanan.nlv.gov.vn)

**Là gì.** Cơ sở dữ liệu toàn văn dùng phần mềm Veridian. Trang chủ ghi 20.395 luận án, 312.817 trang. Sitemap thực tế liệt kê 20.496 luận án và 317.963 trang (8 file sitemap, 358.956 URL). Có luận án tiếng Pháp, tiếng Anh, tiếng Nga..., phải lọc theo trường `Ngôn ngữ`.

**Lấy được gì, bằng cách nào.**

| Thứ cần lấy | Cách lấy | Số request | Ghi chú |
|---|---|---|---|
| Danh sách luận án | 8 file `sitemap1..8.xml` | 8 | mỗi file ~6 MB |
| Metadata + tóm tắt | `luanan?a=d&d=<mã luận án>`: tên, tác giả, nơi xuất bản, năm, số trang, ngôn ngữ, từ khóa, tóm tắt | 20.496 | HTML ~23 KB, đo được 0,07–0,45 s mỗi request |
| Nội dung từng trang | chỉ có dạng **ảnh** (`imageserver.pl`), trang HTML không chứa chữ OCR | 317.963 | phải tự OCR |

**Rào cản quan trọng (cần chủ dự án quyết).** `robots.txt` của site ghi `Disallow: /luanan/cgi-bin/imageserver/imageserver.pl` và `Disallow: /luanan?a=q` (tìm kiếm). Lấy ảnh trang (nguồn duy nhất của toàn văn) là đi ngược robots.txt. Tôi **chưa tải thử ảnh nào**. Trang điều khoản của site không mở được từ máy này. Hai hướng:

- **A. Chỉ metadata + tóm tắt** (đúng robots.txt): ~5,7 giờ, nhưng chỉ ra vài triệu token (ước tính, mỗi tóm tắt vài trăm đến vài nghìn ký tự). Không đáng kể so với quy mô hàng tỷ token.
- **B. Toàn văn qua ảnh trang**: giá trị cao (văn bản học thuật dài) nhưng cần **xin phép bằng văn bản** Thư viện Quốc gia (info@nlv.gov.vn) trước khi crawl ảnh. Bản quyền luận án thuộc tác giả, cũng như VJOL → gắn `unknown`, quarantine.

**Giới hạn tốc độ.**

- `robots.txt`: `Crawl-delay: 1` (1 giây giữa hai request). Đây là mức tối đa được phép, không đo thấy giới hạn cứng nào khác. Chưa thử vượt vì không có lý do làm vậy.
- Kế hoạch: 1 kết nối, 1 request/giây, User-Agent ghi rõ tên dự án + email liên hệ, gặp lỗi 5xx hoặc 429 thì lùi (backoff) rồi chạy tiếp, mỗi luận án một file checkpoint nên dừng/chạy lại không mất việc.

**Thời gian ước tính để crawl đầy đủ** (1 giây/request, một luồng):

| Việc | Tính | Thời gian |
|---|---|---|
| Sitemap | 8 request | vài giây |
| Metadata (hướng A) | 20.496 × 1 s | **~5,7 giờ** |
| Ảnh trang (hướng B) | 317.963 × 1 s | **~88 giờ ≈ 3,7 ngày** (chưa tính tải ảnh chậm hơn 1 s; cỡ ảnh chưa đo) |
| OCR toàn bộ ảnh bằng cascade VLM | 317.963 trang ÷ (4 GPU × 1,55 trang/s) | **~14 giờ** (dùng tốc độ đo trên trang stbook, trang luận án có thể khác) |

Tổng hướng B: khoảng 4 ngày crawl + 14 giờ OCR. Không thể rút ngắn bằng chạy song song nếu vẫn giữ crawl-delay 1 s.

## 2. Luận văn, luận án, kho ĐHQGHN (repository.vnu.edu.vn)

**Là gì.** DSpace 7.4, giao diện Angular, có REST API công khai ở `https://repository.vnu.edu.vn/server/api`. Toàn kho 114.998 mục, 114.714 mục có file.

**Phần luận văn / luận án** (nhãn `itemtype`): `Thesis` 46.687 (46.540 có file), `Luận văn thạc sĩ` 1.283, `Theses` 974, `Dissertation` 834, `Dissertations` 322 → khoảng **50.100 mục**. Ngoài ra có khóa luận (`Final Year Project (FYP)` 3.622, `Dự án cuối năm` 666), sách 18.393, bài báo 16.499... chưa tính vào kế hoạch này.

**Rào cản: một nửa file cần đăng nhập.** Thử 20 luận văn lấy ở một trang kết quả: 17 có file, trong đó 8 tải được ẩn danh (PDF thật, kiểm tra bằng byte đầu `%PDF-`) và 9 trả `401` (cần tài khoản thư viện). Mẫu rất nhỏ, chỉ từ một trang kết quả, nên tỉ lệ ~47% có sai số lớn. **Không vượt đăng nhập**; chỉ lấy file tải được ẩn danh. Ước tính ~23.500 PDF, trung bình 4,5 MB (mẫu 17 file, lớn nhất 58,5 MB) → ~105 GB. Số trang và việc PDF là scan hay có text chưa đo.

**Quyền.** `dc.rights` trống ở các mẫu; bản quyền thuộc tác giả → `unknown`, quarantine. Chưa tìm thấy điều khoản sử dụng của kho; nên hỏi ĐHQGHN (lic.vnu.edu.vn) trước khi crawl hàng loạt.

**Giới hạn tốc độ.**

- `robots.txt` không có `Crawl-delay`. Chỉ cấm `/search`, `/admin/*`, `/processes`, `/submit`, `/workspaceitems`, `/profile`, `/workflowitems`. REST API `/server/api` không bị cấm.
- Giao diện web trả header `X-RateLimit-Limit: 500` (chưa đo cửa sổ thời gian). API không trả header này; thử 60 request liên tiếp, 0,12 s/request, không có `429`.
- Kế hoạch: 1 kết nối, **tối đa 2 request/giây**, User-Agent có email, backoff khi gặp `429`/5xx, checkpoint mỗi mục. Không tải song song nhiều luồng.

**Thời gian ước tính** (2 request/giây):

| Việc | Tính | Thời gian |
|---|---|---|
| Liệt kê ~50.100 mục (100 mục/trang) | ~501 request | ~4 phút |
| Mỗi mục: bundles + bitstreams + thử tải 1 KB đầu để biết có bị 401 | 3 request × 50.100 | ~150.300 request ≈ **21 giờ** |
| Tải PDF ẩn danh | ~23.500 file, ~105 GB | **~3 giờ** nếu mạng 10 MB/s (chưa đo băng thông); cộng ~3,3 giờ nếu tính 2 request/s |
| **Tổng crawl** | | **~1 ngày** (24–30 giờ) |
| OCR / trích text | số trang chưa biết | chưa ước tính được; PDF có text layer thì dùng PyMuPDF, scan thì qua cascade VLM như VJOL (`PaperLayout`) |

Có thể cắt số request bằng cách lấy bundles kèm bitstreams trong một request (`embed=bitstreams`), nhưng chưa thử.

## 3. Việc cần chủ dự án quyết trước khi chạy

1. Thư viện Quốc gia: chọn hướng A (chỉ tóm tắt) hay B (toàn văn, cần xin phép trước vì robots.txt cấm ảnh trang).
2. Có gửi email xin phép Thư viện Quốc gia và ĐHQGHN không (đề xuất: có, cả hai).
3. ĐHQGHN: chỉ lấy file tải được ẩn danh (đề xuất), hay xin tài khoản/dữ liệu thẳng từ thư viện.
4. Có lấy luận án ngoại ngữ không (tiếng Pháp, Anh...). Giáo trình tiếng Anh đã được giữ (xem `docs/DECISION_LOG.md`), nên đề xuất giữ.

## 4. Chưa có code

Chưa viết crawler cho hai nguồn này. Khi chốt, theo quy ước một thư mục script mỗi nguồn: `scripts/nlv_luanan/` và `scripts/vnu_repo/`, checkpoint theo mục, kết quả vào `data/raw/`; phần trích text dùng lại bộ xử lý PDF bài báo của VJOL/VISTA.
