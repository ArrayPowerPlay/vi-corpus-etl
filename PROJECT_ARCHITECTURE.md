# Kiến trúc dự án vi-corpus-etl

Tài liệu mô tả **cấu trúc repo và cấu trúc dữ liệu**. Thiết kế các bước xử lý (parse → language → normalize → quality → dedup → knowledge unit) nằm ở [`docs/PIPELINE.md`](docs/PIPELINE.md); danh mục nguồn ở [`docs/SOURCES.md`](docs/SOURCES.md); chiến lược giáo trình ở [`docs/GIAO_TRINH.md`](docs/GIAO_TRINH.md).

Nguyên tắc chung:
- **Mỗi nguồn dữ liệu có gói code riêng (`vi_corpus/<nguồn>/`) và thư mục script riêng (`scripts/<nguồn>/`)**; phần dùng chung nằm ở `vi_corpus/common/`.
- **Raw-first**: `data/raw/` là bản gốc, không bao giờ bị sửa. Mọi bước xử lý đọc `raw/` và ghi ra `interim/` hoặc `processed/`.
- **Checkpoint theo đơn vị nhỏ** (file / cuốn sách), ghi nguyên tử; crash thì chạy lại đúng lệnh cũ.
- Một script tổng, `scripts/run_all.py`, chạy lần lượt script của từng nguồn.

## 1. Cây thư mục code

```
vi-corpus-etl/
├── CLAUDE.md, README.md, PROJECT_ARCHITECTURE.md
├── pyproject.toml, uv.lock          # thư viện quản lý bằng uv (nhóm ocr: uv sync --group ocr)
├── .env                             # HF_TOKEN (không đưa lên git)
├── docs/                            # PIPELINE, SOURCES, ROADMAP, GIAO_TRINH
├── scripts/
│   ├── run_all.py                   # script tổng: chạy các script con bên dưới
│   ├── sea/                         # download_all.py, download_sea_*.py (3 bộ), count_rows.py
│   ├── stbook/                      # crawl.py (tải sách), ocr.py (OCR PDF → text)
│   ├── giao_trinh/                  # extract.py (xử lý thô giáo trình), profile.py (khảo sát)
│   ├── gpu_keepalive.py             # giữ GPU trên Run:ai không bị tự dừng khi job chỉ dùng CPU
│   └── run_with_gpu_keepalive.sh    # chạy một script bất kỳ kèm gpu_keepalive, chạy nền
├── vi_corpus/
│   ├── common/                      # dùng chung cho mọi nguồn
│   │   ├── registry.py              #   sổ đăng ký nguồn: owner, license, domain, định dạng, đường dẫn raw
│   │   ├── schema.py                #   schema chung của clean corpus + KPI truy vết nguồn
│   │   ├── state.py                 #   ghi JSON nguyên tử, khoá chống chạy trùng, cấu hình log
│   │   ├── ocr.py                   #   OCR một trang: PaddleOCR tìm dòng + VietOCR đọc chữ
│   │   └── pdf_text.py              #   trích text từ PDF có lớp chữ (PyMuPDF)
│   ├── sea/                         # nguồn SEA (Hugging Face)
│   │   ├── datasets.py              #   danh sách 3 bộ: repo, thư mục tiếng Việt
│   │   ├── hub.py                   #   token, liệt kê file, tải có retry, kiểm tra kích thước/sha256
│   │   ├── checkpoint.py            #   state từng file + manifest
│   │   ├── downloader.py            #   vòng tải song song, khoá, dọn file dở, --status
│   │   ├── cli.py                   #   tham số dòng lệnh dùng chung cho 4 script tải
│   │   └── reader.py                #   đọc parquet/jsonl.gz về schema chung
│   ├── stbook/                      # nguồn sách stbook.vn
│   │   ├── crawler/                 #   crawl metadata + PDF (categories, client, content, parse_*, storage, main)
│   │   └── ocr_books.py             #   tìm sách, OCR có checkpoint theo cuốn, đọc kết quả về schema chung
│   ├── giao_trinh/                  # nguồn giáo trình (`extract.py` kiểm kê + trích text, `records.py` bản ghi + Parquet)
│   ├── vista/, vjol/                # thư mục trống giữ chỗ, chưa có code
│   └── __init__.py
└── tests/                           # test không cần mạng: uv run pytest
```

## 2. Cấu trúc thư mục dữ liệu

`--data-root` (mặc định biến `SEA_DATA_ROOT`, nếu không có thì `./data`). Mọi đường dẫn bên dưới tính từ đó; đường dẫn ghi trong state/kết quả đều là **tương đối**, nên chuyển cả thư mục sang ổ khác vẫn chạy tiếp được.

```
data/
├── raw/                                   # BẢN GỐC, không sửa
│   ├── sea_vi/                            #   nguồn SEA, giữ nguyên cấu trúc thư mục như trên HF
│   │   ├── sea_instruct_2602/Vietnamese/*.parquet          (12 file, ~2,7 GB)
│   │   ├── sea_pile_v2/vi/*.parquet                        (253 file, ~132 GB)
│   │   └── sea_lion_pile_v1/sea-pile-mc4/vi/*.jsonl.gz     (329 file, ~107 GB)
│   │       (mỗi bộ có thêm .cache/huggingface/ do thư viện HF tạo, đừng xoá khi đang tải)
│   ├── stbook/<danh-mục>/books.json, content/<product_id>.pdf   # đúng cấu trúc của crawler
│   ├── giao_trinh/<ngành>/<môn>/<file>    #   bản sao từ Google Drive bằng rclone (16 ngành)
│   └── VJOL/                              #   cấu trúc sẽ bổ sung khi có dữ liệu
├── interim/                               # kết quả trung gian của từng nguồn
│   ├── stbook_ocr/<danh-mục>/<product_id>.json     # text từng trang + metadata sách
│   └── giao_trinh_text/<sha256>.json               # text từng trang + method, khoá theo sha256 file
├── processed/                             # kết quả cuối của từng nguồn (theo schema chung)
│   └── giao_trinh/giao_trinh.parquet
├── state/                                 # checkpoint + khoá chống chạy trùng
│   ├── sea_instruct_2602/, sea_pile_v2/, sea_lion_pile_v1/   # <tên-file>.json + _manifest.json + .run.lock
│   ├── stbook_ocr/                        # .run.lock
│   └── ...                                # giao_trinh: do bước xử lý giáo trình tạo
└── logs/<việc>_<ngày_giờ>.log             # log từng lần chạy
```

Ghi chú: state của SEA nằm ở `state/<bộ>/` (không nằm trong `raw/sea_vi/`) và chỉ lưu đường dẫn tương đối trong repo HF, kích thước, sha256, nên di chuyển `raw/` không làm mất checkpoint.

## 3. Luồng từng nguồn

| Nguồn | Tải về | raw | Bước xử lý | interim | processed |
|---|---|---|---|---|---|
| SEA | `scripts/sea/download_all.py` (Hugging Face, có checkpoint) | `raw/sea_vi/<bộ>/` | `vi_corpus/sea/reader.py` đọc thẳng về schema chung (đã là text, bỏ bước parse) | (không cần) | kế hoạch: các stage chung |
| stbook | `scripts/stbook/crawl.py` | `raw/stbook/` | `scripts/stbook/ocr.py` (PaddleOCR + VietOCR, checkpoint theo cuốn) | `interim/stbook_ocr/` | `ocr_books.iter_records`: mỗi cuốn một bản ghi |
| giáo trình | `rclone copy` từ Google Drive (chạy tay, xem `docs/GIAO_TRINH.md`) | `raw/giao_trinh/` | `scripts/giao_trinh/extract.py`: kiểm kê sha256, trích text PDF/pptx, OCR trang scan (xuất Parquet tự động cuối lần chạy) | `interim/giao_trinh_text/<sha256>.json` | `processed/giao_trinh/giao_trinh.parquet` |
| VJOL, VISTA | không crawl trong repo này | `raw/VJOL/` (VISTA: đường dẫn tạm) | chưa có: chờ cấu trúc dữ liệu | | |

Sau khi mỗi nguồn ra bản ghi theo `vi_corpus/common/schema.py`, các stage chung (language, normalize, quality, dedup, knowledge unit, audit) xử lý giống nhau cho mọi nguồn; phần này đang ở mức thiết kế (`docs/PIPELINE.md`).

## 4. Vai trò từng gói

- **`vi_corpus/common/`**: không biết gì về một nguồn cụ thể. `registry.py` là nơi duy nhất khai báo đường dẫn raw của từng nguồn (`raw/sea_vi/...`, `raw/stbook`, `raw/VJOL`, ...) và thông tin owner/license/rights; `state.py` cung cấp cơ chế checkpoint/khoá/log mà mọi bước chạy lâu dùng chung.
- **`vi_corpus/sea/`**: tải phần tiếng Việt của 3 bộ SEA và đọc lại về schema chung. Đường dẫn raw dựng trong `downloader.DatasetPaths` (`raw/sea_vi/<bộ>`), state ở `state/<bộ>`.
- **`vi_corpus/stbook/`**: crawler lấy sách từ stbook.vn (chép từ repo `stbook-crawler`, mặc định ghi vào `raw/stbook`) và `ocr_books.py` OCR PDF dạng ảnh.
- **`vi_corpus/giao_trinh/`**: kiểm kê file theo sha256, trích text theo định dạng, làm sạch theo trang (`clean_pages` khi sinh bản ghi), sinh bản ghi schema chung (chưa phân loại sách/slide, ngôn ngữ, chất lượng). Đầu ra: `interim/giao_trinh_text/<sha256>.json` và `processed/giao_trinh/giao_trinh.parquet`.
- **`vi_corpus/vista/`, `vi_corpus/vjol/`**: giữ chỗ, chưa có code (YAGNI đến khi có cấu trúc dữ liệu).

## 5. `scripts/run_all.py` nối các phần

`run_all.py` chỉ có một bảng `PARTS` (phần → thư mục raw cần có, script) và chạy từng script bằng tiến trình con, truyền `--data-root`:

| Phần | Script | Điều kiện chạy |
|---|---|---|
| `sea` | `scripts/sea/download_all.py` | luôn chạy (là bước tải, tự tạo `raw/sea_vi`) |
| `stbook` | `scripts/stbook/ocr.py` | có `raw/stbook` |
| `giao_trinh` | `scripts/giao_trinh/extract.py` | có `raw/giao_trinh` |

```bash
uv run python scripts/run_all.py --data-root /duong/dan/data                    # cả 3 phần theo thứ tự trên
uv run python scripts/run_all.py --data-root /duong/dan/data --only sea,giao_trinh
uv run python scripts/run_all.py --data-root /duong/dan/data --status           # chuyển --status cho từng script
```

- Phần nào thiếu thư mục raw thì bị bỏ qua và in một dòng log nói rõ.
- Một phần lỗi không làm dừng các phần sau; cuối cùng in tóm tắt (`sea: ok | stbook: bỏ qua | giao_trinh: lỗi (mã 1)`) và thoát với mã khác 0 nếu có phần lỗi.
- Muốn chạy nền trên Run:ai: `./scripts/run_with_gpu_keepalive.sh scripts/run_all.py --data-root /duong/dan/data`.
- Tham số riêng của từng script (ví dụ `--device cpu` của OCR, `--workers` của SEA) không đi qua `run_all`; cần thì chạy trực tiếp script của phần đó.
- Thêm nguồn mới: tạo `vi_corpus/<nguồn>/`, `scripts/<nguồn>/`, thêm một dòng vào `PARTS` và một `SourceSpec` vào `registry.py`.
