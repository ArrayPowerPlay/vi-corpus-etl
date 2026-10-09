# vi-corpus-etl

ETL xây **corpus tiếng Việt cho LLM** từ nhiều nguồn, có truy vết nguồn gốc (provenance) cho từng bản ghi.

| Nguồn | Nội dung | Trạng thái |
|---|---|---|
| SEA (3 bộ của AI Singapore) | Văn bản web + hội thoại instruct, phần tiếng Việt | Đã có bước **tải** (mục 1–7 bên dưới); bước xử lý: kế hoạch |
| stbook.vn | Sách NXB Chính trị quốc gia Sự thật (PDF dạng ảnh, tải bằng repo `stbook-crawler`) | Đã có bước **OCR → text** (Phần B) |
| Giáo trình (Google Drive) | Giáo trình đại học 16 ngành (PDF, pptx), tải bằng rclone | Đang viết bước xử lý (`docs/GIAO_TRINH.md`) |
| VJOL, VISTA | Bài báo khoa học (PDF), **chỉ xử lý** dữ liệu có sẵn, không crawl | Kế hoạch: `raw/VJOL/`, cấu trúc sẽ bổ sung khi có dữ liệu |

Chiến lược xử lý (nguồn → parse → ngôn ngữ → làm sạch → chất lượng → loại trùng → knowledge unit → audit):
xem [`docs/PIPELINE.md`](PIPELINE.md), [`docs/SOURCES.md`](SOURCES.md) và lộ trình [`docs/ROADMAP.md`](ROADMAP.md).
Các quyết định đã chốt qua từng phiên làm việc: [`docs/DECISION_LOG.md`](DECISION_LOG.md); sơ đồ luồng chính và luồng con: [`docs/diagrams/`](diagrams/).
Cấu trúc repo và thư mục dữ liệu: [`docs/PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md).
Phần còn lại của README mô tả phần đã có code: **Phần A** tải dữ liệu SEA, **Phần B** OCR sách stbook, **Phần C** chạy tất cả bằng `scripts/run_all.py`, **Phần D** pipeline xử lý end-to-end trên N mẫu (có visualize, chạy nhiều GPU), **Phần E** công cụ hệ thống đọc .doc / .ppt / .djvu, **Phần F** so sánh engine OCR.

---

# Phần A — Tải dữ liệu SEA

Tải **phần tiếng Việt** của 3 bộ dữ liệu do AI Singapore công bố trên Hugging Face, giữ nguyên file gốc.
Có **checkpoint**: nếu bị crash, mất mạng hoặc máy khởi động lại, chỉ cần chạy lại đúng lệnh cũ là tải tiếp.

| Bộ dữ liệu | Repo trên Hugging Face | Thư mục tiếng Việt | Số file | Dung lượng | Định dạng |
|---|---|---|---|---|---|
| SEA-Instruct-2602 | [`aisingapore/SEA-Instruct-2602`](https://huggingface.co/datasets/aisingapore/SEA-Instruct-2602) (bị khoá, cần xin quyền) | `Vietnamese/` | 12 | ~2,7 GB | parquet |
| SEA-PILE-v2 | [`aisingapore/SEA-PILE-v2`](https://huggingface.co/datasets/aisingapore/SEA-PILE-v2) | `vi/` | 253 | ~132 GB | parquet |
| SEA-LION-Pile v1 | [`aisingapore/SEA-PILE-v1`](https://huggingface.co/datasets/aisingapore/SEA-PILE-v1) | `sea-pile-mc4/vi/` | 329 | ~107 GB | jsonl.gz |
| **Tổng** | | | **594** | **~242 GB** | |

> Bước tải **chỉ tải về, chưa lọc**. Thư mục tiếng Việt đã được tác giả chia sẵn theo ngôn ngữ.
> Nếu sau này cần lọc thì bước lọc sẽ đọc từ `data/raw/sea_vi/` và ghi ra một thư mục riêng, không phải tải lại.

---

## 1. Chuẩn bị (làm một lần)

### 1.1. Yêu cầu
- Linux (máy chủ Jupyter Lab) và [uv](https://docs.astral.sh/uv/) để cài thư viện. uv tự lo Python ≥ 3.10 nếu máy chưa có.
  Nếu máy chủ chưa có uv:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh     # hoặc: pip install uv
  ```
- Ổ đĩa trống **≥ 260 GB** cho cả 3 bộ (242 GB dữ liệu + chỗ cho file đang tải dở).
- Tài khoản Hugging Face **đã được duyệt quyền** vào `aisingapore/SEA-Instruct-2602`:
  mở trang dataset, bấm đồng ý điều khoản.

### 1.2. Lấy code về máy chủ Jupyter
Mở **Terminal** trong Jupyter Lab (File → New → Terminal):
```bash
git clone https://github.com/ArrayPowerPlay/vi-corpus-etl.git
cd vi-corpus-etl
uv sync --no-dev     # tạo .venv và cài đúng phiên bản thư viện ghi trong uv.lock
```
- `uv sync --no-dev` chỉ cài thư viện cần để tải. Bỏ `--no-dev` nếu muốn chạy test.
- Mọi lệnh bên dưới đều chạy qua `uv run ...`, nên không cần tự kích hoạt `.venv`.

### 1.3. Khai báo token Hugging Face
```bash
cp .env.example .env
nano .env          # hoặc mở .env bằng trình soạn thảo của Jupyter
```
Nội dung file `.env`:
```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```
- `.env` đã nằm trong `.gitignore` nên **không bị đẩy lên git**.
- Nếu biến môi trường `HF_TOKEN` đã được đặt sẵn (ví dụ `export HF_TOKEN=...`) thì biến môi trường được ưu tiên hơn file `.env`.
- Tạo token tại https://huggingface.co/settings/tokens (quyền **Read** là đủ).

---

## 2. Chạy

### 2.1. Các script

| Script | Tải bộ nào |
|---|---|
| `scripts/sea/download_sea_instruct_2602.py` | SEA-Instruct-2602 |
| `scripts/sea/download_sea_pile_v2.py` | SEA-PILE-v2 |
| `scripts/sea/download_sea_lion_pile_v1.py` | SEA-LION-Pile v1 |
| `scripts/sea/download_all.py` | Cả 3 bộ, lần lượt: Instruct → v2 → v1 (bộ nhỏ trước) |

Cả 4 script dùng **chung một bộ tham số**.

### 2.2. Tham số

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--data-root ĐƯỜNG_DẪN` | biến `SEA_DATA_ROOT`, nếu không có thì `./data` | Thư mục lưu dữ liệu. **Nên trỏ tới ổ đĩa lớn nhất.** |
| `--workers N` | `8` | Số file tải cùng lúc. Mạng mạnh có thể tăng lên 12–16. Nếu hay bị lỗi 429 (HF chặn vì gọi quá nhiều) thì giảm xuống 4. |
| `--limit-files N` | không giới hạn | Chỉ tải N file đầu tiên, để **chạy thử**. |
| `--verify-sha256` | tắt | Sau khi tải, tính mã sha256 và so với HF để chắc chắn file không hỏng. Tốn thêm CPU/đọc đĩa, khoảng vài giây mỗi file. |
| `--max-retries N` | `5` | Số lần thử lại mỗi file khi lỗi mạng. Thời gian chờ tăng dần: 10s, 20s, 40s… tối đa 5 phút. |
| `--status` | tắt | Chỉ **xem tiến độ** rồi thoát, không tải gì. Không cần mạng, chạy được khi đang có tiến trình tải chạy nền. |
| `-h`, `--help` | | Xem hướng dẫn tham số. |

### 2.3. Chạy thử trước (khuyên làm)
```bash
uv run python scripts/sea/download_sea_instruct_2602.py --data-root /duong/dan/data --limit-files 1 --verify-sha256
uv run python scripts/sea/download_sea_instruct_2602.py --data-root /duong/dan/data --status
```
Nếu thấy `Xong 1/1 file` là token và mạng đều ổn.

### 2.4. Chạy thật (chạy nền, không sợ đóng trình duyệt)
**Chạy trong Terminal của Jupyter, không chạy trong ô notebook.** Nếu chạy trong notebook, kernel chết hoặc bấm Restart sẽ làm dừng việc tải.

```bash
cd vi-corpus-etl
nohup uv run python scripts/sea/download_all.py --data-root /duong/dan/data --workers 8 --verify-sha256 \
      > download_all.out 2>&1 &
```
- Sau lệnh này có thể đóng tab, tắt máy tính cá nhân; máy chủ vẫn tiếp tục tải.
- Muốn chỉ tải một bộ thì thay `download_all.py` bằng script của bộ đó.

Cách khác là dùng `tmux`, nếu máy chủ có sẵn:
```bash
tmux new -s sea
uv run python scripts/sea/download_all.py --data-root /duong/dan/data --workers 8
# Ctrl+B rồi D để thoát ra, tiến trình vẫn chạy. Quay lại: tmux attach -t sea
```

Nếu buộc phải chạy từ notebook, dùng ô sau. Nó khởi động tiến trình **tách rời** khỏi kernel:
```python
import subprocess
subprocess.Popen(
    "nohup uv run python scripts/sea/download_all.py --data-root /duong/dan/data --workers 8 > download_all.out 2>&1 &",
    shell=True, cwd="/duong/dan/vi-corpus-etl",
)
```

### 2.5. Theo dõi
```bash
uv run python scripts/sea/download_all.py --data-root /duong/dan/data --status   # tiến độ từng bộ
tail -f download_all.out                                              # log trực tiếp (Ctrl+C để thoát xem)
ls /duong/dan/data/logs/                                              # log chi tiết từng lần chạy
```
Ví dụ kết quả `--status`:
```
[sea_pile_v2] SEA-PILE-v2 (văn bản web, ~132 GB, 253 file parquet)
  Revision : 77573cc846
  Xong     : 120/253 file | 62.4/132.3 GB (47.2%)
  Lỗi      : 0 file
```
> `--status` báo theo danh sách file của **lần chạy gần nhất**. Nếu lần gần nhất là chạy thử với `--limit-files 1`
> thì nó sẽ báo `1/1`. Chạy thật một lần là con số được cập nhật đủ.

### 2.6. Dừng và chạy tiếp
- Dừng: `Ctrl+C` nếu đang chạy trực tiếp. Nếu đang chạy nền:
  ```bash
  pkill -f download_all.py
  ```
- Chạy tiếp: gõ lại **đúng lệnh cũ**. Chương trình sẽ:
  1. bỏ qua các file đã tải xong và đã kiểm tra;
  2. dọn các file tải dở còn sót từ lần trước;
  3. tải lại từ đầu những file đang tải dở lúc bị dừng;
  4. thử lại các file bị lỗi ở lần trước.

---

## 3. Checkpoint hoạt động thế nào

- **Đơn vị checkpoint là từng file** trên Hugging Face, mỗi file khoảng 200–500 MB. Tải xong một file thì kiểm tra kích thước (và sha256 nếu bật `--verify-sha256`), rồi ghi ngay một file đánh dấu vào `data/state/<bộ>/`.
- File đánh dấu được ghi theo kiểu an toàn: ghi ra file tạm rồi đổi tên. Vì vậy dù máy tắt đột ngột giữa lúc ghi, cũng không bao giờ có file đánh dấu bị hỏng.
- Khi crash, phần mất đi chỉ là **các file đang tải dở**, tối đa bằng `--workers` file (8 × ~500 MB ≈ 4 GB). Thư viện `huggingface_hub` không hỗ trợ tải tiếp một file dở, nên những file đó được tải lại từ đầu.
- **Chống chạy trùng:** nếu lỡ chạy hai lệnh cho cùng một bộ, lệnh thứ hai báo
  `Đang có một tiến trình khác tải bộ này` rồi thoát. Khoá tự nhả khi tiến trình kết thúc hoặc crash.
- **Cố định phiên bản:** mỗi lần chạy dùng một mã commit cố định của repo HF. Nếu tác giả cập nhật một file (đổi kích thước hoặc sha256), file đó được tự động tải lại ở lần chạy sau.

---

## 4. Dữ liệu được lưu ở đâu

```
<data-root>/
├── raw/sea_vi/                            # BẢN GỐC, giữ nguyên file và cấu trúc thư mục như trên HF
│   ├── sea_instruct_2602/Vietnamese/train-000xx-of-00012.parquet
│   ├── sea_pile_v2/vi/train-00xxx-of-00253.parquet
│   └── sea_lion_pile_v1/sea-pile-mc4/vi/mc4-vi-00xxx-00328.jsonl.gz
├── state/<bộ>/                            # checkpoint
│   ├── _manifest.json                     #   danh sách file cần tải (dùng cho --status)
│   └── <tên-file>.json                    #   trạng thái từng file: done / failed + kích thước, sha256, thời gian
└── logs/<bộ>_<ngày_giờ>.log               # log từng lần chạy
```
Mỗi `raw/sea_vi/<bộ>/` còn có thư mục ẩn `.cache/huggingface/` do thư viện HF tạo ra. Thư mục này nhỏ, **đừng xoá khi đang tải**.

### Đã tải theo cấu trúc cũ (`raw/<bộ>/`)? Chuyển sang cấu trúc mới, không phải tải lại
Trước đây dữ liệu nằm ở `raw/<bộ>/`. Trên máy chủ, chạy đúng các lệnh sau (thay đường dẫn cho đúng), **không cần di chuyển `state/`**:
```bash
cd /duong/dan/data/raw
mkdir -p sea_vi
mv sea_instruct_2602 sea_pile_v2 sea_lion_pile_v1 sea_vi/
```
`state/<bộ>/` chỉ lưu đường dẫn tương đối trong repo HF, kích thước và sha256, nên chạy lại lệnh tải cũ (hoặc `--status`) sẽ thấy `Xong N/N file` và không tải lại gì.
Chỉ chuyển các thư mục SEA; `raw/stbook/` giữ nguyên chỗ cũ.

### Cột dữ liệu
- **SEA-PILE-v2** (parquet): `text, dump, timestamp, url, warc-record-id`
- **SEA-LION-Pile v1** (jsonl.gz, mỗi dòng một JSON): `id, text`
- **SEA-Instruct-2602** (parquet): 23 cột metadata (`prompt_primary_language`, `prompt_primary_domain`, `prompt_complexity`, …) và `conversations`.
  Chú ý: `conversations` là **chuỗi** dạng Python (nháy đơn, `None`), phải đọc bằng `ast.literal_eval`, **không** dùng `json.loads`.

### Đọc dữ liệu đã tải
```python
from datasets import load_dataset   # cần thêm thư viện: uv add datasets

pile_v2 = load_dataset("parquet", data_files="/duong/dan/data/raw/sea_vi/sea_pile_v2/vi/*.parquet",
                       split="train", streaming=True)
pile_v1 = load_dataset("json", data_files="/duong/dan/data/raw/sea_vi/sea_lion_pile_v1/sea-pile-mc4/vi/*.jsonl.gz",
                       split="train", streaming=True)
instruct = load_dataset("parquet", data_files="/duong/dan/data/raw/sea_vi/sea_instruct_2602/Vietnamese/*.parquet",
                        split="train")

import ast
messages = ast.literal_eval(instruct[0]["conversations"])   # list[{"role": ..., "content": ...}]
```

---

## 5. Thời gian ước tính

Khi thử nghiệm, tốc độ tải từ Hugging Face là 16–34 MB/s. Thời gian thực tế phụ thuộc vào mạng của máy chủ:

| Tốc độ mạng | Cả 3 bộ (~242 GB) |
|---|---|
| 20 MB/s | ~3,4 giờ |
| 50 MB/s | ~1,3 giờ |
| 100 MB/s | ~40 phút |

---

## 6. Xử lý sự cố

| Hiện tượng | Nguyên nhân / cách xử lý |
|---|---|
| `Không tìm thấy HF_TOKEN` | Chưa tạo file `.env`, hoặc file `.env` không nằm ở thư mục gốc repo. Xem mục 1.3. |
| `GatedRepoError` / `401` / `403` với SEA-Instruct | Tài khoản chưa được duyệt quyền vào dataset, hoặc token sai hay hết hạn. |
| Nhiều cảnh báo `thử lại sau …s`, lỗi `429` | HF đang giới hạn tốc độ. Giảm `--workers` (ví dụ 4). |
| `Đang có một tiến trình khác tải bộ này` | Đã có một lệnh đang chạy. Xem bằng `ps aux | grep download_`, hoặc chờ nó xong. |
| `--status` báo có file lỗi | Chạy lại đúng lệnh cũ để thử lại. Nếu vẫn lỗi, xem chi tiết trong `logs/`. |
| Hết ổ đĩa | Chuyển sang ổ lớn hơn bằng `--data-root`. Có thể chép sẵn thư mục `data/` cũ sang ổ mới, checkpoint vẫn dùng tiếp được. |

---

# Phần B — OCR sách stbook.vn

Sách do repo [`stbook_crawler`](https://github.com/ArrayPowerPlay/stbook_crawler) tải về là **PDF dạng ảnh** (mỗi trang là một ảnh JPEG, không có chữ để copy).
Muốn đưa vào corpus thì phải **OCR** (nhận dạng chữ trong ảnh):

1. **PaddleOCR** tìm vị trí các dòng chữ trên trang.
2. **VietOCR** đọc chữ trong từng dòng. (Bộ đọc chữ của PaddleOCR làm mất dấu tiếng Việt nên không dùng.)

Kết quả thử trên sách thật: gần như đúng hoàn toàn, thỉnh thoảng sai chữ hoa/thường hoặc dấu câu.

## B1. Chuẩn bị

```bash
uv sync --group ocr     # cài thêm PaddleOCR, VietOCR, torch, PyMuPDF (vài GB, lần đầu hơi lâu)
```
Chép (hoặc tạo symlink) **nguyên thư mục `data/` của stbook-crawler** vào `<data-root>/raw/stbook/`, không sửa gì bên trong:
```bash
ln -s /duong/dan/stbook_crawler/data /duong/dan/data/raw/stbook
```
Hoặc để nguyên chỗ cũ và truyền `--stbook-root /duong/dan/stbook_crawler/data`.
Có thể tải sách ngay trong repo này: `uv run python scripts/stbook/crawl.py --download-pdf` (ghi vào `<data-root>/raw/stbook`).

Cấu trúc stbook-crawler tạo ra (chỉ đọc, không sửa):
```
raw/stbook/
├── <danh-mục>/books.json                 # metadata sách: tên, tác giả, năm XB, số trang, ...
├── <danh-mục>/content/<product_id>.pdf   # PDF sách miễn phí
├── <danh-mục>/content/<product_id>_pages/  # sách đang tải dở → bỏ qua
└── crawl.log
```

## B2. Chạy

```bash
# chạy thử 1 cuốn (máy không có GPU thì thêm --device cpu)
uv run python scripts/stbook/ocr.py --data-root /duong/dan/data --limit-books 1

# chạy thật, chạy nền trong Terminal của Jupyter
nohup uv run python scripts/stbook/ocr.py --data-root /duong/dan/data > ocr_stbook.out 2>&1 &

# xem tiến độ
uv run python scripts/stbook/ocr.py --data-root /duong/dan/data --status
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--data-root` | biến `SEA_DATA_ROOT`, nếu không có thì `./data` | Thư mục gốc dữ liệu (dùng chung với Phần A). |
| `--stbook-root` | `<data-root>/raw/stbook` | Thư mục `data/` của stbook-crawler. |
| `--device` | `cuda` | Chạy VietOCR trên GPU (`cuda`) hay CPU (`cpu`). |
| `--limit-books N` | không giới hạn | Chỉ OCR N cuốn, để chạy thử. |
| `--status` | tắt | Chỉ in số cuốn đã OCR xong / tổng số cuốn có PDF. |

- **Checkpoint theo cuốn**: xong cuốn nào ghi file kết quả cuốn đó ngay. Bị ngắt (kể cả `kill -9`) thì chạy lại đúng lệnh cũ,
  chỉ mất cuốn đang làm dở. Nếu PDF được tải lại (kích thước đổi) thì cuốn đó được OCR lại.
- **PDF hỏng** (stbook-crawler bị kill đúng lúc đang ghi PDF) được phát hiện và bỏ qua, log ghi
  `PDF hỏng, cần xoá và tải lại bằng stbook-crawler`. Xoá file PDF đó, chạy lại stbook-crawler rồi chạy lại lệnh OCR.
- Stbook-crawler vẫn đang tải thì vẫn OCR được: lần chạy sau sẽ làm tiếp các cuốn mới tải xong.
- Chống chạy trùng giống Phần A (khoá trong `state/stbook_ocr/`), log trong `logs/stbook_ocr_<ngày_giờ>.log`.
- Tốc độ: khi thử trên CPU là ~3,5 giây/trang. Trên GPU nhanh hơn nhiều. Bước tìm dòng chữ (PaddleOCR) mặc định chạy CPU;
  muốn nó chạy GPU thì cài thêm `paddlepaddle-gpu` theo [hướng dẫn của Paddle](https://www.paddlepaddle.org.cn/install/quick).
- Lần chạy đầu tự tải trọng số mô hình (~600 MB) về `~/.paddlex/` và `/tmp/`.

## B3. Kết quả

```
<data-root>/interim/stbook_ocr/<danh-mục>/<product_id>.json
```
Mỗi file gồm: `book` (metadata gốc trong `books.json`), `pdf_path`, `pdf_size`, `ocr` (tên mô hình), `created_at`,
và `pages` (danh sách text từng trang, giữ nguyên xuống dòng như trên trang sách).

Để đưa vào pipeline, `vi_corpus.stbook.ocr_books.iter_records` đọc các file này và sinh **mỗi cuốn một bản ghi** theo schema chung
(`source_key = "stbook"`, `source_path` trỏ về PDF gốc). Lưu ý:
- Text là **kết quả OCR thô**: còn số trang, chú thích cuối trang, trang bìa/trang ban biên tập. Làm sạch ở bước normalize/quality sau.
- Có vài sách tiếng Anh (vd bản dịch Cương lĩnh), bước nhận diện ngôn ngữ sẽ lọc.
- Sách có bản quyền của NXB (chỉ được đọc miễn phí online), nên `rights_status = "unknown"` → sẽ bị quarantine cho tới khi xác nhận quyền.

---

# Phần C — Chạy tất cả bằng `scripts/run_all.py`

Mỗi nguồn có một thư mục script riêng (`scripts/sea/`, `scripts/stbook/`, `scripts/giao_trinh/`); `run_all.py` chạy lần lượt script của từng phần
(SEA tải → stbook OCR → giáo trình trích text), truyền `--data-root` cho từng script:

```bash
uv run python scripts/run_all.py --data-root /duong/dan/data                     # cả 3 phần
uv run python scripts/run_all.py --data-root /duong/dan/data --only sea,giao_trinh   # chọn phần
uv run python scripts/run_all.py --data-root /duong/dan/data --status            # xem tiến độ từng phần
```
- Phần xử lý (`stbook`, `giao_trinh`) thiếu thư mục `raw/...` thì bị bỏ qua, có dòng log nói rõ. `sea` là bước tải nên luôn chạy.
- Một phần lỗi không làm dừng các phần sau; cuối cùng in tóm tắt và thoát với mã khác 0 nếu có phần lỗi.
- Tham số riêng của từng script (`--workers`, `--device cpu`, ...) không đi qua `run_all`; muốn dùng thì chạy trực tiếp script của phần đó.
- Giáo trình tải bằng `rclone` (xem `docs/GIAO_TRINH.md`), không nằm trong `run_all`.

---

# Phần D — Pipeline xử lý end-to-end (thử trên N mẫu, có visualize, chạy nhiều GPU)

Lấy mẫu **N bản ghi** từ `raw/` (SEA + stbook) rồi cho chạy qua toàn bộ pipeline, cuối cùng ra bộ sạch, knowledge unit, audit và **report.html** có bản đồ embedding.
Mục đích: kiểm tra pipeline và chất lượng dữ liệu trước khi chạy toàn bộ. Thiết kế từng stage: [`docs/PIPELINE.md`](PIPELINE.md).

| # | Stage (`--until`) | Làm gì | File kết quả trong thư mục run |
|---|---|---|---|
| 1 | `ingest` | Lấy mẫu từ `raw/` theo `--total` và `--mix` (stbook: nguyên cuốn đã OCR) | `01_ingest.parquet` |
| 2 | `prepare` | Chuẩn hóa Unicode, xoá dòng lặp, cắt đoạn theo token (~1.024, tối đa 2.048), đếm token bằng tokenizer Qwen3 | `02_prepare.parquet` |
| 3 | `language` | Nhận diện ngôn ngữ bằng fastText `lid.176` theo từng đoạn văn, ghi `lang_mix` — **song song CPU** | `03_language.parquet` |
| 4 | `quality` | Số đo, điểm 0–100, band A/B/C/D, reason code — **song song CPU** | `04_quality.parquet` |
| 5 | `dedup` | Trùng theo văn bản: chính xác + gần trùng (MinHash) + văn bản nằm trong văn bản lớn hơn, rồi trùng chính xác theo đoạn; cổng quyền chỉ gắn nhãn. Ghi kho dấu vân tay cho dedup vòng 2 | `05_dedup.parquet`, `<data-root>/state/dedup_index/<nguồn>/<run>.parquet` |
| 6 | `embed` | Vector hóa văn bản — **song song nhiều GPU** | `06_embeddings.parquet` |
| 7 | `reduce` | Gom cụm HDBSCAN trên embedding gốc; PCA + UMAP xuống 2D chỉ để vẽ (cuML nếu có, không thì scikit-learn/umap-learn) | `07_reduced.parquet` |
| 8 | `finalize` | Bộ sạch, knowledge unit, khối CPT, quét nhiễm benchmark, audit, bản đồ plotly, báo cáo | `clean.parquet`, `knowledge_units.parquet`, `cpt_blocks.parquet`, `audit.json`, `report.html` (đã nhúng bản đồ) |

Mọi kết quả nằm ở `<data-root>/processed/pipeline_runs/<run-name>/` (mặc định `run_<total>_seed<seed>`). Mỗi stage có checkpoint kèm **vân tay** (tham số của stage + phiên bản code + dữ liệu đầu vào): chạy lại thì stage nào đã có file và vân tay khớp sẽ được bỏ qua; đổi tham số (tokenizer, ngưỡng, bộ nhúng...) hoặc dữ liệu `raw/` thì stage đó và các stage sau tự chạy lại (log ghi rõ). Muốn cố ý dùng lại kết quả cũ: thêm `--keep-stale`.

## D1. Chuẩn bị (làm một lần)

```bash
uv sync                                 # thư viện cơ bản
uv sync --group viz                     # bản đồ embedding: pandas, plotly, scikit-learn, umap-learn
uv sync --group curator                 # chạy song song: NeMo Curator + Ray + transformers + torch
# Server GPU: nên dùng bản CUDA của Curator (giống ViLA)
uv pip install "nemo-curator[text_cuda12]>=0.7"
# Muốn OCR sách còn chưa OCR:  uv sync --group ocr --group curator --group viz
```
Cần có sẵn `data/raw/sea_vi/...` (SEA đã tải, Phần A) và sách stbook đã OCR ở `data/interim/stbook_ocr/` (Phần B, hoặc `--ocr-books`). `--data-root` mặc định lấy từ biến môi trường `SEA_DATA_ROOT`, nếu không có thì `./data`.
Lần chạy đầu tải tokenizer Qwen3 (từ Hugging Face) và mô hình ngôn ngữ fastText `lid.176.bin` (~126 MB, vào `VI_CORPUS_MODEL_DIR`, mặc định `~/.cache/vi_corpus`); các lần sau dùng lại.
Không có GPU/mạng (thử trên máy nhỏ): dùng `--embedder tfidf --tokenizer words --lang-model heuristic` (CPU, không tải gì).

## D2. Chạy thử nhanh (khuyên làm trước)

```bash
uv run python scripts/run_pipeline.py --data-root /duong/dan/data --total 500 --run-name thu \
    --embedder tfidf                    # 500 mẫu, chạy tuần tự, không cần GPU
```
Xong thì mở `.../pipeline_runs/thu/report.html`.

## D3. Chạy toàn bộ

Tham số chọn số mẫu: **`--total`** (tổng số mẫu của mọi nguồn cộng lại; mặc định 10000), **`--mix`** (tỉ lệ giữa các nguồn, mặc định `sea_pile_v2=30,sea_lion_pile_v1=30,sea_instruct_2602=15,stbook=25`), `--seed` (đổi seed ra mẫu khác).

```bash
# 10.000 mẫu, tuần tự trong một tiến trình
uv run python scripts/run_pipeline.py --data-root /duong/dan/data --total 10000

# 2.000 mẫu, chỉ SEA-PILE-v2 và stbook, tỉ lệ 60/40
uv run python scripts/run_pipeline.py --data-root /duong/dan/data --total 2000 --mix sea_pile_v2=60,stbook=40

# 10.000 mẫu, song song trên 4 GPU A100 (Ray cục bộ, executor Xenna như ViLA)
uv run python scripts/run_pipeline.py --data-root /duong/dan/data --total 10000 --executor xenna --num-gpus 4
```
Với stbook, `--total` tính theo số **đoạn** (sách được cắt đoạn ~1.024 token), không phải số cuốn. Bài web dài hơn 2.048 token cũng bị cắt, nhưng `--total` của nguồn web vẫn tính theo số bài.
Chạy lâu thì chạy nền: `tmux new -s pipe` rồi chạy lệnh trong đó (Ctrl+B rồi D để thoát ra, `tmux attach -t pipe` để quay lại).

## D4. Chạy từng bước

Mỗi lệnh dừng sau stage chọn bằng `--until`; lệnh sau (cùng `--data-root`, `--total`, `--mix`, `--seed`) làm tiếp từ checkpoint:

```bash
R="--data-root /duong/dan/data --total 10000"
uv run python scripts/run_pipeline.py $R --until ingest       # 1. lấy mẫu
uv run python scripts/run_pipeline.py $R --until prepare      # 2. chuẩn hóa + cắt đoạn
uv run python scripts/run_pipeline.py $R --until language     # 3. ngôn ngữ
uv run python scripts/run_pipeline.py $R --until quality      # 4. chất lượng
uv run python scripts/run_pipeline.py $R --until dedup        # 5. loại trùng
uv run python scripts/run_pipeline.py $R --until embed        # 6. embedding (GPU)
uv run python scripts/run_pipeline.py $R --until reduce       # 7. giảm chiều + gom cụm
uv run python scripts/run_pipeline.py $R                      # 8. finalize: bộ sạch + bản đồ + report
```
Đổi tham số thì không cần làm gì thêm (vân tay tự phát hiện). Sửa **code** của một stage mà chưa tăng hằng số phiên bản của nó, hoặc run cũ chưa có vân tay: thêm `--force-from quality` (xoá kết quả từ `quality` trở đi, giữ các stage trước).

Các tham số mới (đợt code 2026-10-08, chi tiết `docs/PIPELINE.md` mục 6):

| Tham số | Ý nghĩa |
|---|---|
| `--tokenizer hf:<repo>\|tiktoken:<enc>\|words` | Bộ đếm token (mặc định `hf:Qwen/Qwen3-0.6B`); `--compare-tokenizers a,b` ghi thêm tổng token theo tokenizer khác vào audit |
| `--lang-model fasttext:lid.176\|heuristic` | Bộ nhận diện ngôn ngữ (mặc định fastText) |
| `--embed-scope all\|kept` | Nhúng mọi bản ghi (để vẽ cả bản bị loại) hay chỉ bản giữ lại |
| `--cluster-space raw\|pca50\|umap10`, `--min-cluster-size N`, `--min-samples N` | Không gian và tham số HDBSCAN (mặc định embedding gốc, `max(2, min(20, n/10))`) |
| `--cpt-context N` | Độ dài tối đa (token) của khối CPT trong `cpt_blocks.parquet` (mặc định 4.096) |
| `--no-dedup-index` | Không ghi kho dấu vân tay dedup vòng 1 |
| `--global-verdict <verdict.parquet>` | Áp kết quả dedup vòng 2 trước khi dựng knowledge unit |
| `--keep-stale` | Dùng lại kết quả cũ dù vân tay lệch |

### Dedup vòng 2 (giữa các nguồn / các lần chạy)

Chạy từng nguồn xong (mỗi lần chạy ghi kho `state/dedup_index/`), gộp trùng toàn cục rồi áp lại:

```bash
uv run python scripts/dedup_global.py --data-root /duong/dan/data            # -> processed/dedup_global/verdict.parquet + verdict_stats.json
uv run python scripts/run_pipeline.py $R --global-verdict /duong/dan/data/processed/dedup_global/verdict.parquet
```
Thêm nguồn mới: chạy pipeline cho nguồn đó, rồi chạy lại `dedup_global.py` (chỉ đọc kho dấu vân tay, không đọc lại text).

Cờ của `dedup_global.py`: `--threshold` (mặc định = ngưỡng fuzzy của `RunConfig`, 0,8), `--priority stbook=0,sea_pile_v2=1` (đổi thứ tự giữ bản, mặc định R-15), `--rights-gate enforce` (văn bản quyền chưa rõ bị đánh `rights` → `rejected:rights` khi áp; mặc định `tag` chỉ gắn nhãn). Đổi ưu tiên hay rights gate chỉ cần chạy lại vòng 2 rồi áp lại `--global-verdict`. Mỗi văn bản lấy dòng kho ghi mới nhất; văn bản mà lần chạy gần nhất đã loại (bia mộ) không vào vòng 2 và có status `dropped` trong verdict. Chỉ nên áp `--global-verdict` cho lần chạy mới nhất của mỗi nguồn (lần chạy cũ hơn còn giữ văn bản `dropped` thì manifest đếm `superseded`).

### Lấy mẫu cho LLM chấm điểm

```bash
uv run python scripts/make_judge_sample.py --run-dir /duong/dan/data/processed/pipeline_runs/run_100000_seed42 \
    --n 50000 --out /duong/dan/data/processed/judge/sample_50k.parquet
```
Đọc `04_quality.parquet` (chạy tới `--until quality` là đủ), lấy ~10% hỏi-đáp SEA-Instruct, ~70% mẫu qua rule / ~30% bị rule loại, chia đều nguồn × khoảng độ dài. Script chấm bằng LLM chưa có (thang điểm chưa duyệt).

## D5. Chạy song song nhiều GPU

Dùng **NeMo Curator + Ray** như ViLA: mỗi stage nặng là một `ProcessingStage` khai báo tài nguyên, executor tạo nhiều actor và chia các phân vùng (mặc định 500 bản ghi) cho chúng.
Ở stage `embed` mỗi actor giữ một bản mô hình trên **một GPU riêng** (4 GPU = 4 actor chạy cùng lúc); `language`, `quality` chạy nhiều actor CPU. `dedup` và `reduce` cần nhìn toàn bộ corpus nên chạy trong tiến trình chính.

| Tham số | Ý nghĩa |
|---|---|
| `--executor xenna\|ray_actor_pool\|ray_data` | Bật chạy song song bằng executor này (không có = tuần tự). `xenna` là mặc định của ViLA |
| `--num-gpus N`, `--num-cpus N` | Số GPU / CPU Ray được dùng (mặc định tự phát hiện) |
| `--ray-address auto` | Nối vào cụm Ray có sẵn thay vì chạy Ray cục bộ |
| `--rows-per-task N` | Số bản ghi mỗi phân vùng (nhỏ = cân tải tốt hơn, nhiều overhead hơn) |
| `--gpus-per-worker 0.5` | Hai actor chia một GPU (mô hình nhỏ); mặc định 1.0 = một actor / GPU |
| `--embedder hf:<model>` | Mô hình nhúng HF, mặc định `hf:intfloat/multilingual-e5-base`; `--embed-prompt` đặt tiền tố (E5 cần `passage: `) |
| `--embed-batch-size N` | Số văn bản mỗi lần chạy mô hình (giảm nếu hết VRAM) |

OCR sách chưa OCR cũng chạy được nhiều GPU (mỗi GPU một cuốn): thêm `--ocr-books N --executor xenna --num-gpus 4` (cần `--group ocr`).
Kiểm tra GPU đang chạy bằng `nvidia-smi` ở terminal khác. Kết quả song song **giống hệt** chạy tuần tự (đã kiểm bằng test).

## D6. Xem kết quả

- `report.html`: số liệu từng stage, phân bố band/ngôn ngữ/điểm, lý do loại, mẫu văn bản, và mục **Bản đồ embedding** (chọn cách tô màu bằng các nút).
- Bản đồ embedding nằm **ngay trong `report.html`** (plotly.js và dữ liệu nhúng sẵn, ~vài MB, không cần mạng, không iframe): chỉ cần tải mỗi file `report.html` về rồi mở bằng trình duyệt. Bấm nút `<umap|pca>-<nguồn|trạng thái|band|ngôn ngữ|cụm>` để đổi cách vẽ / tô màu; rê chuột vào điểm để đọc đoạn đầu văn bản và reason code. Cách đọc: tô theo **nguồn** để xem các nguồn tách nhau thế nào; tô theo **trạng thái** để xem mẫu bị loại (chất lượng / trùng / quyền) nằm ở vùng nào, mẫu rác thường dồn thành cụm riêng.
- `audit.json`: tỉ lệ lineage, số trùng, số bị quarantine, tokenizer, token/từ, kết quả quét nhiễm benchmark (đặt file `*.txt` / `*.jsonl` vào `configs/benchmarks/`; hiện rỗng). `manifest.json`: cấu hình, vân tay, thời gian từng stage, tham số gom cụm, số candidate / instruction.

## D7. Xử lý sự cố

- `Không import được umap-learn` / `Chưa cài plotly`: chạy `uv sync --group viz`. Thiếu UMAP thì bản đồ vẫn có PCA.
- Hết VRAM ở `embed`: giảm `--embed-batch-size`; hoặc `--gpus-per-worker 1.0`.
- Ray báo nhiều cụm (`multiple Ray instances`): chạy `ray stop` rồi thử lại, hoặc nối cụm cố định bằng `--ray-address`.
- `ray_actor_pool`/`xenna` treo ở stage đầu: thử `--executor ray_actor_pool`, và kiểm tra `--num-gpus` không lớn hơn số GPU thật.
- Một run bị chặn "đã có tiến trình đang chạy": chỉ một lệnh được chạy cho mỗi `--run-name` cùng lúc.

---

# Phần E — Cài công cụ hệ thống để đọc .doc / .ppt / .djvu (giáo trình)

Giáo trình có một số file định dạng cũ mà thư viện Python không đọc được, cần hai phần mềm cài vào máy (không cài qua `uv`):

| Phần mềm | Dùng cho | Lệnh kiểm tra đã cài chưa |
|---|---|---|
| LibreOffice (chạy không giao diện) | đổi `.doc` / `.ppt` sang `.docx` / `.pptx` rồi đọc như file mới | `soffice --version` |
| djvulibre | lấy chữ từ `.djvu` (`djvutxt`), xuất ảnh trang để OCR (`ddjvu`) | `djvutxt --help` |

> Trạng thái: đã chốt cài (`docs/DECISION_LOG.md`, mục Q5); code đọc các định dạng này (D-05) **chưa viết**. Khi có code, thiếu
> công cụ thì file tương ứng được ghi `skipped="tool_missing"`, lần chạy không bị hỏng; cài xong chạy lại để xử lý các file đó.

Đếm trước xem có bao nhiêu file mỗi định dạng (chạy ở thư mục gốc repo trên server):
```bash
find data/raw/giao_trinh -type f | sed 's/.*\.//' | tr 'A-Z' 'a-z' | sort | uniq -c | sort -rn
```

**Máy có quyền root (Ubuntu / Debian)**:
```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends libreoffice-writer libreoffice-impress djvulibre-bin
soffice --version && djvutxt --help | head -1     # kiểm tra
```

**Container không có root (vd Jupyter trên Run:ai)**: cài vào môi trường conda riêng rồi thêm vào `PATH`:
```bash
conda create -y -n tools -c conda-forge libreoffice djvulibre
export PATH="$(conda env list | awk '$1=="tools"{print $NF}')/bin:$PATH"   # thêm dòng này vào ~/.bashrc
soffice --version && djvutxt --help | head -1     # kiểm tra
```

Lưu ý:
- LibreOffice chạy không giao diện (`soffice --headless`) và cần thư mục HOME ghi được; nếu báo lỗi profile thì đặt `export HOME=/tmp/lohome` trước khi chạy.
- Hai tiến trình `soffice` dùng chung một profile sẽ khoá nhau; code sẽ dùng profile riêng cho mỗi tiến trình (`-env:UserInstallation=file:///tmp/lo_<pid>`).

---

# Phần F — So sánh engine OCR (F-11, chạy trên server GPU)

Engine OCR hiện tại (Paddle detect + VietOCR) không xuất được `² ³`, nháy cong, `– —` và trộn dòng ở trang hai cột. Trước
khi thay bằng mô hình thị giác - ngôn ngữ, chạy một đợt so sánh theo quy tắc chọn đã chốt trong `docs/DECISION_LOG.md`
(F-11, O-02). Ứng viên khai báo ở `configs/ocr_bakeoff/engines.json`: `baseline`, `paddleocr_vl`, `dots_ocr`,
`qwen3vl_8b`, `qwen3vl_4b` (thêm engine = thêm một mục).

| Bước | Lệnh | Kết quả trong `<data-root>/processed/ocr_bakeoff/` |
|---|---|---|
| 1. Chọn trang (CPU) | `uv run python scripts/ocr_bakeoff/select_pages.py --data-root <root>` | `pages/`, `gt/`, `pages.jsonl`, `selection.json` (bộ A ~300 trang giáo trình có đáp án, bộ B ~40 trang stbook thật) |
| 2. Tập âm tiết (CPU) | `uv run python scripts/ocr_bakeoff/build_syllables.py --data-root <root>` | `syllables.json` (từ SEA-PILE v2) |
| 3. Chạy engine (GPU) | `uv run python scripts/ocr_bakeoff/run_engine.py --data-root <root> --engine <tên> --gpu <số> [--serve --vllm-bin <vllm>]` | `runs/<engine>/outputs.jsonl` (đầu ra thô từng trang), `run_info.json`, `server.log` |
| 4. Chấm điểm | `uv run python scripts/ocr_bakeoff/score.py --data-root <root> --ppl [--hour-budget N]` | `report/summary.json`, `report/per_page.csv`, `report/hours.csv`, `report/report.html` |

Chạy trọn cả 4 bước, mỗi engine một GPU (trong tmux):
```bash
DATA_ROOT=/duong/dan/data VLLM_BIN=/opt/vllm-env/bin/vllm HOUR_BUDGET=30 ./scripts/ocr_bakeoff/run_bakeoff.sh
```

Chuẩn bị môi trường:
- `baseline`: `uv sync --group ocr` (muốn phần detect chạy GPU thì cài thêm `paddlepaddle-gpu`).
- `dots_ocr`, `qwen3vl_*`: cài vLLM (>= 0.11) vào môi trường **riêng** (vd `python -m venv /opt/vllm-env && /opt/vllm-env/bin/pip install vllm`) rồi trỏ `--vllm-bin`. `--serve` tự dựng server bằng lệnh `serve` trong cấu hình, chờ tải xong mô hình, chạy xong thì tắt. Không dùng `--serve` thì tự dựng server trước (cổng ở `base_url`).
- `paddleocr_vl`: cần `paddleocr[doc-parser]` và `paddlepaddle-gpu` trong môi trường chạy script (`PADDLE_VL_RUN` trong `run_bakeoff.sh`).
- Thử nhanh một engine trước: `--limit 5`, rồi mở `runs/<engine>/outputs.jsonl` xem trường `text` và `error`.

Đọc kết quả:
- `summary.json` → `decision`: `winner` (engine chọn), `option_c` (có engine đủ chất lượng nhưng vượt giới hạn số giờ chạy: chỉ chạy VLM cho trang bị nghi), `no_candidate` (giữ baseline). Chưa đặt `--hour-budget` (số giờ tối đa server được chạy để OCR cả kho) thì kết luận chỉ là tạm.
- `hours.csv` (cũng có trong `summary.json` → `hours` và bảng "Số giờ chạy" của `report.html`): mỗi engine một dòng, gồm số trang và số giờ đã chạy thật trong đợt so sánh, số giờ chờ server / nạp mô hình, trang/giây trên 1 GPU, số giờ để OCR cả kho khi 4 GPU cùng chạy (`corpus_hours`), số giờ GPU tương ứng (`corpus_gpu_hours` = × 4) và số giờ tách theo nguồn.
- `report.html`: bảng cổng loại (xanh qua, đỏ trượt, xám chưa kiểm), số đo bộ A theo tầng, bộ B (âm tiết lạ, perplexity, `?`), từng trang bộ B và vài trang hai cột bộ A đặt cạnh nhau với ảnh gốc.
- Chạy lại được ở mọi bước: `run_engine.py` chỉ làm trang còn thiếu / lỗi (`--redo` để làm lại), `score.py` chấm lại bao nhiêu lần cũng được. Chọn lại bộ trang (`select_pages.py --overwrite`) sẽ xoá kết quả engine cũ.

---

# Cấu trúc code

Xem chi tiết (cây thư mục, cấu trúc `data/`, luồng từng nguồn) trong [`docs/PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md). Tóm tắt:

```
vi_corpus/
├── common/            # dùng chung: registry (sổ đăng ký nguồn), schema, state (checkpoint/khoá/log), ocr, pdf_text, parsers/ (docx, doc/ppt, djvu, html, epub)
├── sea/               # tải SEA (datasets, hub, checkpoint, downloader, cli) + reader về schema chung
├── stbook/            # crawler stbook.vn + ocr_books (OCR có checkpoint, đọc kết quả về schema chung)
├── giao_trinh/        # xử lý giáo trình (đang triển khai)
├── pipeline/          # pipeline end-to-end: ingest…finalize, embed/reduce/viz, report, curator (chạy song song)
├── ocr_bakeoff/       # so sánh engine OCR (Phần F): chọn trang, engine, adapter văn bản, số đo, chấm điểm
└── vista/, vjol/      # chỗ trống, chưa có code
scripts/
├── run_all.py         # chạy script của các phần theo thứ tự
├── run_pipeline.py    # pipeline end-to-end trên N mẫu (Phần D)
├── dedup_global.py    # dedup vòng 2 trên kho dấu vân tay
├── make_judge_sample.py  # mẫu phân tầng cho LLM chấm điểm
├── sea/               # 4 script tải, count_rows.py
├── stbook/            # crawl.py, ocr.py
├── giao_trinh/        # extract.py (đang viết), profile.py
└── ocr_bakeoff/       # select_pages.py, build_syllables.py, run_engine.py, score.py, run_bakeoff.sh (Phần F)
tests/                 # test (không cần mạng): uv run pytest
pyproject.toml         # khai báo thư viện; uv.lock ghi phiên bản chính xác (commit cả hai)
```
Muốn thêm một bộ dữ liệu HF mới: thêm một `DatasetSpec` vào `vi_corpus/sea/datasets.py`, rồi tạo script mới theo mẫu của một script có sẵn.

## Giấy phép dữ liệu
Các bộ SEA dùng giấy phép [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/), và người dùng cần tuân thủ [CommonCrawl ToU](https://commoncrawl.org/terms-of-use/).
Khi dùng dữ liệu, hãy trích dẫn AI Singapore theo hướng dẫn trên trang của từng dataset.
Sách stbook.vn thuộc bản quyền NXB Chính trị quốc gia Sự thật; chỉ dùng cho nghiên cứu cho tới khi xác nhận được quyền sử dụng.
