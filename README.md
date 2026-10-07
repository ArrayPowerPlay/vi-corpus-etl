# vi-corpus-etl

ETL xây **corpus tiếng Việt cho LLM** từ nhiều nguồn, có truy vết nguồn gốc (provenance) cho từng bản ghi.

| Nguồn | Nội dung | Trạng thái |
|---|---|---|
| SEA (3 bộ của AI Singapore) | Văn bản web + hội thoại instruct, phần tiếng Việt | Đã có bước **tải** (mục 1–7 bên dưới); bước xử lý: kế hoạch |
| stbook.vn | Sách NXB Chính trị quốc gia Sự thật (PDF dạng ảnh, tải bằng repo `stbook-crawler`) | Đã có bước **OCR → text** (Phần B) |
| Giáo trình (Google Drive) | Giáo trình đại học 16 ngành (PDF, pptx), tải bằng rclone | Đang viết bước xử lý (`docs/GIAO_TRINH.md`) |
| VJOL, VISTA | Bài báo khoa học (PDF), **chỉ xử lý** dữ liệu có sẵn, không crawl | Kế hoạch: `raw/VJOL/`, cấu trúc sẽ bổ sung khi có dữ liệu |

Chiến lược xử lý (nguồn → parse → ngôn ngữ → làm sạch → chất lượng → loại trùng → knowledge unit → audit):
xem [`docs/PIPELINE.md`](docs/PIPELINE.md), [`docs/SOURCES.md`](docs/SOURCES.md) và lộ trình [`docs/ROADMAP.md`](docs/ROADMAP.md).
Cấu trúc repo và thư mục dữ liệu: [`PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md).
Phần còn lại của README mô tả phần đã có code: **Phần A** tải dữ liệu SEA, **Phần B** OCR sách stbook, **Phần C** chạy tất cả bằng `scripts/run_all.py`, **Phần D** pipeline xử lý end-to-end trên N mẫu (có visualize, chạy nhiều GPU).

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
Mục đích: kiểm tra pipeline và chất lượng dữ liệu trước khi chạy toàn bộ. Thiết kế từng stage: [`docs/PIPELINE.md`](docs/PIPELINE.md).

| # | Stage (`--until`) | Làm gì | File kết quả trong thư mục run |
|---|---|---|---|
| 1 | `ingest` | Lấy mẫu từ `raw/` theo `--total` và `--mix` (stbook: nguyên cuốn đã OCR) | `01_ingest.parquet` |
| 2 | `prepare` | Chuẩn hóa Unicode, cắt sách thành đoạn, đếm từ/token | `02_prepare.parquet` |
| 3 | `language` | Nhận diện ngôn ngữ (vi/en/other) — **song song CPU** | `03_language.parquet` |
| 4 | `quality` | Số đo, điểm 0–100, band A/B/C/D, reason code — **song song CPU** | `04_quality.parquet` |
| 5 | `dedup` | Trùng chính xác + gần trùng (MinHash), cổng quyền (`unknown` → quarantine) | `05_dedup.parquet` |
| 6 | `embed` | Vector hóa văn bản — **song song nhiều GPU** | `06_embeddings.parquet` |
| 7 | `reduce` | PCA + UMAP xuống 2D, gom cụm HDBSCAN (cuML nếu có, không thì scikit-learn/umap-learn) | `07_reduced.parquet` |
| 8 | `finalize` | Bộ sạch, knowledge unit, audit, bản đồ plotly, báo cáo | `clean.parquet`, `knowledge_units.parquet`, `audit.json`, `report.html` (đã nhúng bản đồ) |

Mọi kết quả nằm ở `<data-root>/processed/pipeline_runs/<run-name>/` (mặc định `run_<total>_seed<seed>`). Mỗi stage có checkpoint: chạy lại **đúng lệnh cũ** thì stage nào đã có file sẽ được bỏ qua.

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
Không có GPU/mạng (thử trên máy nhỏ): dùng `--embedder tfidf` (CPU, không tải mô hình).

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
Với stbook, `--total` tính theo số **đoạn** (sách được cắt đoạn ~600 từ), không phải số cuốn.
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
Sửa ngưỡng hay code của một stage rồi muốn làm lại từ stage đó: thêm `--force-from quality` (xoá kết quả từ `quality` trở đi, giữ các stage trước).

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
- `audit.json`: tỉ lệ lineage, số trùng, số bị quarantine. `manifest.json`: cấu hình và thời gian từng stage.

## D7. Xử lý sự cố

- `Không import được umap-learn` / `Chưa cài plotly`: chạy `uv sync --group viz`. Thiếu UMAP thì bản đồ vẫn có PCA.
- Hết VRAM ở `embed`: giảm `--embed-batch-size`; hoặc `--gpus-per-worker 1.0`.
- Ray báo nhiều cụm (`multiple Ray instances`): chạy `ray stop` rồi thử lại, hoặc nối cụm cố định bằng `--ray-address`.
- `ray_actor_pool`/`xenna` treo ở stage đầu: thử `--executor ray_actor_pool`, và kiểm tra `--num-gpus` không lớn hơn số GPU thật.
- Một run bị chặn "đã có tiến trình đang chạy": chỉ một lệnh được chạy cho mỗi `--run-name` cùng lúc.

---

# Cấu trúc code

Xem chi tiết (cây thư mục, cấu trúc `data/`, luồng từng nguồn) trong [`PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md). Tóm tắt:

```
vi_corpus/
├── common/            # dùng chung: registry (sổ đăng ký nguồn), schema, state (checkpoint/khoá/log), ocr, pdf_text
├── sea/               # tải SEA (datasets, hub, checkpoint, downloader, cli) + reader về schema chung
├── stbook/            # crawler stbook.vn + ocr_books (OCR có checkpoint, đọc kết quả về schema chung)
├── giao_trinh/        # xử lý giáo trình (đang triển khai)
├── pipeline/          # pipeline end-to-end: ingest…finalize, embed/reduce/viz, report, curator (chạy song song)
└── vista/, vjol/      # chỗ trống, chưa có code
scripts/
├── run_all.py         # chạy script của các phần theo thứ tự
├── run_pipeline.py    # pipeline end-to-end trên N mẫu (Phần D)
├── sea/               # 4 script tải, count_rows.py
├── stbook/            # crawl.py, ocr.py
└── giao_trinh/        # extract.py (đang viết), profile.py
tests/                 # test (không cần mạng): uv run pytest
pyproject.toml         # khai báo thư viện; uv.lock ghi phiên bản chính xác (commit cả hai)
```
Muốn thêm một bộ dữ liệu HF mới: thêm một `DatasetSpec` vào `vi_corpus/sea/datasets.py`, rồi tạo script mới theo mẫu của một script có sẵn.

## Giấy phép dữ liệu
Các bộ SEA dùng giấy phép [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/), và người dùng cần tuân thủ [CommonCrawl ToU](https://commoncrawl.org/terms-of-use/).
Khi dùng dữ liệu, hãy trích dẫn AI Singapore theo hướng dẫn trên trang của từng dataset.
Sách stbook.vn thuộc bản quyền NXB Chính trị quốc gia Sự thật; chỉ dùng cho nghiên cứu cho tới khi xác nhận được quyền sử dụng.
