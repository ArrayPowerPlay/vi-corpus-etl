# Việc cần chạy trên server (SERVER_TASK)

> Danh sách việc phải làm **trên GPU server** (4× A100 40GB, Jupyter Lab, Linux), vì máy local không có dữ liệu thật và không có GPU.
> Mỗi việc ghi: trả lời câu hỏi nào, lệnh chạy, cách đọc kết quả, nơi ghi kết quả. Lập ngày 2026-10-08, theo quyết định N-02 / N-11 trong `docs/DECISION_LOG.md`.
> Kết quả dán vào mục "Bảng ghi kết quả" cuối file, rồi báo lại cho Claude ở phiên sau để cập nhật `DECISION_LOG.md`.

## 0. Quy ước chung

- Chạy bằng `tmux` hoặc `nohup` từ **Jupyter Terminal**, không chạy từ notebook cell. Ví dụ: `tmux new -s do` rồi chạy lệnh, thoát bằng `Ctrl+B` rồi `D`, quay lại bằng `tmux attach -t do`.
- Đặt đường dẫn dữ liệu một lần: `export SEA_DATA_ROOT=/duong/dan/data` (các script đọc biến này làm `--data-root` mặc định). Chỗ nào ghi `<root>` là đường dẫn đó.
- Trong lúc chạy việc chỉ dùng CPU mà vẫn muốn giữ workload GPU của Run:ai: bọc lệnh bằng `./scripts/run_with_gpu_keepalive.sh <script> <tham số>`.
- Ghi lại **số liệu thô** (giây, số trang, GB) chứ không chỉ kết luận; Claude sẽ tính lại.
- Mỗi việc dưới đây có nhãn: **[CHẠY NGAY]** (dùng script đã có) hoặc **[CHỜ CODE]** (cần script viết ở phiên code sau; ở đây ghi cách chạy dự kiến, sẽ cập nhật lệnh chính xác khi có code).

| Mã | Việc | Trả lời cho | Nhãn | Thời gian ước lượng | GPU |
|---|---|---|---|---|---|
| S-0 | Kiểm tra máy và cài thư viện | mọi việc | CHẠY NGAY | 10–20 phút | không |
| S-1 | Đếm tổng số trang PDF (stbook, giáo trình) | N-02 (OCR tốn bao nhiêu) | CHẠY NGAY | vài phút | không |
| S-2 | Khảo sát giáo trình: tỉ lệ trang scan | N-02 | CHẠY NGAY | 10–30 phút | không |
| S-3 | Đo tốc độ OCR (trang/giây) | N-02 | CHẠY NGAY | 10–20 phút | 1 |
| S-4 | Tổng số token SEA và tỉ lệ token/từ | N-11, mục tiêu token | CHẠY NGAY (bản thô) | 30–60 phút | không |
| S-5 | Tốc độ và RAM của pipeline hiện tại (D-06 bước đầu) | N-11 | CHẠY NGAY | 30–90 phút | tùy |
| S-6 | Thử Qwen3-Embedding-0.6B: tốc độ + so 256 / 512 / 1024 chiều | D-11, N-04 | CHỜ CODE | 1 giờ | 1 |
| S-7 | Dựng LLM judge (vLLM) + đo tốc độ trên 1.000 mẫu | N-06, ngân sách GPU | CHỜ CODE | 2–3 giờ | 4 |
| S-8 | Đo trùng giữa các nguồn (SEA-PILE v1 và v2) | N-04 (có cần kiểm trùng ngay khi chạy nguồn sau không) | CHỜ CODE | vài giờ | không |
| S-9 | Đo trùng của FineWeb-2 với SEA, tải 1 shard | N-09 (tùy chọn, khi quyết thêm FineWeb-2) | CHỜ CODE | 1–2 giờ | không |
| S-10 | VJOL: chép về server, đếm file / trang, khảo sát PDF text hay scan | R-25 (thời gian xử lý VJOL, số lần parse) | CHẠY NGAY | 1–3 giờ (tùy tốc độ chép) | không |
| S-11 | Đo dương tính giả của fuzzy dedup (chọn tham số MinHash) | N-14 / R-34 (đổi shingle 5 từ, 16×8, ≥0,8 sang shingle 3 từ, 32×4, ≥0,7 hay không) | CHỜ CODE | vài giờ (chạy pipeline 100k + đọc ~150 cặp) | không |

Thứ tự khuyến nghị: S-10 (VJOL ưu tiên cao nhất, R-15) → S-0 → S-1 → S-2 → S-3 (có đủ số để tính giờ OCR) → S-4 → S-5. S-6 đến S-9 và S-11 làm sau phiên code (S-11 cần script `scripts/dedup_pairs_sample.py`, viết ở gói G-05 bước 2).

---

## S-0. Kiểm tra máy và cài thư viện

```bash
nvidia-smi                       # phải thấy 4 GPU A100 40GB; ghi lại GPU đang bận hay rảnh
nproc && free -g && df -h $SEA_DATA_ROOT   # số CPU, RAM (GB), dung lượng đĩa còn trống
cd /duong/dan/vi-corpus-etl && git pull
uv sync                          # nền
uv sync --group ocr              # OCR (S-1, S-2, S-3)
uv sync --group curator          # chỉ khi chạy song song nhiều GPU (S-5)
```

Ghi lại: số CPU, RAM, dung lượng đĩa trống, GPU nào rảnh. Lưu ý: `uv sync --group X` chỉ cài thêm nhóm X; nếu một lệnh sau không thấy thư viện thì chạy lại `uv sync --group ocr --group curator` (gộp nhóm) để không bị gỡ nhóm khác.

## S-1. Đếm tổng số trang PDF  [CHẠY NGAY]

**Mục đích**: biết OCR phải xử lý bao nhiêu trang (khoản GPU lớn nhất chưa đo). Chỉ đọc metadata của PDF, không OCR.

Tạo file `count_pages.py` **ngoài repo** (vd `~/count_pages.py`):

```python
"""Đếm số file và số trang PDF dưới một thư mục (chỉ đọc metadata, không OCR)."""
import sys
from pathlib import Path

import pymupdf

root = Path(sys.argv[1])
files = pages = 0
errors = []
for pdf in sorted(root.rglob("*.pdf")):
    try:
        with pymupdf.open(pdf) as doc:
            files += 1
            pages += len(doc)
    except Exception as exc:  # file hỏng: ghi lại, không dừng
        errors.append((str(pdf), str(exc)[:80]))
print(f"{root}: {files} file PDF, {pages} trang, {len(errors)} file lỗi")
for path, msg in errors[:20]:
    print("  LỖI", path, msg)
```

Chạy (từ thư mục repo, dùng môi trường có pymupdf):

```bash
uv run --group ocr python ~/count_pages.py $SEA_DATA_ROOT/raw/stbook
uv run --group ocr python ~/count_pages.py $SEA_DATA_ROOT/raw/giao_trinh
```

Đã thử trên PDF giả ở máy local (đếm đúng số trang, báo lỗi file hỏng); chưa chạy trên dữ liệu thật. Với stbook, thư mục `<id>_pages/` (sách đang tải dở) chứa ảnh JPEG chứ không có `.pdf` nên không bị đếm.

**Ghi lại**: số file, số trang của từng nguồn. Ghi chú: số trang giáo trình gồm cả trang đã có text (không cần OCR); S-2 cho biết tỉ lệ trang cần OCR.

## S-2. Khảo sát giáo trình: tỉ lệ trang scan  [CHẠY NGAY]

**Mục đích**: biết bao nhiêu phần trăm trang giáo trình phải OCR (trang có text thì không tốn GPU).

```bash
uv run --group ocr python scripts/giao_trinh/survey.py --data-root $SEA_DATA_ROOT --per-dir 5   # thử nhanh
uv run --group ocr python scripts/giao_trinh/survey.py --data-root $SEA_DATA_ROOT                # đủ
```

Script lấy tối đa 40 trang rải đều mỗi PDF (`--max-pages`), phân loại trang `text / tcvn3 / scan / empty`, in bảng theo ngành và ghi CSV. Cột `scan` là trang cần OCR. Tỉ lệ scan nhân với tổng số trang ở S-1 ra số trang cần OCR của giáo trình. (Sau này dùng được cho VJOL: `--root <root>/raw/VJOL`.)

**Ghi lại**: tỉ lệ scan chung và theo ngành (đặc biệt nhóm y học), tỉ lệ tcvn3, vị trí file CSV.

## S-3. Đo tốc độ OCR (trang/giây)  [CHẠY NGAY]

**Mục đích**: quy ra giờ-GPU cho OCR = tổng trang cần OCR ÷ (trang/giây) ÷ 3600.

```bash
export CUDA_VISIBLE_DEVICES=0    # đo trên đúng 1 GPU
time uv run --group ocr python scripts/stbook/ocr.py --data-root $SEA_DATA_ROOT --limit-books 2 --device cuda
```

Trong lúc chạy, mở cửa sổ tmux thứ hai: `nvidia-smi dmon -s u` (cột `sm` là mức dùng GPU) và `top` (CPU).

Tính trang/giây: `time` cho tổng giây; số trang lấy từ file kết quả (mỗi cuốn là một JSON có khóa `pages`):

```bash
uv run python -c "
import json, glob, os
files = sorted(glob.glob(os.environ['SEA_DATA_ROOT'] + '/interim/stbook_ocr/*/*.json'), key=os.path.getmtime)[-2:]
print(sum(len(json.load(open(f))['pages']) for f in files), 'trang trong', len(files), 'cuốn mới nhất')
"
```

Trang/giây = số trang ÷ giây thực (bỏ ~1 phút nạp mô hình lần đầu nếu log cho thấy thời điểm bắt đầu). Nếu hai cuốn quá ít để đo, tăng `--limit-books` lên 5.

**Điều cần quan sát**: `uv sync --group ocr` cài paddlepaddle **bản CPU** (xem CLAUDE.md), nên bước phát hiện dòng chữ chạy trên CPU và chỉ VietOCR chạy trên GPU. Nếu `nvidia-smi` cho `sm` thấp (< 30%) mà CPU kín, nút thắt là bước phát hiện trên CPU, và cần nghĩ tới `paddlepaddle-gpu` hoặc chạy nhiều tiến trình song song. Đây là giả thuyết từ cấu hình, chưa kiểm chứng.

**Ghi lại**: trang/giây trên 1 GPU, mức `sm` GPU, số CPU kín, kích thước trang trung bình nếu thấy được.

Chạy song song 4 GPU: `scripts/run_pipeline.py --ocr-books N --executor ray_actor_pool --num-gpus 4` (cần `--group curator`); chưa kiểm trên GPU thật, đo thêm một lần nếu một GPU không đủ.

## S-4. Tổng số token SEA và tỉ lệ token/từ  [CHẠY NGAY, bản thô]

**Mục đích**: biết kho thô có bao nhiêu token (con số "30 tỷ" trong D-11 chỉ là ước lượng) để biết cần giữ lại bao nhiêu % mới đạt mục tiêu ≥10B clean (W7).

1. Số dòng từng bộ: `uv run python scripts/sea/count_rows.py --data-root $SEA_DATA_ROOT` (chỉ đọc metadata parquet).
2. Token trên mỗi dòng bằng tokenizer thật: chạy pipeline trên mẫu, đếm bằng Qwen3:

   ```bash
   uv run python scripts/run_pipeline.py --data-root $SEA_DATA_ROOT --total 10000 --mix sea_pile_v2=40,sea_lion_pile_v1=40,sea_instruct_2602=20 \
       --tokenizer Qwen/Qwen3-0.6B --embedder tfidf --run-name do_token_10k
   ```

   `--embedder tfidf` chạy CPU, khỏi nạp mô hình embedding. Khóa nguồn trong `--mix` là các khóa của `DEFAULT_MIX` (`sea_pile_v2`, `sea_lion_pile_v1`, `sea_instruct_2602`, `stbook`). Pipeline hiện tại đếm token bằng `transformers.AutoTokenizer` (cần mạng để tải tokenizer lần đầu); bản dùng thư viện `tokenizers` thuần (D-01) sẽ viết ở phiên code.
3. Đọc `<root>/processed/pipeline_runs/do_token_10k/audit.json`, khóa `tokens`: mỗi nguồn có `words_in` và `tokens_in` (số từ và số token của mẫu). Token trung bình mỗi bản ghi = `tokens_in` ÷ số bản ghi của nguồn; tỉ lệ token/từ = `tokens_in` ÷ `words_in`. **Tổng token ước lượng = số dòng (bước 1) × token trung bình mỗi dòng**, theo từng bộ. (Không có file này nếu chạy với `--until` sớm: audit chỉ ghi ở stage `finalize`.)

**Ghi lại**: số dòng và token/bản ghi trung bình của mỗi bộ, tổng token ước lượng, tỉ lệ token/từ.

## S-5. Tốc độ và RAM của pipeline hiện tại  [CHẠY NGAY]

**Mục đích**: bước đầu của D-06 — biết stage nào chậm và RAM tối đa trước khi viết lại theo shard.

```bash
/usr/bin/time -v uv run python scripts/run_pipeline.py --data-root $SEA_DATA_ROOT --total 100000 \
    --embedder tfidf --until dedup --run-name do_speed_100k 2> speed_100k.time.txt
```

- Thời gian từng stage: `manifest.json` của run (khóa `seconds` của mỗi stage) hoặc dòng log `[stage] N bản ghi (X s)`.
- RAM tối đa: dòng `Maximum resident set size (kbytes)` trong `speed_100k.time.txt`.
- Nếu 100.000 bản ghi hết RAM, hạ xuống `--total 20000` rồi `50000`, ghi lại điểm vỡ. Pipeline hiện tại nạp cả danh sách vào RAM mỗi stage (đã biết), nên điểm vỡ chính là con số cần cho D-06.
- docs/s = số bản ghi ÷ giây của stage. MB/s = tổng MB văn bản ÷ giây (tổng MB lấy từ `report.html` hoặc từ cỡ file `01_*.parquet`).

**Ghi lại**: docs/s của từng stage (`ingest`, `prepare`, `language`, `quality`, `dedup`), RAM tối đa, `--total` tối đa chạy được.

## S-6. Thử Qwen3-Embedding-0.6B  [CHỜ CODE]

**Mục đích**: (a) tốc độ nhúng đoạn ~1.024 token trên 1 A100; (b) chọn số chiều lưu (256 / 512 / 1024) bằng đo, không bằng cảm tính.

Sau khi `embed.py` được sửa (R-04: pooling token cuối, bỏ tiền tố `passage: `, cắt theo token):

1. Lấy 2.000 đoạn ~1.024 token từ `clean.parquet` của một run (nguồn trộn: web và sách).
2. Tạo 1.000 cặp **trùng gần** nhân tạo (xoá 10–30% câu, đổi vài từ) làm cặp dương, và 1.000 cặp khác bài làm cặp âm; thêm cặp âm "khó": hai đoạn cùng chủ đề, khác bài.
3. Nhúng 1.024 chiều một lần; cắt còn 128 / 256 / 512, **chuẩn hoá L2 lại** sau khi cắt; với mỗi cỡ tính ngưỡng cosine tách dương và âm tốt nhất, F1 tại ngưỡng đó.
4. Đo thêm: đoạn/giây, GPU memory, batch lớn nhất.

Script và lệnh chính xác sẽ ghi ở đây sau phiên code. Đầu ra mong muốn: bảng (số chiều → ngưỡng, F1, dung lượng/đoạn) và đoạn/giây.

## S-7. Dựng LLM judge và đo tốc độ  [CHỜ CODE]

**Mục đích**: biết LLM chấm 50.000 mẫu tốn bao nhiêu giờ-GPU, và có chạy được trên A100 40GB không.

Mô hình đã chốt: `Qwen3.5-27B` (DECISION_LOG R-12), không có mô hình dự phòng và không so sánh judge. Dạng bf16 của mô hình 27B (~54 GB) **không vừa một thẻ 40GB**, nên mỗi bản chạy trên 2 thẻ (tensor parallel); 4 thẻ chạy được 2 bản song song.

Cài vLLM trong **môi trường riêng** (thư viện nặng, dễ xung đột phiên bản torch với môi trường OCR / curator):

```bash
mkdir -p ~/vllm_env && cd ~/vllm_env && uv venv && uv pip install vllm
CUDA_VISIBLE_DEVICES=0,1 ~/vllm_env/.venv/bin/vllm serve <model> --tensor-parallel-size 2 --port 8000 &
CUDA_VISIBLE_DEVICES=2,3 ~/vllm_env/.venv/bin/vllm serve <model> --tensor-parallel-size 2 --port 8001 &
```

Kiểm tra trước: bản vLLM cài được có hỗ trợ kiến trúc của mô hình không (mô hình mới ra thường cần vLLM mới); nếu báo lỗi kiến trúc, nâng cấp vLLM (không đổi mô hình). Với Qwen3 nên tắt chế độ "suy nghĩ" khi chấm (tham số `chat_template_kwargs: {"enable_thinking": false}` trong yêu cầu); kiểm tra tài liệu của mô hình đã chọn.

Đo: gửi 1.000 đoạn ~1.024 token với câu lệnh chấm, ghi tổng giây, token vào/giây, token ra/giây, tỉ lệ trả lời đúng định dạng JSON. Giờ-GPU cho 50.000 mẫu = (giây cho 1.000 mẫu × 50) × 4 GPU ÷ 3600 (nếu dùng cả 4 thẻ). Ước lượng nháp của Claude: khoảng 8–9 giờ-GPU với mô hình ~30B dense, **chưa đo**.

Script chấm và câu lệnh chấm (thang 0–5) sẽ viết ở phiên code.

## S-8. Đo trùng giữa các nguồn  [CHỜ CODE]

Sau khi có dedup vòng 1 (D-09) cho hai bộ SEA-PILE: đo tỉ lệ văn bản của v1 trùng với v2 (cùng nguồn gốc CommonCrawl nên có thể rất cao). Kết quả quyết định có cần kiểm tra trùng với các nguồn đã xử lý ngay khi chạy nguồn sau hay không (N-04), và có nên tải thêm FineWeb-2. Cũng đo luôn "trùng bao hàm" (một file giáo trình là chương của file khác): tỉ lệ đoạn của file A xuất hiện trong file B.

## S-9. Đo trùng của FineWeb-2  [CHỜ CODE, tùy chọn]

Chỉ làm khi quyết định thêm FineWeb-2 (`vie_Latn`). Tải 1 shard, đo tỉ lệ trùng với SEA-PILE v1 / v2 bằng cùng cách S-8. Nếu > 70% trùng thì cân nhắc bỏ nguồn này.

## S-10. VJOL: chép, đếm, khảo sát  [CHẠY NGAY]

**Mục đích**: trả lời "xử lý VJOL mất bao lâu" và "mỗi PDF phải parse mấy lần" (DECISION_LOG R-25). Claude chỉ xem được một phần qua MinIO MCP (container MCP mất kết nối giữa chừng).

Dữ liệu ở MinIO: bucket `datacrawl`, thư mục `vjol.info.vn/`:

```
vjol.info.vn/
├── 1970/ … 2026/          # PDF phẳng, tên <số>_<số>.pdf (đoán: <article_id>_<galley_id>, cần kiểm bằng metadata)
│                          # 1970/ = 1.535 file: nhiều khả năng là bài không có ngày (mốc 0 của Unix), không phải năm 1970
└── metadata/
    ├── vjol_issues.jsonl            4 MB
    ├── vjol_listing.jsonl         233 MB
    ├── vjol_merged.jsonl          469 MB   (mới nhất, 2026-09-29)
    └── vjol_pdf_download_log.jsonl 34 MB
```

1. **Chép về** `<root>/raw/VJOL/` giữ nguyên cấu trúc (bằng `mc` hoặc `rclone` với khóa của bạn):

   ```bash
   mc alias set ailab https://storage-ailab.icenter.ai <ACCESS_KEY> <SECRET_KEY>
   mc du --depth 2 ailab/datacrawl/vjol.info.vn/          # dung lượng theo năm (ghi lại)
   mc mirror ailab/datacrawl/vjol.info.vn/ $SEA_DATA_ROOT/raw/VJOL/
   ```

   Chạy lại `mc mirror` là tiếp tục (bỏ qua file đã có). Ghi lại tổng dung lượng và thời gian chép.

2. **Đếm file và trang**: `uv run --group ocr python ~/count_pages.py $SEA_DATA_ROOT/raw/VJOL` (script ở S-1).

3. **Khảo sát text / scan / TCVN3** (5 PDF mỗi năm ≈ 180 PDF, rồi bản đủ nếu cần):

   ```bash
   uv run --group ocr python scripts/giao_trinh/survey.py --data-root $SEA_DATA_ROOT --root $SEA_DATA_ROOT/raw/VJOL --per-dir 5
   ```

   Cột "ngành" trong bảng là thư mục cấp 1, tức năm. Ghi lại theo năm: tỉ lệ trang `text` / `scan` / `tcvn3` / `empty`, `vi_ratio`, `bad_ratio`.

4. **Xem metadata** (không nạp cả file vào RAM):

   ```bash
   for f in $SEA_DATA_ROOT/raw/VJOL/metadata/*.jsonl; do echo "== $f"; wc -l < "$f"; head -c 3000 "$f"; echo; done
   ```

   Gửi cho Claude: số dòng mỗi file và 1–2 dòng mẫu của mỗi file (để biết trường nào có: tiêu đề, tóm tắt, tạp chí, ngôn ngữ, năm, giấy phép, ánh xạ tên file PDF ↔ bài báo).

5. Chọn tay 3 PDF (một bài cũ trước 2010, một bài mới 2 cột, một bài nhiều công thức), gửi đường dẫn cho Claude để thử parse ở phiên code.

---

## S-11. Đo dương tính giả của fuzzy dedup  [CHỜ CODE]

**Mục đích**: quyết N-14 (R-34): có đổi MinHash từ cấu hình A (shingle 5 từ, 16 băng × 8 hàng, Jaccard ≥ 0,8) sang C (shingle 3 từ, 32 × 4, ≥ 0,7) hay không. Thí nghiệm 2026-10-08 trên 59 văn bản kỹ thuật cho thấy C bắt 86% bản sửa 5% từ (A: 0%) và 100% bản cắt 10% đầu + 10% cuối (A: 44%), nhưng 59 văn bản không đủ để biết C có xoá nhầm dữ liệu tốt hay không. Việc này đọc các cặp thật để đo.

1. **Chạy pipeline trên mẫu lớn, dừng trước dedup** (dùng `--total` lớn nhất mà S-5 cho thấy RAM chịu được; 100.000 nếu được):

   ```bash
   uv run python scripts/run_pipeline.py --data-root $SEA_DATA_ROOT --total 100000 --run-name run_100000_seed42 --until quality
   ```

   Kết quả là `04_quality.parquet` trong `<root>/processed/pipeline_runs/run_100000_seed42/`. (Nếu gói G-03 / G-04 đã đổi tokenizer và ngôn ngữ thì chạy sau khi chúng xong, để mẫu giống pipeline thật.)

2. **Lấy mẫu cặp** bằng script ở phiên code (tên dự kiến `scripts/dedup_pairs_sample.py`, lệnh chính xác ghi ở đây sau phiên code). Script tính chữ ký MinHash cho cả A và C trên các bản ghi đã qua `quality`, rồi ghi `pairs_sample.jsonl` gồm, **mỗi nhóm 50 cặp ngẫu nhiên**:
   - nhóm 1: Jaccard ước lượng ≥ 0,8 (đối chứng: cặp cả A lẫn C đều bắt);
   - nhóm 2: Jaccard trong [0,7; 0,8) (cặp **chỉ C** bắt thêm; đây là nhóm cần đọc kỹ nhất);
   - nhóm 3: Jaccard trong [0,6; 0,7) (chỉ để biết giới hạn).

   Mỗi cặp ghi: hai `doc_id`, hai `source_key`, Jaccard ước lượng, số từ, ~300 ký tự đầu của mỗi văn bản. Script cũng in số liệu toàn bộ: số cặp A bắt, số cặp C bắt, số cặp chỉ C bắt, chia theo cặp **cùng nguồn** và **khác nguồn**.

3. **Đọc từng cặp và gán nhãn** (một trong ba):
   - `trung`: cùng nội dung, bỏ một bản là đúng;
   - `cung-mau`: cùng khuôn nhưng thông tin khác (tin theo mẫu, mô tả sản phẩm, hỏi-đáp, văn bản pháp luật): **không được bỏ**;
   - `khac`: không liên quan.

4. **Ghi lại**: bảng nhóm × (`trung`, `cung-mau`, `khac`) × nguồn, và các số liệu toàn bộ ở bước 2.

**Tiêu chí đề xuất (Claude, chủ dự án chốt)**: chuyển sang C nếu ≥ 90% cặp trong nhóm 2 là `trung`. Nếu `cung-mau` tập trung ở một nguồn (ví dụ SEA-Instruct hoặc tin theo mẫu) thì giữ A hoặc tắt fuzzy riêng cho nguồn đó (cấu hình theo `SourceProfile`), không đổi chung. Ngưỡng 90% là đề xuất, chưa được duyệt.

---

## Bảng ghi kết quả

(Điền sau khi chạy. Ghi ngày, người chạy, lệnh đã chạy nếu khác hướng dẫn.)

| Mã | Ngày | Kết quả (số liệu thô) | Ghi chú |
|---|---|---|---|
| S-0 | | CPU: , RAM: GB, đĩa trống: GB | |
| S-1 | | stbook: file / trang; giáo trình: file / trang | |
| S-2 | | tỉ lệ scan chung: %; theo ngành: ; tcvn3: % | |
| S-3 | | trang/giây (1 GPU): ; sm GPU: %; CPU kín: | |
| S-4 | | dòng và token/dòng mỗi bộ; tổng token ước lượng: | |
| S-5 | | docs/s từng stage; RAM tối đa: GB; `--total` tối đa: | |
| S-6 | | | |
| S-7 | | | |
| S-8 | | | |
| S-9 | | | |
| S-10 | | số file: ; dung lượng: GB; thời gian chép: ; số trang: ; tỉ lệ text / scan / tcvn3 theo năm: ; số dòng metadata: | |
| S-11 | | số cặp A bắt: ; C bắt: ; chỉ C bắt: (cùng nguồn / khác nguồn); nhóm 2 (0,7–0,8): trung / cung-mau / khac = ; nhóm 1: ; nhóm 3: | |
