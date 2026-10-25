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
| S-12 | VJOL: trích text mẫu 3.000 bài (4 GPU) rồi chạy pipeline 10.000 đoạn | J-05 (số liệu để ước giờ cả kho VJOL, tỉ lệ trang scan / flagged / baseline) | CHẠY NGAY | trích text: vài giờ (chưa đo); pipeline: ~1 giờ | 4 |
| S-14 | VJOL: đo nguồn chữ đảo ngược / producer / tốc độ embed, rồi chạy `vjol_10k_v3` và so với `v2` | J-22, J-23 | CHẠY SAU S-13 | trích lại ~3 giờ đồng hồ (4 GPU), pipeline ~1 giờ | 4 |
| S-15 | VJOL: đo gốc trên `vjol_10k_v3`, sinh danh sách từ không dấu, chạy lại pipeline từ checkpoint cũ (`vjol_10k_v4a`) | J-25, J-26, J-27 | CHẠY SAU S-14 | CPU vài chục phút; pipeline ~1 giờ + perplexity vài phút GPU | 1 |
| S-16 | VJOL: chạy lại sau sửa J-28 (`vjol_10k_v4b`), lập bộ gán nhãn và tập chữ đúng, đo lại trang nghi | J-28 | CHẠY SAU S-15 | pipeline ~10 phút (checkpoint có sẵn) + CPU vài phút | 1 |
| S-17 | VJOL: trích lại một lần (cache OCR đã nạp) rồi chạy pipeline `vjol_10k_v5` | J-29 | CHẠY SAU S-16 | trích lại vài chục phút trên 4 GPU (3.652 trang nghi + 27 bài ScanGate, nhiều trang có sẵn trong cache); pipeline ~10 phút | 4 |
| S-18 | VJOL: trích lại một lần sau sửa J-33 (khôi phục "ư", trang không lớp chữ, câu rác VLM, năm 1970 ...) rồi chạy pipeline `vjol_10k_v6` và so với `v5` | J-34 | CHẠY SAU S-17 | trích lại vài chục phút trên 4 GPU (chỉ vài trăm trang mới phải OCR, phần còn lại có trong cache); pipeline ~10 phút | 4 |
| S-19 | VJOL: chạy lại pipeline `vjol_10k_v7` sau sửa J-35 (không trích lại) và so với `v6` | J-35 | CHẠY SAU S-18 | pipeline ~10 phút | 1 |

Thứ tự khuyến nghị: S-10 (VJOL ưu tiên cao nhất, R-15) → S-0 → S-1 → S-2 → S-3 (có đủ số để tính giờ OCR) → S-4 → S-5. S-6 đến S-9 và S-11 làm sau phiên code (S-11 cần script `scripts/dedup_pairs_sample.py`, viết ở gói G-05 bước 2).

---

## S-0. Kiểm tra máy và cài thư viện

```bash
source /home/jovyan/dinh/env.sh  # HOME bị xoá khi server tự tắt; uv, vLLM, cache đặt trên Lustre (docs/README.md Phần F, O-06)
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

Script lấy tối đa 40 trang rải đều mỗi PDF (`--max-pages`), phân loại trang `text / tcvn3 / scan / empty`, in bảng theo ngành và ghi CSV. Cột `scan` là trang cần OCR. Tỉ lệ scan nhân với tổng số trang ở S-1 ra số trang cần OCR của giáo trình. (Sau này dùng được cho VJOL: `--root <root>/raw/vjol`.)

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
source /home/jovyan/dinh/env.sh   # vLLM đã cài bằng `uv tool install` theo docs/README.md Phần F (O-06)
CUDA_VISIBLE_DEVICES=0,1 vllm serve <model> --tensor-parallel-size 2 --port 8000 &
CUDA_VISIBLE_DEVICES=2,3 vllm serve <model> --tensor-parallel-size 2 --port 8001 &
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

1. **Chép về** `<root>/raw/vjol/` (chữ thường, J-02; **dữ liệu đã có trên server nên có thể bỏ bước này**) giữ nguyên cấu trúc (bằng `mc` hoặc `rclone` với khóa của bạn):

   ```bash
   mc alias set ailab https://storage-ailab.icenter.ai <ACCESS_KEY> <SECRET_KEY>
   mc du --depth 2 ailab/datacrawl/vjol.info.vn/          # dung lượng theo năm (ghi lại)
   mc mirror ailab/datacrawl/vjol.info.vn/ $SEA_DATA_ROOT/raw/vjol/
   ```

   Chạy lại `mc mirror` là tiếp tục (bỏ qua file đã có). Ghi lại tổng dung lượng và thời gian chép.

2. **Đếm file và trang**: `uv run --group ocr python ~/count_pages.py $SEA_DATA_ROOT/raw/vjol` (script ở S-1).

3. **Khảo sát text / scan / TCVN3** (5 PDF mỗi năm ≈ 180 PDF, rồi bản đủ nếu cần):

   ```bash
   uv run --group ocr python scripts/giao_trinh/survey.py --data-root $SEA_DATA_ROOT --root $SEA_DATA_ROOT/raw/vjol --per-dir 5
   ```

   Cột "ngành" trong bảng là thư mục cấp 1, tức năm. Ghi lại theo năm: tỉ lệ trang `text` / `scan` / `tcvn3` / `empty`, `vi_ratio`, `bad_ratio`.

4. **Xem metadata** (không nạp cả file vào RAM):

   ```bash
   for f in $SEA_DATA_ROOT/raw/vjol/metadata/*.jsonl; do echo "== $f"; wc -l < "$f"; head -c 3000 "$f"; echo; done
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

## S-12. VJOL: trích text mẫu và chạy pipeline 10.000 đoạn  [CHẠY NGAY]

**Mục đích**: lần chạy thật đầu tiên của VJOL (J-05). Cho số liệu thật để: ước giờ cả kho, biết tỉ lệ trang scan / bị loại / `baseline` (O-27 c), biết N bài cần để đủ 10.000 đoạn, xem `report.html` của nguồn `vjol`. Dữ liệu đã nằm ở `<root>/raw/vjol/` (J-02).

```bash
cd /home/jovyan/dinh/vi-corpus-etl && git fetch origin && git checkout worktree-vjol-pipeline && git pull
source /home/jovyan/dinh/env.sh                       # server cũ có vLLM sẵn; server mới phải cài vLLM trước (tạo env.sh như server cũ)
# 1) thử không GPU 20 bài (kiểm đường dẫn, metadata, đọc PDF), ~1 phút
uv run python scripts/vjol/extract.py --data-root data --sample-docs 20 --no-ocr && uv run python scripts/vjol/extract.py --data-root data --status
# 2) trích text mẫu 3.000 bài trên 4 GPU (trong tmux); chạy lại đúng lệnh này nếu bị ngắt
DATA_ROOT=data VLLM_BIN=$(which vllm) SAMPLE_DOCS=3000 ./scripts/vjol/run_extract_vlm.sh
# 3) pipeline 10.000 đoạn VJOL (lần đầu tải tokenizer Qwen3 + fastText + mô hình nhúng)
uv run --group curator --group viz python scripts/run_pipeline.py --data-root data --mix vjol=10000 --total 10000 --run-name vjol_10k
```

Sau lệnh 2, nếu log pipeline báo `chỉ lấy được X/10000 mẫu` thì tăng `SAMPLE_DOCS` (vd 6000) rồi chạy lại lệnh 2 và 3 (chỉ làm thêm phần mới; pipeline tự chạy lại stage bị ảnh hưởng).

**Báo lại cho Claude** (để ghi vào `DECISION_LOG.md`): in của `extract.py --status` (số PDF trong raw / đã trích / lỗi / bỏ qua; số trang theo `method`; trang `ocr_vlm` theo `status` và engine; tỉ lệ `flagged`; số trang `baseline`; tổng giờ), số bài có metadata (`metadata_found` trong cột `meta` của `01_ingest.parquet`), `manifest.json` của run (số đoạn mỗi bài ở mục `chunk`, phân bố ngôn ngữ vi / en / khác ở `language`, số đoạn bị loại và lý do), thời gian chạy lệnh 2 và 3, và file `report.html`.

---

## S-13. VJOL: chạy lại mẫu 3.000 bài sau khi sửa chữ hỏng do font (J-10)  [CHẠY SAU S-12]

**Mục đích**: ~12% đoạn của lần chạy S-12 là chữ hỏng do font PDF. Bản code mới dò trang lỗi font và đưa qua OCR VLM (vẽ lại cả trang). Phải đo trước khi tốn GPU để chốt ngưỡng, rồi chạy lại mẫu và so với J-09.

```bash
cd /home/jovyan/dinh/vi-corpus-etl && git fetch origin && git checkout worktree-vjol-pipeline && git pull
source /home/jovyan/dinh/env.sh
# 1) (J-18: ngưỡng đã chốt, bỏ lượt đo thêm; bước này chỉ để xem trước, có thể bỏ qua) đo CPU trên checkpoint cũ với ngưỡng mới
#    unknown cần data/processed/ocr_bakeoff/syllables.json, thiếu thì bỏ qua và cảnh báo
ls -la data/processed/ocr_bakeoff/syllables.json
uv run python scripts/vjol/check_garbled.py --data-root data
# 2) chạy lại mẫu 3.000 bài trên 4 GPU, trong tmux: checkpoint cũ tự bị làm lại; ghi lại giờ thực bằng `date` trước và sau.
#    Kiểm trước là có data/processed/ocr_bakeoff/syllables.json (nếu thiếu, log có dòng "Không thấy tập âm tiết" và tín hiệu unknown bị bỏ)
date; DATA_ROOT=data VLLM_BIN=$(which vllm) SAMPLE_DOCS=3000 ./scripts/vjol/run_extract_vlm.sh; date
# 3) đo lại: số trang lỗi font còn sót phải gần 0 (trang đã OCR giữ garbled_text nên script vẫn đếm được)
uv run python scripts/vjol/check_garbled.py --data-root data
# 4) pipeline mới, tên run mới để so với vjol_10k
uv run --group curator --group viz python scripts/run_pipeline.py --data-root data --mix vjol=10000 --total 10000 --run-name vjol_10k_v2
```

**Lưu ý**: lệnh 2 làm lại **cả** 3.000 bài, kể cả OCR lại trang scan (khoảng 1,6 giờ theo ước, chưa đo). Nếu không chờ được, báo Claude để thêm cờ chỉ làm lại các bài bị nghi.

**Báo lại cho Claude**: `tar` ở lệnh 1 (hoặc nội dung `summary.json` + `examples.txt`), `extract.py --status` sau lệnh 2 (dòng "Lỗi font", số trang `ocr_vlm`, `flagged`), giờ thực của lệnh 2 (hai dòng `date`), `summary.json` của lần đo ở lệnh 3, `manifest.json` + `report.html` của `vjol_10k_v2` (số đoạn band D lý do `garbled_font`).

---

## S-14. VJOL: đo (chữ đảo ngược, producer, embed) rồi chạy `vjol_10k_v3`  [CHẠY SAU S-13]

**Mục đích**: J-22 / J-23. Bước 1-3 đo bằng CPU hoặc GPU nhẹ, chưa sửa dữ liệu (đã chạy 2026-10-23, kết quả ở J-24); bước 4 làm lại trích text (tín hiệu `nodiac` mới nên `EXTRACT_VERSION` 3 làm lại cả 3.000 bài, kể cả OCR lại trang scan, ~3 giờ đồng hồ trên 4 GPU như S-13); bước 5-7 chạy pipeline mới và so với `vjol_10k_v2`.

```bash
cd /home/jovyan/dinh/vi-corpus-etl && git fetch origin && git checkout worktree-vjol-pipeline && git pull
source /home/jovyan/dinh/env.sh
uv sync --group curator --group viz --group ocr
uv run pytest -q                                   # kỳ vọng 242 test qua
ls -la data/processed/ocr_bakeoff/syllables.json   # cần cho reversed_text, odd_words, tín hiệu unknown
# 1) (J-22 (1)) chữ đảo ngược đến từ lớp chữ nhúng (method=text) hay từ OCR VLM (method=ocr_vlm)? CPU, vài giây
uv run python scripts/vjol/diagnose.py --data-root data inspect 8053925644 26f746692e | tee data/processed/vjol_diagnose_inspect.txt
# 2) (J-22 (3) A) phân bố producer / creator của PDF; CPU, vài phút -> data/processed/vjol_diagnose/{producers.csv,summary.json}
uv run python scripts/vjol/diagnose.py --data-root data producers
# 3) (J-22 (8)) đo riêng tốc độ embed: DỪNG hết vLLM / tiến trình dùng GPU trước (xem danh sách in ra); dùng run cũ để lấy 640 đoạn
nvidia-smi
uv run --group viz python scripts/bench_embed.py --run-dir data/processed/pipeline_runs/vjol_10k_v2 --n 640 2>&1 | tee data/processed/vjol_bench_embed.txt
# 4) làm lại trích text 3.000 bài trên 4 GPU (tmux); ghi giờ thực bằng `date`
# (J-24) từ lần này trang scan nghi bị xoay được xoay thử 180/90/270 độ; xem log: grep 'Xoay thử' data/logs/vjol_extract_gpu*.log
date; DATA_ROOT=data VLLM_BIN=$(which vllm) SAMPLE_DOCS=3000 ./scripts/vjol/run_extract_vlm.sh; date
nvidia-smi                                         # phải không còn tiến trình vLLM nào trước khi chạy pipeline (bước 5)
uv run python scripts/vjol/check_garbled.py --data-root data      # có thêm tín hiệu nodiac (summary.json, new_signals.txt)
# 5) pipeline mới; embed yêu cầu GPU (không có thì dừng với lỗi), bản đồ cụm trên UMAP 10 chiều (J-22 (9))
uv run --group curator --group viz python scripts/run_pipeline.py --data-root data --mix vjol=10000 --total 10000 \
    --run-name vjol_10k_v3 --embed-device cuda --cluster-space umap10
# 6) (J-22 (10)) bộ đọc tay: 300 ngẫu nhiên + 100 nghi ngờ -> data/processed/pipeline_runs/vjol_10k_v3/review/review_sample.{csv,md}
uv run python scripts/vjol/review_sample.py --run-dir data/processed/pipeline_runs/vjol_10k_v3
# 7) gói kết quả gửi lại
tar czf vjol_s14.tar.gz data/processed/vjol_diagnose data/processed/vjol_diagnose_inspect.txt data/processed/vjol_bench_embed.txt \
    data/processed/vjol_garbled_check data/processed/pipeline_runs/vjol_10k_v3/{manifest.json,report.html,review}
```

**Báo lại cho Claude**: gói `vjol_s14.tar.gz`, `extract.py --status` sau bước 4 và hai dòng `date`. Claude so với `vjol_10k_v2` (J-21 / J-22): số đoạn lỗi chữ nặng (v2: ~236, band A ~180), đoạn có email (v2: 597), mã `math_font_junk` / `private_use_chars` / `lost_diacritics` / `reversed_text` / `odd_words` / `reference_list` trong `manifest.json`, `symbol_chars` ở mục `normalize`, tỷ lệ nhiễu và số cụm của HDBSCAN (v2: 93,3% nhiễu trên 768 chiều), giây nhúng 10.000 đoạn (v2: 4.252) và dòng log `[embed] ... device=...`. Từ gói này chốt (1) B / C, (3) A và phần còn lại của (8).

---

## S-15. VJOL: đo gốc, danh sách từ không dấu, pipeline `vjol_10k_v4a` (chưa trích lại)  [CHẠY SAU S-14]

**Mục đích**: J-27 đã code Bước 0-4 của J-25. Lượt này CHƯA trích lại text (không tốn GPU trích): (1) đo gốc bằng công cụ mới trên `vjol_10k_v3`; (2) lập 400 đoạn để Claude đọc gán nhãn (chia hai nửa theo bài) và tập chữ đúng; (3) sinh danh sách từ không dấu từ SEA-PILE; (4) tái hiện các lỗi trên PDF thật bằng `diagnose.py inspect`; (5) nối producer với đoạn xấu (J-26); (6) chạy lại pipeline từ checkpoint v3 (đổi A, C, E, F, D1, D3 chỉ nằm ở bước tạo bản ghi và stage quality, không cần trích lại). Trích lại (Bước 3: gạch nối thay ư/ơ, A phần trang) và mẫu mới 3.000 bài hạt giống khác làm SAU, khi Claude đã chỉnh ngưỡng bằng nửa tune của bộ đáp án (J-26).

```bash
cd /home/jovyan/dinh/vi-corpus-etl && git fetch origin && git checkout worktree-vjol-pipeline && git pull
source /home/jovyan/dinh/env.sh
uv sync --group curator --group viz --group ocr
uv run pytest -q                                   # kỳ vọng 263 test qua
V3=data/processed/pipeline_runs/vjol_10k_v3
# 1) danh sách từ không dấu từ kho chữ sạch (CPU, vài phút) -> data/processed/ocr_bakeoff/unaccented.json; kiểm in ra: "doanh", "cung" phải KHÔNG có
uv run python scripts/build_unaccented.py --data-root data
# 2) Bước 0: số đo gốc trên v3 (CPU, vài giây mỗi lệnh)
uv run python scripts/vjol/audit_run.py --run-dir $V3                                   # -> $V3/audit/{audit.json,audit.md,audit_journals.csv}
uv run python scripts/vjol/groundtruth.py sample --run-dir $V3 --n 400                  # -> $V3/groundtruth/labeling_sample.{csv,md}
uv run python scripts/vjol/groundtruth.py clean-set --run-dir $V3 --n 300               # -> $V3/groundtruth/clean_set.jsonl
uv run python scripts/vjol/groundtruth.py check-clean --clean-set $V3/groundtruth/clean_set.jsonl --data-root data   # phải "0 bị gắn cờ nhầm"
# 3) tái hiện trên PDF thật (J-25): gạch nối 243535 (2009), 244285 (2004); mất dấu nhầm c00360926b; lỗi đọc font acd7883f9b; OCR cũ 02b971e693
uv run python scripts/vjol/diagnose.py --data-root data inspect 243535 244285 c00360926b acd7883f9b 02b971e693 | tee data/processed/vjol_diagnose_inspect_j25.txt
# 4) trang bị nghi theo từng tín hiệu, có thêm "hyphen" và danh sách mới (CPU)
uv run python scripts/vjol/check_garbled.py --data-root data
# 5) producer: một dòng mỗi bài, rồi đoạn xấu còn lọt ở v3 theo producer (J-26)
uv run python scripts/vjol/diagnose.py --data-root data producers
uv run python scripts/vjol/diagnose.py --data-root data chunks-by-producer --run-dir $V3
# 6) nạp bộ nhớ đệm OCR từ checkpoint cũ (CPU, vài phút; để lần trích lại sau không OCR lại trang đã OCR)
uv run python scripts/vjol/seed_ocr_cache.py --data-root data
# 7) pipeline từ checkpoint v3 với mã mới + perplexity Qwen3-0.6B (GPU nhỏ); phải không còn tiến trình vLLM nào
nvidia-smi
uv run --group curator --group viz python scripts/run_pipeline.py --data-root data --mix vjol=10000 --total 10000 \
    --run-name vjol_10k_v4a --embed-device cuda --cluster-space umap10 --perplexity-model Qwen/Qwen3-0.6B
uv run python scripts/vjol/audit_run.py --run-dir data/processed/pipeline_runs/vjol_10k_v4a
# 8) gói kết quả gửi lại
tar czf vjol_s15.tar.gz data/processed/ocr_bakeoff/unaccented.json data/processed/vjol_diagnose data/processed/vjol_diagnose_inspect_j25.txt data/processed/vjol_garbled_check \
    $V3/audit $V3/groundtruth data/processed/pipeline_runs/vjol_10k_v4a/{manifest.json,report.html,audit}
```

**Báo lại cho Claude**: gói `vjol_s15.tar.gz`. Claude (1) đọc 400 đoạn của `labeling_sample.md` và điền nhãn (cột `label`; chia hai nửa theo bài: tune để chỉnh ngưỡng, holdout chỉ để kiểm), (2) so `audit.json` của v3 và v4a với tiêu chí nghiệm thu J-25 (0 cờ lên tập chữ đúng; 102 đoạn bị loại oan được trả lại; `econst` hết bất thường; tiêu đề trang < 0,2%; ...), (3) xem `producers_chunks.csv` để quyết quy tắc D2, (4) chốt ngưỡng `hyphen`, `ODD_SOFT`, `PPL_HIGH`, độ chính xác nhãn tham khảo (>= 95% mới được loại hẳn). Sau đó mới có lượt trích lại một lần và mẫu mới (J-26).

## S-16. VJOL: chạy lại sau sửa J-28 (`vjol_10k_v4b`)  [CHẠY SAU S-15]

**Mục đích**: J-28 sửa ba lỗi đo (mất dấu bắt nhầm tiếng Anh / đường dẫn, che số điện thoại nhầm số liệu bảng, gạch nối bắt nhầm "C- section"). Chưa trích lại text. Chạy lại pipeline từ checkpoint cũ ra `vjol_10k_v4b` (cùng mẫu với v4a để so), rồi lập bộ gán nhãn từ chính v4b. **Chép CẢ khối lệnh bên dưới cùng một lần** (S-15 hỏng vì thiếu dòng gán biến).

```bash
cd /home/jovyan/dinh/vi-corpus-etl
git fetch origin && git checkout worktree-vjol-pipeline && git pull
source /home/jovyan/dinh/env.sh
uv sync --group curator --group viz --group ocr
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -c "from vi_corpus.common.ocr import PageOcr; PageOcr(device='cuda'); print('baseline_ok')" 2>&1 | tail -3   # phải in baseline_ok (mất 1-3 phút)
uv run pytest -q                                   # kỳ vọng 272 test qua
nvidia-smi                                         # không còn tiến trình vLLM nào
[ -d data/interim/vjol_text_v3_backup ] || cp -r data/interim/vjol_text data/interim/vjol_text_v3_backup   # chỉ sao lưu một lần
date; DATA_ROOT=data VLLM_BIN=$(which vllm) SAMPLE_DOCS=3000 ./scripts/vjol/run_extract_vlm.sh && date && \
V5=data/processed/pipeline_runs/vjol_10k_v5 && \
uv run python scripts/vjol/extract.py --data-root data --status && \
uv run --group curator --group viz python scripts/run_pipeline.py --data-root data --mix vjol=10000 --total 10000 \
    --run-name vjol_10k_v5 --embed-device cuda --cluster-space umap10 --perplexity-model Qwen/Qwen3-0.6B && \
uv run python scripts/vjol/audit_run.py --run-dir $V5 && \
uv run python scripts/vjol/groundtruth.py sample --run-dir $V5 --n 400 && \
uv run python scripts/vjol/groundtruth.py clean-set --run-dir $V5 --n 300 && \
uv run python scripts/vjol/groundtruth.py check-clean --clean-set $V5/groundtruth/clean_set.jsonl --data-root data && \
uv run python scripts/vjol/check_garbled.py --data-root data && \
tar czf vjol_s17.tar.gz data/processed/ocr_bakeoff/syllables.json data/processed/ocr_bakeoff/unaccented.json data/processed/vjol_garbled_check \
    data/logs/vjol_extract_gpu*.out $V5/{manifest.json,04_quality.parquet,audit,groundtruth}
```

Các lệnh sau bước trích nối bằng `&&`: trích text lỗi thì dừng, không chạy pipeline trên bản trích cũ (lần đầu chạy S-17 bị đúng chuyện này, J-30). Lỗi môi trường của engine baseline giờ lộ ra trước khi bật vLLM; dòng `baseline_ok` ở trên kiểm trước một mình. Nguyên nhân lần hỏng trước: paddlex đòi gói `opencv-contrib-python` mà repo chỉ cài bản headless; đã thay bằng gói giả `third_party/opencv-contrib-python-shim` (J-31), `uv sync --group ocr` sẽ cài nó.

**Báo lại cho Claude**: gói `vjol_s16.tar.gz`. Claude so v4b với v4a (kỳ vọng: `lost_diacritics` khoảng 195 thay vì 476, không còn `<PHONE>` trên số liệu bảng), đọc 400 đoạn gán nhãn (nửa tune chỉnh ngưỡng, nửa holdout chỉ kiểm), rồi chốt D2 theo producer, lần trích lại một lần và mẫu 3.000 bài hạt giống khác (J-26).

## S-17. VJOL: trích lại một lần và chạy `vjol_10k_v5`  [CHẠY SAU S-16]

**Mục đích**: mọi sửa phía trích text (gạch nối `đ-ợc` -> OCR, chữ trần đã bỏ nhầm tiếng Anh / đường dẫn, quy tắc producer ABBYY + ScanGate, J-29) chỉ có tác dụng khi trích lại. Đổi danh sách từ không dấu làm `extract_version` đổi nên TOÀN BỘ 2.937 checkpoint được làm lại; bước đọc lớp chữ rất nhanh, chỉ trang nghi mới OCR và trang đã có trong `interim/vjol_ocr_cache/` (nạp ở S-15) không OCR lại. Sao lưu checkpoint cũ trước. Dùng đúng `SAMPLE_DOCS=3000 SEED=0` như các lần trước để giữ cùng tập bài.

```bash
cd /home/jovyan/dinh/vi-corpus-etl
git fetch origin && git checkout worktree-vjol-pipeline && git pull
source /home/jovyan/dinh/env.sh
uv sync --group curator --group viz --group ocr
uv run pytest -q                                   # kỳ vọng 271 test qua
nvidia-smi                                         # không còn tiến trình vLLM nào
cp -r data/interim/vjol_text data/interim/vjol_text_v3_backup
date; DATA_ROOT=data VLLM_BIN=$(which vllm) SAMPLE_DOCS=3000 ./scripts/vjol/run_extract_vlm.sh; date
uv run python scripts/vjol/extract.py --data-root data --status
V5=data/processed/pipeline_runs/vjol_10k_v5
uv run --group curator --group viz python scripts/run_pipeline.py --data-root data --mix vjol=10000 --total 10000 \
    --run-name vjol_10k_v5 --embed-device cuda --cluster-space umap10 --perplexity-model Qwen/Qwen3-0.6B
uv run python scripts/vjol/audit_run.py --run-dir $V5
uv run python scripts/vjol/groundtruth.py sample --run-dir $V5 --n 400
uv run python scripts/vjol/groundtruth.py clean-set --run-dir $V5 --n 300
uv run python scripts/vjol/groundtruth.py check-clean --clean-set $V5/groundtruth/clean_set.jsonl --data-root data   # phải "0 bị gắn cờ nhầm"
uv run python scripts/vjol/check_garbled.py --data-root data
tar czf vjol_s17.tar.gz data/processed/ocr_bakeoff/syllables.json data/processed/ocr_bakeoff/unaccented.json data/processed/vjol_garbled_check \
    data/logs/vjol_extract_gpu*.out $V5/{manifest.json,04_quality.parquet,audit,groundtruth}
```

**Báo lại cho Claude**: giờ trích (hai dòng `date`), dòng `--status`, và gói `vjol_s17.tar.gz`. Claude so v5 với v4b (kỳ vọng: `lost_letters` về gần 0, `NCNN` 2003 hết band A rác, `tcdantochocvatongiaohoc` hết bất thường), rồi đo tỷ lệ âm tiết lạ cho họ lỗi đọc font `tvu` bằng `syllables.json`.

## S-18. VJOL: trích lại sau sửa J-33 và chạy `vjol_10k_v6`  [CHẠY SAU S-17]

**Mục đích**: J-34 code các việc A-H, J của J-33. Phía trích text đổi (`EXTRACT_VERSION` 4 -> 5: trang không lớp chữ được vẽ ra thử và OCR nếu có nội dung, trang trắng ghi `blank_page`, câu "There is no text in the image" bị lọc) nên TOÀN BỘ 2.937 checkpoint được làm lại; đọc lớp chữ rất nhanh, trang đã OCR nằm trong `interim/vjol_ocr_cache/` không OCR lại, chỉ trang `no_text_layer` mới (dự kiến vài trăm, phần lớn trong 302 trang `empty` của v5) cần GPU. Các sửa còn lại (khôi phục "ư", bỏ trang ban biên tập, cắt tiêu đề tạp chí ở mọi vị trí, năm 1970, ô bảng) chạy lúc sinh bản ghi nên không cần trích lại, nhưng `syllables.json` PHẢI có (thiếu thì bước khôi phục "ư" bị bỏ, log ghi cảnh báo). Dùng đúng `SAMPLE_DOCS=3000 SEED=0` như các lần trước.

```bash
cd /home/jovyan/dinh/vi-corpus-etl
git fetch origin && git checkout worktree-vjol-pipeline && git pull
source /home/jovyan/dinh/env.sh
uv sync --group curator --group viz --group ocr
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -c "from vi_corpus.common.ocr import PageOcr; PageOcr(device='cuda'); print('baseline_ok')" 2>&1 | tail -3   # phải in baseline_ok
uv run pytest -q                                   # kỳ vọng 291 test qua
nvidia-smi                                         # không còn tiến trình vLLM nào
ls -l data/processed/ocr_bakeoff/syllables.json    # phải có file này

# (1 phút, tuỳ chọn) J-33 (A): xem ảnh trang 244006_217600.pdf có thật sự in "-" thay cho "ư" không; gửi /tmp/p1.png và dòng in ra
uv run python - <<'PY'
import glob, pymupdf
path = glob.glob("data/raw/vjol/*/244006_217600.pdf")[0]
doc = pymupdf.open(path); page = doc[1]
page.get_pixmap(dpi=110).save("/tmp/p1.png")
print(path, "| font:", [f[3] for f in page.get_fonts()], "| mẫu lớp chữ:", page.get_text()[:300].replace("\n", " "))
PY

[ -d data/interim/vjol_text_s17_backup ] || cp -r data/interim/vjol_text data/interim/vjol_text_s17_backup   # sao lưu một lần
date; DATA_ROOT=data VLLM_BIN=$(which vllm) SAMPLE_DOCS=3000 ./scripts/vjol/run_extract_vlm.sh && date && \
V5=data/processed/pipeline_runs/vjol_10k_v5 && V6=data/processed/pipeline_runs/vjol_10k_v6 && \
uv run python scripts/vjol/extract.py --data-root data --status && \
uv run --group curator --group viz python scripts/run_pipeline.py --data-root data --mix vjol=10000 --total 10000 \
    --run-name vjol_10k_v6 --embed-device cuda --cluster-space umap10 --perplexity-model Qwen/Qwen3-0.6B && \
uv run python scripts/vjol/audit_run.py --run-dir $V5 ; \
uv run python scripts/vjol/audit_run.py --run-dir $V6 ; \
uv run python scripts/vjol/compare_runs.py --old $V5 --new $V6 | tee $V6/compare_v5_v6.txt && \
uv run python scripts/vjol/groundtruth.py sample --run-dir $V6 --n 400 && \
uv run python scripts/vjol/groundtruth.py clean-set --run-dir $V6 --n 300 && \
uv run python scripts/vjol/groundtruth.py check-clean --clean-set $V6/groundtruth/clean_set.jsonl --data-root data && \
uv run python scripts/vjol/check_garbled.py --data-root data && \
tar czf vjol_s18.tar.gz data/processed/ocr_bakeoff/syllables.json data/processed/vjol_garbled_check data/logs/vjol_extract_gpu*.out \
    $V6/{manifest.json,report.html,04_quality.parquet,05_dedup.parquet,audit,groundtruth,compare_v5_v6.txt} /tmp/p1.png
```

Dấu `;` sau hai lệnh `audit_run.py` là cố ý: `audit_run.py` thoát mã 1 khi còn nhóm tạp chí bị loại bất thường (không phải lỗi), và `&&` sẽ dừng cả chuỗi ở đó (lần chạy S-18 đầu bị đúng chuyện này, không ra `vjol_s18.tar.gz`). Với $V5: v5 có thể không còn `04_quality.parquet` hợp lệ thì lệnh đó báo lỗi mà không dừng chuỗi (lúc đó `compare_runs.py` sẽ báo thiếu audit của v5 và dừng; dùng số v5 ở J-32 để so).

**Báo lại cho Claude**: hai dòng `date`, dòng `--status` (phải có `no_text_layer`, `blank_page`, số bài không có chữ), gói `vjol_s18.tar.gz` (CÙNG MỘT lần chạy: `report.html`, `manifest.json`, `04_quality.parquet`, `audit/`, `groundtruth/` đều của v6), và cho biết lệnh nào lỗi (nếu có). **Chủ dự án**: gán nhãn cột `label` của `$V6/groundtruth/labeling_sample.csv` (J-33 I; nửa `tune` chỉnh ngưỡng, nửa `holdout` đo độ chính xác thật).

**Kỳ vọng so với v5** (J-33 "cách đo xong"): giữ 9.375 / 10.000 -> cao hơn (khoảng 385 đoạn `lost_letters` quay lại band A); `lost_letters` 387 -> gần 0; `year=1970` 330 đoạn -> 0 (`missing` tăng tương ứng); "There is no text" 4 đoạn lọt -> 0; `journal_header` 4,6% -> thấp hơn; số bài không vào ingest 25 -> chỉ còn bài thật sự trắng (log "VJOL: ... checkpoint không thành bản ghi"); `font_misread` trong `audit.md` cho biết có cần xử lý ê -> ấ / ờ (J-33 H, ngưỡng 0,5%); `dropped_rate_macro` so với `dropped_rate_micro` cho biết mẫu lệch tạp chí ảnh hưởng thế nào (J-33 J).

## S-19. VJOL: chạy lại pipeline `vjol_10k_v7` sau sửa J-35  [CHẠY SAU S-18]

**Mục đích**: J-35 sửa hai chỗ chạy lúc sinh bản ghi (khôi phục "ư" thêm dạng, cắt tiêu đề "Tạp chí X số N - năm"), nên KHÔNG trích lại, KHÔNG cần vLLM; chỉ chạy lại pipeline từ checkpoint của S-18. Không sinh lại mẫu gán nhãn (giữ `labeling_sample.csv` của v6 để chủ dự án gán nhãn tiếp).

```bash
cd /home/jovyan/dinh/vi-corpus-etl
git pull
source /home/jovyan/dinh/env.sh
uv sync --group curator --group viz --group ocr
uv run pytest -q                                   # kỳ vọng 293 test qua
nvidia-smi                                         # không còn tiến trình vLLM nào
V6=data/processed/pipeline_runs/vjol_10k_v6; V7=data/processed/pipeline_runs/vjol_10k_v7
date
uv run --group curator --group viz python scripts/run_pipeline.py --data-root data --mix vjol=10000 --total 10000 \
    --run-name vjol_10k_v7 --embed-device cuda --cluster-space umap10 --perplexity-model Qwen/Qwen3-0.6B
date
uv run python scripts/vjol/audit_run.py --run-dir $V7
uv run python scripts/vjol/compare_runs.py --old $V6 --new $V7 | tee $V7/compare_v6_v7.txt
tar czf vjol_s19.tar.gz $V7/{manifest.json,report.html,04_quality.parquet,audit,compare_v6_v7.txt}
```

Các lệnh viết từng dòng riêng (không nối `&&`) vì `audit_run.py` thoát mã 1 khi còn nhóm tạp chí bất thường.

**Báo lại cho Claude**: gói `vjol_s19.tar.gz` và cho biết lệnh nào lỗi. **Kỳ vọng so với v6**: `hyphen_lost` 1,75% -> khoảng 0,4% (phần còn lại là ký hiệu thật như `t- Statistics`), `journal_header` 4,42% -> khoảng 3,2%, giữ >= 9.768, `lost_letters` vẫn ~3.

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
| S-14 | | inspect: nguồn chữ đảo ngược (text / ocr_vlm); producer top: ; embed giây/640 đoạn và device: ; extract giờ: ; v3 so với v2: | |
| S-10 | | số file: ; dung lượng: GB; thời gian chép: ; số trang: ; tỉ lệ text / scan / tcvn3 theo năm: ; số dòng metadata: | |
| S-12 | | PDF trích / lỗi / bỏ qua: ; trang theo method: ; ocr_vlm ok / blank / flagged: ; trang baseline: ; giờ trích text: ; đoạn mỗi bài: ; ngôn ngữ vi / en / khác: | |
| S-11 | | số cặp A bắt: ; C bắt: ; chỉ C bắt: (cùng nguồn / khác nguồn); nhóm 2 (0,7–0,8): trung / cung-mau / khac = ; nhóm 1: ; nhóm 3: | |
