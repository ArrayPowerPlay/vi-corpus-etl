# Kiến trúc dự án vi-corpus-etl

Tài liệu mô tả **cấu trúc repo và cấu trúc dữ liệu**. Thiết kế các bước xử lý (parse → language → normalize → quality → dedup → knowledge unit) nằm ở [`docs/design/PIPELINE.md`](PIPELINE.md); danh mục nguồn ở [`docs/sources/SOURCES.md`](../sources/SOURCES.md); chiến lược giáo trình ở [`docs/sources/GIAO_TRINH.md`](../sources/GIAO_TRINH.md); các quyết định đã chốt ở [`docs/project/DECISION_LOG.md`](../project/DECISION_LOG.md).

Nguyên tắc chung:
- **Mỗi nguồn dữ liệu có gói code riêng (`vi_corpus/<nguồn>/`) và thư mục script riêng (`scripts/<nguồn>/`)**; phần dùng chung nằm ở `vi_corpus/common/`.
- **Raw-first**: `data/raw/` là bản gốc, không bao giờ bị sửa. Mọi bước xử lý đọc `raw/` và ghi ra `interim/` hoặc `processed/`.
- **Checkpoint theo đơn vị nhỏ** (file / cuốn sách), ghi nguyên tử; crash thì chạy lại đúng lệnh cũ.
- Một script tổng, `scripts/run_all.py`, chạy lần lượt script của từng nguồn.

## 1. Cây thư mục code

```
vi-corpus-etl/
├── CLAUDE.md                        # file .md duy nhất ở gốc repo; mọi tài liệu khác nằm trong docs/
├── Makefile                         # make count-ocr, make pipeline-cpu, make pipeline-progress (VJOL + VISTA, chỉ CPU)
├── pyproject.toml, uv.lock          # thư viện quản lý bằng uv (nhóm ocr: uv sync --group ocr; curator, viz)
├── .env                             # HF_TOKEN, khoá OCR nội bộ (không đưa lên git)
├── docs/
│   ├── README.md                    # hướng dẫn sử dụng từng phần + mục lục toàn bộ tài liệu
│   ├── project/                     # quản lý dự án: DECISION_LOG, PROGRESS, ROADMAP, TASKS, SERVER_TASK
│   ├── design/                      # thiết kế: PROJECT_ARCHITECTURE (file này), PIPELINE
│   ├── sources/                     # nguồn dữ liệu: SOURCES, SOURCE_CANDIDATES, GIAO_TRINH, VISTA, LUAN_AN, PERMISSION_EMAILS, VJOL_PIPELINE.html
│   ├── training/                    # đề xuất CPT + SFT: SUGGESTION (bản đầu), SUGGESTION_v2 (bản hiện hành)
│   └── diagrams/                    # SVG: 00 (luồng chính), 01..08 + 22, 23 (vẽ bằng scripts/draw_diagrams.py), 09..13 (luồng con khác), 14..18 (SUGGESTION), 19..21 (SUGGESTION_v2), 26 (phụ thuộc task)
├── configs/
│   ├── ocr_bakeoff/engines.json     # cấu hình engine OCR cho đợt so sánh (lệnh dựng server, lời nhắc, giấy phép)
│   └── benchmarks/                  # benchmark để quét nhiễm (hiện rỗng, K-01)
├── scripts/
│   ├── run_all.py                   # script tổng: chạy script con của từng nguồn
│   ├── run_pipeline.py              # pipeline chung (ingest → finalize) trên mẫu N hoặc cả kho (--take-all, --shard I/N)
│   ├── run_pipeline_cpu.sh          # chạy pipeline chung cho VJOL + VISTA chỉ bằng CPU, chia phần (gọi qua make pipeline-cpu)
│   ├── pipeline_progress.py         # tiến độ + giờ dự kiến xong của make pipeline-cpu
│   ├── dedup_global.py              # dedup vòng 2 toàn cục trên kho dấu vân tay
│   ├── cpt_pack.py                  # cắt + ghép segment CPT thành chuỗi 8.192 token
│   ├── make_judge_sample.py         # mẫu 50k phân tầng cho LLM chấm
│   ├── count_ocr_pages.py           # đếm trang cần OCR của VJOL / VISTA (make count-ocr)
│   ├── build_unaccented.py          # sinh danh sách từ không dấu từ kho chữ sạch
│   ├── draw_diagrams.py             # vẽ lại sơ đồ docs/diagrams/00..08, 22, 23
│   ├── bench_embed.py, head_clean.py  # đo tốc độ embed; xem đầu văn bản đã làm sạch
│   ├── sea/                         # download_all.py, download_sea_*.py (3 bộ), count_rows.py
│   ├── stbook/                      # crawl.py, ocr.py (Paddle + VietOCR), ocr_vlm.py + run_ocr_vlm.sh (chuỗi VLM, O-20), ocr_vlm_bench*, check_ocr_vlm.py, filter_ocr_pages.py (O-30)
│   ├── giao_trinh/                  # extract.py (xử lý thô giáo trình), survey.py (khảo sát)
│   ├── vjol/                        # extract.py + run_extract_vlm.sh (trích text), check_garbled*.py, audit_run.py, groundtruth.py, compare_runs.py, font_canary.py, plan_reocr.py, ...
│   ├── vista/                       # extract.py + run_extract_vlm.sh, run_vista.sh (cả đường đi), check_raw.py, producer_survey.py, summarize_runs.py
│   ├── anle/                        # download.py (án lệ từ HF, NS-03)
│   ├── ocr_bakeoff/                 # so sánh engine OCR (F-11): select_pages, build_syllables, run_engine, score, diagnose_gt, eval_guard, run_bakeoff.sh
│   ├── ocr_internal/                # thử OCR nội bộ VNPT trên mẫu VJOL + VISTA (OI-01): trial_vjol_vista.py, review_report.py
│   ├── source_probe/                # probe_hf.py: kiểm nhanh chất lượng nguồn ứng viên (NS-11)
│   ├── gpu_keepalive.py             # giữ GPU trên Run:ai không bị tự dừng khi job chỉ dùng CPU
│   └── run_with_gpu_keepalive.sh    # chạy một script bất kỳ kèm gpu_keepalive, chạy nền
├── vi_corpus/
│   ├── common/                      # dùng chung cho mọi nguồn
│   │   ├── registry.py, schema.py, state.py    # sổ đăng ký nguồn; schema + KPI truy vết; ghi JSON nguyên tử, khoá, log
│   │   ├── ocr.py, ocr_cascade.py, ocr_cache.py  # OCR một trang (Paddle + VietOCR); chuỗi VLM theo trang; bộ nhớ đệm OCR
│   │   ├── pdf_text.py, batch_extract.py, parsers/  # trích text PDF / pptx / docx / doc / ppt / djvu / html / epub; vòng xử lý theo lô có checkpoint
│   │   ├── garbled.py, defects.py, unaccented.py, unicode_clean.py, references.py, ocr_noise.py  # bộ dò chữ hỏng, khiếm khuyết, làm sạch
│   │   └── parquet_export.py
│   ├── sea/                         # nguồn SEA: datasets, hub, checkpoint, downloader, cli, reader
│   ├── stbook/                      # crawler/ (crawl PDF), ocr_books.py, ocr_vlm.py, ocr_parallel.py, ocr_check.py, ocr_filter.py
│   ├── giao_trinh/                  # extract.py (kiểm kê + trích text), records.py (bản ghi + Parquet)
│   ├── vjol/                        # bộ xử lý bài báo PDF dùng chung VJOL + VISTA: layout, extract, cli, records, clean, hyphen_fix, hyphen_lm, reocr, font_canary, audit, ...
│   ├── vista/                       # phần riêng của VISTA: layout, records, raw_check, producer_survey, summary
│   ├── pipeline/                    # pipeline chung: config, runner, ingest, normalize, lines, chunk, tokens, language, spans, quality, perplexity, dedup, dedup_global, embed, reduce, viz, knowledge, cpt_pack, contamination, audit, report, curator, judge_sample, io
│   ├── ocr_bakeoff/                 # so sánh engine OCR + cascade, guard, metrics, scoring
│   ├── source_probe/                # specs, sample, measure: kiểm nhanh nguồn ứng viên
│   └── __init__.py
├── third_party/                     # opencv-contrib-python-shim: gói giả để paddlex không kéo opencv-contrib-python (J-31)
├── reports/                         # báo cáo nghiên cứu (hiện: Knowledge injection cho SFT tiếng Việt.md)
└── tests/                           # ~413 test không cần mạng: uv run pytest; tests/data/ = ca thử vàng VJOL
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
│   ├── vjol/<năm>/<article_id>_<galley_id>.pdf, metadata/*.jsonl   # bài báo VJOL (chữ thường, J-02)
│   ├── vista/<năm>/<publication_id>.pdf, unknown/, metadata/*.jsonl  # bài báo VISTA (chép từ MinIO, VS-05)
│   └── anle/                              #   án lệ tải từ HF bằng scripts/anle/download.py (NS-03)
├── interim/                               # kết quả trung gian của từng nguồn
│   ├── stbook_ocr/<danh-mục>/<product_id>.json     # text từng trang + metadata sách (chuỗi VLM: thêm page_info, stats)
│   ├── stbook_ocr_baseline/<danh-mục>/             # bản kết quả của bộ OCR cũ, chép sang trước khi chuỗi VLM ghi đè
│   ├── giao_trinh_text/<sha256>.json               # text từng trang + method, khoá theo sha256 file
│   ├── vjol_text/<sha256>.json, vista_text/<sha256>.json  # checkpoint trích text bài báo (method, tier, status, extract_version)
│   └── vjol_ocr_cache/                             # bộ nhớ đệm OCR theo trang (dùng lại khi trích lại)
├── processed/                             # kết quả cuối của từng nguồn (theo schema chung)
│   ├── giao_trinh/giao_trinh.parquet, vjol/vjol.parquet, vista/vista.parquet
│   ├── pipeline_runs/<tên lần chạy>/        # 01_ingest … 07_reduced, clean, knowledge_units, cpt_segments, audit, manifest, report.html
│   ├── dedup_global/verdict.parquet         # kết quả dedup vòng 2
│   ├── ocr_count/, ocr_bakeoff/, source_probe/, ...  # kết quả các công cụ đo
├── state/                                 # checkpoint + khoá chống chạy trùng
│   ├── sea_instruct_2602/, sea_pile_v2/, sea_lion_pile_v1/   # <tên-file>.json + _manifest.json + .run.lock
│   ├── stbook_ocr/                        # .run.lock
│   ├── stbook_ocr_vlm/<worker>/, claims/  # .run.lock từng tiến trình + khoá nhận từng cuốn
│   ├── dedup_index/<nguồn>/<run>.parquet  # kho dấu vân tay dedup vòng 1
│   └── ...                                # giao_trinh, vjol_extract, vista_extract, ocr_count: do các bước tương ứng tạo
└── logs/<việc>_<ngày_giờ>.log             # log từng lần chạy
```

Ghi chú: state của SEA nằm ở `state/<bộ>/` (không nằm trong `raw/sea_vi/`) và chỉ lưu đường dẫn tương đối trong repo HF, kích thước, sha256, nên di chuyển `raw/` không làm mất checkpoint.

## 3. Luồng từng nguồn

| Nguồn | Tải về | raw | Bước xử lý | interim | processed |
|---|---|---|---|---|---|
| SEA | `scripts/sea/download_all.py` (Hugging Face, có checkpoint) | `raw/sea_vi/<bộ>/` | `vi_corpus/sea/reader.py` đọc thẳng về schema chung (đã là text, bỏ bước parse) | (không cần) | kế hoạch: các stage chung |
| stbook | `scripts/stbook/crawl.py` | `raw/stbook/` | `scripts/stbook/ocr_vlm.py` (chuỗi VLM Qwen3-VL-8B + bộ kiểm + OCR cũ dự phòng, O-20; `scripts/stbook/ocr.py` = bộ OCR cũ, checkpoint theo cuốn) | `interim/stbook_ocr/` | `ocr_books.iter_records`: mỗi cuốn một bản ghi |
| giáo trình | `rclone copy` từ Google Drive (chạy tay, xem `docs/sources/GIAO_TRINH.md`) | `raw/giao_trinh/` | `scripts/giao_trinh/extract.py`: kiểm kê sha256, trích text PDF/pptx, OCR trang scan (xuất Parquet tự động cuối lần chạy) | `interim/giao_trinh_text/<sha256>.json` | `processed/giao_trinh/giao_trinh.parquet` |
| VJOL | không crawl trong repo này | `raw/vjol/` | `scripts/vjol/extract.py`: PyMuPDF cho mọi trang; trang scan và trang lỗi font (J-10) OCR bằng chuỗi VLM (`run_extract_vlm.sh` cho 4 GPU); bài có lỗi nặng OCR lại cả bài; `--no-ocr` cho bản CPU (trang scan để `needs_ocr`); xem sơ đồ 22 | `interim/vjol_text/<sha256>.json` | `vjol/records.iter_records`: bỏ bìa / mục lục / tài liệu tham khảo / tiêu đề chạy, nối metadata tạp chí; `processed/vjol/vjol.parquet` |
| VISTA | không crawl trong repo này; đã chép từ MinIO về `raw/vista/` (VS-05) | `raw/vista/` | `scripts/vista/extract.py` (dùng chung bộ xử lý của VJOL; luật producer OCR đời cũ); `run_vista.sh` chạy cả đường đi | `interim/vista_text/<sha256>.json` | `vista/records.iter_records`: một bản ghi / PDF, `meta.publication_ids`; `processed/vista/vista.parquet` |

Sau khi mỗi nguồn ra bản ghi theo `vi_corpus/common/schema.py`, các stage chung (prepare, language, quality, dedup, finalize) xử lý giống nhau cho mọi nguồn (`scripts/run_pipeline.py`; thiết kế và sơ đồ ở `docs/design/PIPELINE.md`). Nguồn đã có adapter ingest: SEA (3 bộ), stbook, giáo trình, VJOL, VISTA; các nguồn đã chọn khác (`phap_luat`, `anle`, `fineweb2_vi`, `wiki_vi`, `finepdfs_*`, `luanvan_vnu`) chưa có adapter (xem `docs/sources/SOURCES.md`).

## 4. Vai trò từng gói

- **`vi_corpus/common/`**: không biết gì về một nguồn cụ thể. `registry.py` là nơi duy nhất khai báo đường dẫn raw của từng nguồn (`raw/sea_vi/...`, `raw/stbook`, `raw/vjol`, ...) và thông tin owner/license/rights; `state.py` cung cấp cơ chế checkpoint/khoá/log mà mọi bước chạy lâu dùng chung.
- **`vi_corpus/sea/`**: tải phần tiếng Việt của 3 bộ SEA và đọc lại về schema chung. Đường dẫn raw dựng trong `downloader.DatasetPaths` (`raw/sea_vi/<bộ>`), state ở `state/<bộ>`.
- **`vi_corpus/stbook/`**: crawler lấy sách từ stbook.vn (chép từ repo `stbook-crawler`, mặc định ghi vào `raw/stbook`) `ocr_books.py` OCR PDF dạng ảnh bằng Paddle + VietOCR và `ocr_vlm.py` OCR bằng chuỗi VLM (engine chính từ O-20).
- **`vi_corpus/giao_trinh/`**: kiểm kê file theo sha256, trích text theo định dạng, làm sạch theo trang (`clean_pages` khi sinh bản ghi), sinh bản ghi schema chung (chưa phân loại sách/slide, ngôn ngữ, chất lượng). Đầu ra: `interim/giao_trinh_text/<sha256>.json` và `processed/giao_trinh/giao_trinh.parquet`.
- **`vi_corpus/vjol/`** là bộ xử lý bài báo PDF dùng chung (`layout.py`, `records.py`, `cli.py`, `extract.py`, `clean.py` ...); **`vi_corpus/vista/`** chỉ chứa phần riêng của VISTA (bố cục, nối metadata, kiểm bản chép).

## 5. `scripts/run_all.py` nối các phần

`run_all.py` chỉ có một bảng `PARTS` (phần → thư mục raw cần có, script) và chạy từng script bằng tiến trình con, truyền `--data-root`:

| Phần | Script | Điều kiện chạy |
|---|---|---|
| `sea` | `scripts/sea/download_all.py` | luôn chạy (là bước tải, tự tạo `raw/sea_vi`) |
| `stbook` | `scripts/stbook/ocr.py` | có `raw/stbook` |
| `giao_trinh` | `scripts/giao_trinh/extract.py` | có `raw/giao_trinh` |
| `vjol` | `scripts/vjol/extract.py` | có `raw/vjol` (xử lý MỌI PDF với tham số mặc định) |
| `vista` | `scripts/vista/extract.py` | có `raw/vista` |

```bash
uv run python scripts/run_all.py --data-root /duong/dan/data                    # cả 5 phần theo thứ tự trên
uv run python scripts/run_all.py --data-root /duong/dan/data --only sea,giao_trinh
uv run python scripts/run_all.py --data-root /duong/dan/data --status           # chuyển --status cho từng script
```

- Phần nào thiếu thư mục raw thì bị bỏ qua và in một dòng log nói rõ.
- Một phần lỗi không làm dừng các phần sau; cuối cùng in tóm tắt (`sea: ok | stbook: bỏ qua | giao_trinh: lỗi (mã 1)`) và thoát với mã khác 0 nếu có phần lỗi.
- Muốn chạy nền trên Run:ai: `./scripts/run_with_gpu_keepalive.sh scripts/run_all.py --data-root /duong/dan/data`.
- Tham số riêng của từng script (ví dụ `--device cpu` của OCR, `--workers` của SEA) không đi qua `run_all`; cần thì chạy trực tiếp script của phần đó.
- Lưu ý: phần `stbook` của `run_all` vẫn gọi `scripts/stbook/ocr.py` (bộ OCR cũ Paddle + VietOCR), không phải chuỗi VLM đang là engine chính (O-20); chạy `scripts/stbook/ocr_vlm.py` / `run_ocr_vlm.sh` trực tiếp. Pipeline chung (`run_pipeline.py`) không nằm trong `run_all`.
- Thêm nguồn mới: tạo `vi_corpus/<nguồn>/`, `scripts/<nguồn>/`, thêm một dòng vào `PARTS` và một `SourceSpec` vào `registry.py`.
