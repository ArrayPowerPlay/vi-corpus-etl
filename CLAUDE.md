# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

`vi-corpus-etl` (renamed from `sea-vi-crawler`) builds a Vietnamese LLM corpus from several sources (see "Data sources"). Target metrics come from the weekly goals table: >=300M clean tokens, >=95% source lineage, exact+fuzzy dedup, knowledge units for CPT/SFT/Hybrid.

**Current state**: code exists per source: SEA download (`vi_corpus/sea/`), stbook crawl + OCR (`vi_corpus/stbook/`), giáo trình raw processing (`vi_corpus/giao_trinh/`, implemented: inventory, extract, page clean, records; strategy in `docs/GIAO_TRINH.md`). The shared processing pipeline (ingest -> parse -> language -> normalize -> quality -> dedup -> knowledge unit -> audit) is a design: see `docs/PIPELINE.md`, `docs/SOURCES.md`, `docs/ROADMAP.md`. Repo layout and data layout: `PROJECT_ARCHITECTURE.md` (Vietnamese). Decisions already made with the user: hybrid architecture (own disk-chained Parquet stages plus optional NeMo Curator adapters), **no train/val/test split at the corpus level** (split later at knowledge-unit level, by dedup family), repo/package name `vi-corpus-etl` / `vi_corpus`, one folder of scripts per data source plus one orchestrator (`scripts/run_all.py`).

## Data sources

Data root (`--data-root`, default env `SEA_DATA_ROOT`, otherwise `./data`) has `raw/` (untouched originals), `interim/`, `processed/`, `state/` (checkpoints, run locks), `logs/`. Raw layout:

```
data/raw/
├── sea_vi/{sea_instruct_2602, sea_pile_v2, sea_lion_pile_v1}/   # SEA, mirrors the HF path layout
├── stbook/<slug>/{books.json, content/<product_id>.pdf}         # stbook-crawler layout
├── giao_trinh/<ngành>/<môn>/<file>                              # rclone copy of the Drive folder
└── VJOL/                                                        # structure TBD
```

| source | what | raw path | status |
|---|---|---|---|
| SEA (AI Singapore, HF) | only the Vietnamese partition of 3 datasets, byte-for-byte | `raw/sea_vi/<key>/` | download done, processing = design |
| stbook.vn | books as image-only PDFs, OCR'd here | `raw/stbook/` | crawl + OCR code exists |
| giao_trinh | Google Drive university textbooks, PDF/pptx | `raw/giao_trinh/` | download by rclone; `scripts/giao_trinh/extract.py` |
| VJOL | scientific papers (PDF), only **processed**, never crawled here | `raw/VJOL/` (registry key `vjol`) | placeholder, structure will be given later; no scripts yet |
| VISTA | scientific papers (PDF), same as VJOL | placeholder path in registry | placeholder |

- **SEA**: AI Singapore datasets on Hugging Face.

  | key | repo | Vietnamese dir | size |
  |---|---|---|---|
  | `sea_instruct_2602` | `aisingapore/SEA-Instruct-2602` (gated) | `Vietnamese/` | 12 parquet, ~2.7 GB |
  | `sea_pile_v2` | `aisingapore/SEA-PILE-v2` | `vi/` | 253 parquet, ~132 GB |
  | `sea_lion_pile_v1` | `aisingapore/SEA-PILE-v1` | `sea-pile-mc4/vi/` | 329 jsonl.gz, ~107 GB |

  Runs on a remote Jupyter Lab server (Linux), launched from the Jupyter Terminal with `nohup`/`tmux`, not from notebook cells. Processing reads `raw/sea_vi/` and writes elsewhere, never re-downloads. `SEA-PILE-v1` is the repo name for what the user calls "SEA-LION-Pile v1"; only its mC4 portion is on HF.
- **stbook**: books are downloaded by `scripts/stbook/crawl.py` (code copied from the old repo `stbook-crawler`; that repo's `data/` folder can also be copied/symlinked unchanged into `raw/stbook/`). `<id>_pages/` = book still downloading, ignored. PDFs are image-only (one JPEG per page), so `run_ocr` OCRs each book with PaddleOCR detection (`PP-OCRv5_mobile_det`) + VietOCR recognition (`vgg_transformer`) and writes `interim/stbook_ocr/<slug>/<product_id>.json` atomically. Checkpoint per book, redone if the PDF size changes. `iter_records` turns those JSONs into one schema record per book. Rights: publisher copyright, `unknown` -> quarantine.
- **giao_trinh**: Google Drive folder "Tổng hợp giáo trình Đại học", id `1DRUL5cFHIusB3PoLXJ6tcpuSNJjfZUTt`, 16 ngành, >=1,498 files (mostly PDF, some pptx/ppt/docx/djvu). Nearly all entries are Drive **shortcuts** to other people's files, so download with:
  `rclone copy gdrive: data/raw/giao_trinh --drive-root-folder-id 1DRUL5cFHIusB3PoLXJ6tcpuSNJjfZUTt --transfers 4 --drive-acknowledge-abuse`
  `gdown` does not work (access denied, 50-files-per-folder cap). English books are **kept** (user decision). Rights status unknown. Strategy: `docs/GIAO_TRINH.md`. Processing command: `uv run --group ocr python scripts/giao_trinh/extract.py --data-root <root> [--device cuda|cpu] [--no-ocr] [--limit-files N] [--status]` (`scripts/run_all.py` calls it with `--data-root` [`--status`]). Per-file checkpoint keyed by file sha256 in `interim/giao_trinh_text/<sha256>.json`; the parquet `processed/giao_trinh/giao_trinh.parquet` is exported automatically at the end of every run; exit code 1 if any file errored. Raw processing only (no language/quality/dedup yet); real OCR is untested locally (needs the GPU server).
- **VJOL / VISTA**: scientific papers, PDFs that are only processed (paths are placeholders until the data arrives). Do not write VJOL/VISTA scripts before the structure is known.

## Commands

```bash
uv sync                                   # install deps from uv.lock (dev group: pytest, pyarrow for inspecting parquet)
uv run pytest                             # all tests (no network)
uv run pytest tests/test_checkpoint.py::test_is_done_checks_size_and_sha   # single test
uv add <pkg> / uv add --dev <pkg>         # add a dependency; commit pyproject.toml + uv.lock together

# run every part in sequence (subprocess per part, --data-root passed through)
uv run python scripts/run_all.py --data-root <root>                       # sea -> stbook -> giao_trinh
uv run python scripts/run_all.py --data-root <root> --only sea,giao_trinh # choose parts
uv run python scripts/run_all.py --data-root <root> --status              # forwarded to each part
```
`run_all.py` keeps going when a part fails and exits non-zero with a summary. It skips `stbook` / `giao_trinh` when their raw folder does not exist (log line says so); `sea` is a download step, so it is never skipped.

```bash
# per-part scripts (one folder per source under scripts/)
uv run python scripts/sea/download_sea_instruct_2602.py --data-root <scratch>/data --limit-files 1 --verify-sha256   # real smoke test, needs HF_TOKEN in .env; keep test data out of the repo
uv run python scripts/sea/download_all.py --data-root <scratch>/data --status
uv run python scripts/sea/count_rows.py --data-root <root>
uv sync --group ocr                       # OCR deps (paddleocr, paddlepaddle CPU, vietocr, torch, pymupdf); not installed by plain `uv sync`
uv run python scripts/stbook/ocr.py --data-root <scratch>/data --limit-books 1 --device cpu   # needs <scratch>/data/raw/stbook
uv run python scripts/stbook/crawl.py --download-pdf                      # writes <data-root>/raw/stbook
```
The four SEA scripts share the flags `--data-root`, `--workers` (8), `--limit-files`, `--verify-sha256`, `--max-retries` (5), and `--status`. `scripts/gpu_keepalive.py` and `scripts/run_with_gpu_keepalive.sh` (keep a Run:ai GPU workload alive during CPU-only jobs) are not tied to a source and wrap any script, e.g. `./scripts/run_with_gpu_keepalive.sh scripts/run_all.py --data-root <root>`. Dependencies are managed with uv only (`pyproject.toml` + `uv.lock`); there is no requirements.txt. The project is not an installable package: pytest gets `pythonpath = .` from `[tool.pytest.ini_options]`. Scripts add the repo root to `sys.path` themselves.

## Architecture

Packages: `vi_corpus/common/` (registry, schema, `state.py` = atomic JSON + run lock + logging, `ocr.py`, `pdf_text.py`), `vi_corpus/sea/`, `vi_corpus/stbook/` (`ocr_books.py`, `crawler/`), `vi_corpus/giao_trinh/`. `vista/` and `vjol/` are empty placeholders.

SEA flow: `scripts/sea/*.py` → `vi_corpus.sea.cli.main(keys)` → `vi_corpus.sea.downloader.run_dataset(key)` for each key, run sequentially. `download_all.py` iterates `DATASETS` in dict order, so the small dataset runs first.

- `vi_corpus/sea/datasets.py`: the `DATASETS` registry of `DatasetSpec(key, repo_id, vi_dir)`. To add a dataset, add an entry here and write a new thin script.
- `hub.py`: loads `HF_TOKEN` from the repo-root `.env` (`REPO_ROOT` = `parents[2]`) via python-dotenv; an existing environment variable wins. Lists files through `list_repo_tree` (size + LFS sha256). `with_retry` does exponential backoff and skips retries for config errors (repo/revision/entry not found, 401/403/404). `download_file` wraps `hf_hub_download(local_dir=raw/sea_vi/<key>)` and checks size (plus sha256 when the flag is on); a mismatch deletes the file and retries.
- `checkpoint.py`: one JSON state file per remote file in `state/<key>/` (`/` in the path becomes `__`), written atomically. State stores only **relative** repo paths, sizes and sha256, so moving `raw/` never invalidates it. `_manifest.json` holds the file list of the **last** run, so `--status` works offline (after a `--limit-files` run it shows only that subset).
- `downloader.py`: the order inside `run_dataset` matters.
  1. `acquire_run_lock`: `fcntl.flock` on `state/<key>/.run.lock`, which blocks a duplicate run of the same dataset.
  2. `cleanup_incomplete`: deletes `raw/sea_vi/<key>/.cache/huggingface/download/**/*.incomplete`. Only safe because the lock is already held.
  3. Resolve and pin one revision sha for the whole run.
  4. Skip a file only when `_is_complete` is true: state is `done`, size and sha still match HF, **and** the file on disk (`raw/sea_vi/<key>/<path>`) has the right size.
  5. Download with a `ThreadPoolExecutor` and mark each file done right away. A failure in one file is recorded as `failed` and does not stop the others.

  Ctrl+C calls `os._exit(130)` on purpose, so it does not wait for in-flight downloads.

### Non-obvious constraints
- huggingface_hub 1.x **does not resume partial downloads**: each attempt writes to a uniquely named `*.incomplete`. The checkpoint unit is therefore the whole file, and a crash costs at most `--workers` in-flight files. Do not promise byte-level resume.
- Keep `raw/sea_vi/<key>/` mirroring the HF path layout, and keep the hidden `.cache/huggingface/` folder that HF writes there. Paths are relative to `--data-root`, so a data folder can be moved to another disk and resumed. SEA data downloaded under the old layout (`raw/<key>/`, before `sea_vi/` existed) migrates with `mkdir -p raw/sea_vi && mv raw/sea_instruct_2602 raw/sea_pile_v2 raw/sea_lion_pile_v1 raw/sea_vi/`; `state/<key>/` stays where it is and nothing is re-downloaded (verified end to end).
- In SEA-Instruct-2602, `conversations` is a Python-repr **string** (single quotes, `None`). Parse it with `ast.literal_eval`, not `json.loads`.
- OCR quirks, all verified on real stbook pages: PaddleOCR's own recognizer drops Vietnamese diacritics, so only its detector is used. Crops need `PADDING` px or VietOCR hallucinates words at line ends. vietocr still calls `Image.ANTIALIAS` (shimmed in `PageOcr`). gdown needs `pkg_resources`, hence `setuptools<81`. paddle 3.3 on CPU needs `enable_mkldnn=False`.
- A PDF truncated by a killed crawler still opens (PyMuPDF repairs it); `pdf_page_images` rejects it via `doc.is_repaired`.
- Memory: never load a whole parquet/jsonl into RAM on small machines; use pyarrow metadata / `iter_batches`.

## Conventions (from the user's global instructions)
- Every file starts with a header docstring, and every class and function has a docstring. Comments, docstrings, log messages, and README are written in **Vietnamese** with full diacritics.
- Commit messages must **not** include a Claude co-author line.
- For bug fixes, first reproduce the bug end-to-end against real data in a scratch `--data-root`, with `kill -9` mid-run to test resume. For downloads use a small `--limit-files`; for OCR build a few real stbook PDFs with `vi_corpus/stbook/crawler` (`download_book_pages(..., num_pages=8)` + `assemble_pdf`).
