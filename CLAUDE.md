# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

`vi-corpus-etl` (renamed from `sea-vi-crawler`) builds a Vietnamese LLM corpus from several sources: the AI Singapore SEA datasets (Hugging Face), datasets shared via a Google Drive folder, and VISTA/VJOL scientific papers (PDFs that are only **processed**, never crawled here; their paths are placeholders in config until the data arrives). Target metrics come from the weekly goals table: >=300M clean tokens, >=95% source lineage, exact+fuzzy dedup, knowledge units for CPT/SFT/Hybrid.

**Current state**: only the download step exists in code (`vi_corpus/download/`). The processing pipeline (ingest -> parse -> language -> normalize -> quality -> dedup -> knowledge unit -> audit) is a design: see `docs/PIPELINE.md`, `docs/SOURCES.md`, `docs/ROADMAP.md`. Decisions already made with the user: hybrid architecture (own disk-chained Parquet stages plus optional NeMo Curator adapters), **no train/val/test split at the corpus level** (split later at knowledge-unit level, by dedup family), repo/package name `vi-corpus-etl` / `vi_corpus`. Google Drive folder contents were not readable yet (connector failed), so the Drive source is generic.

### SEA download step
Downloads **only the Vietnamese partition** of three AI Singapore datasets from Hugging Face, keeping the original files byte-for-byte, with per-file checkpointing so a crashed run resumes by re-running the same command. It is meant to run on a remote Jupyter Lab server (Linux), launched from the Jupyter Terminal with `nohup`/`tmux`, not from notebook cells. Processing must read from `data/raw/` and write to a separate folder, never re-download.

| key | repo | Vietnamese dir | size |
|---|---|---|---|
| `sea_instruct_2602` | `aisingapore/SEA-Instruct-2602` (gated) | `Vietnamese/` | 12 parquet, ~2.7 GB |
| `sea_pile_v2` | `aisingapore/SEA-PILE-v2` | `vi/` | 253 parquet, ~132 GB |
| `sea_lion_pile_v1` | `aisingapore/SEA-PILE-v1` | `sea-pile-mc4/vi/` | 329 jsonl.gz, ~107 GB |

## Commands

```bash
uv sync                                   # install deps from uv.lock (dev group: pytest, pyarrow for inspecting parquet)
uv run pytest                             # all tests (no network)
uv run pytest tests/test_checkpoint.py::test_is_done_checks_size_and_sha   # single test
uv add <pkg> / uv add --dev <pkg>         # add a dependency; commit pyproject.toml + uv.lock together

# real end-to-end smoke test (needs HF_TOKEN in .env); keep test data out of the repo
uv run python scripts/download_sea_instruct_2602.py --data-root <scratch>/data --limit-files 1 --verify-sha256
uv run python scripts/download_all.py --data-root <scratch>/data --status
```
All four scripts share the flags `--data-root` (default: env `SEA_DATA_ROOT`, otherwise `./data`), `--workers` (8), `--limit-files`, `--verify-sha256`, `--max-retries` (5), and `--status`. Dependencies are managed with uv only (`pyproject.toml` + `uv.lock`); there is no requirements.txt. The project is not an installable package (repo/package names: `vi-corpus-etl` / `vi_corpus`): pytest gets `pythonpath = .` from `[tool.pytest.ini_options]`. The scripts add the repo root to `sys.path` themselves.

## Architecture

The flow goes `scripts/*.py` → `vi_corpus.download.cli.main(keys)` → `vi_corpus.download.downloader.run_dataset(key)` for each key, run sequentially. `download_all.py` iterates `DATASETS` in dict order, so the small dataset runs first.

- `vi_corpus/download/datasets.py`: the `DATASETS` registry of `DatasetSpec(key, repo_id, vi_dir)`. To add a dataset, add an entry here and write a new thin script.
- `hub.py` (same package): loads `HF_TOKEN` from the repo-root `.env` via python-dotenv. An existing environment variable wins. It also lists files through `list_repo_tree`, which provides size and the LFS sha256. `with_retry` does exponential backoff and skips retries for config errors (repo/revision/entry not found, 401/403/404). `download_file` wraps `hf_hub_download(local_dir=raw/<key>)` and checks size (plus sha256 when the flag is on); a mismatch deletes the file and retries.
- `checkpoint.py`: writes one JSON state file per remote file in `state/<key>/` (`/` in the path becomes `__`), always atomically (tmp → fsync → `os.replace`). `_manifest.json` holds the file list of the **last** run, so `--status` works offline. After a `--limit-files` run, it shows only that subset.
- `downloader.py`: the order inside `run_dataset` matters.
  1. `acquire_run_lock`: `fcntl.flock` on `state/<key>/.run.lock`, which blocks a duplicate run of the same dataset.
  2. `cleanup_incomplete`: deletes `raw/<key>/.cache/huggingface/download/**/*.incomplete`. This is only safe because the lock is already held.
  3. Resolve and pin one revision sha for the whole run.
  4. Skip a file only when `_is_complete` is true: state is `done`, size and sha still match HF, **and** the file on disk has the right size.
  5. Download with a `ThreadPoolExecutor` and mark each file done right away. A failure in one file is recorded as `failed` and does not stop the others.

  Ctrl+C calls `os._exit(130)` on purpose, so it does not wait for in-flight downloads.

### Non-obvious constraints
- huggingface_hub 1.x **does not resume partial downloads**: each attempt writes to a uniquely named `*.incomplete`. The checkpoint unit is therefore the whole file, and a crash costs at most `--workers` in-flight files. Do not promise byte-level resume.
- Keep `raw/<key>/` mirroring the HF path layout, and keep the hidden `.cache/huggingface/` folder that HF writes there. Paths are relative to `--data-root`, so a data folder can be moved to another disk and resumed.
- In SEA-Instruct-2602, `conversations` is a Python-repr **string** (single quotes, `None`). Parse it with `ast.literal_eval`, not `json.loads`.
- `SEA-PILE-v1` is the repo name for what the user calls "SEA-LION-Pile v1". Only its mC4 portion is on HF.

## Conventions (from the user's global instructions)
- Every file starts with a header docstring, and every class and function has a docstring. Comments, docstrings, log messages, and README are written in **Vietnamese** with full diacritics.
- Commit messages must **not** include a Claude co-author line.
- For bug fixes, first reproduce the bug end-to-end against real HF data (small `--limit-files` into a scratch `--data-root`, `kill -9` mid-run to test resume).
