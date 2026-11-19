# Lộ trình (theo bảng mục tiêu tuần)

> Bảng mục tiêu đầy đủ W1–W8 và trạng thái từng mục nằm ở [`PROGRESS.md`](PROGRESS.md) (cập nhật 2026-11-19). File này chỉ tóm tắt lộ trình và thứ tự triển khai code. Việc cần chốt / đo / code cho pipeline CPT + SFT: [`TASKS.md`](TASKS.md); việc phải chạy trên server: [`SERVER_TASK.md`](SERVER_TASK.md).

| Tuần | Trọng tâm | Mục tiêu chính | Việc trong repo | Trạng thái (2026-11-19) |
|---|---|---|---|---|
| W1 | Kiểm kê nguồn + thiết kế ingestion | ≥20 nguồn ứng viên (sách / giáo trình, VJOL / VISTA, social, coding, standards); mỗi nguồn có owner, license, domain; clean corpus schema; thiết kế NeMo Curator; knowledge unit + instruction schema; ma trận CPT / SFT / Hybrid | `docs/sources/SOURCES.md`, `docs/sources/SOURCE_CANDIDATES.md`, `docs/design/PIPELINE.md`, source registry | 🟡 gần xong |
| W2 | Ingestion PoC | ≥10 nguồn giá trị cao, ≥3 domain học thuật; ≥1B raw token; ≥300M clean token; parse → language → boilerplate → exact / fuzzy dedup; ≥100k knowledge candidates; audit chất lượng + contamination scan; ≥95% lineage | stage `ingest` … `dedup`, `finalize`, audit, `scripts/vjol`, `scripts/vista`, `scripts/stbook`, `scripts/giao_trinh` | 🟡 đang làm |
| W3 | Filtering pipeline v1 | thêm semantic dedup, quét nhiễm, heuristic chất lượng; đường cong retention | `quality.py`, `contamination.py` | 🟡 một phần |
| W4 | Quality model | gán nhãn ≥50k, bộ lọc chất lượng v1, embedding map v1 | `make_judge_sample.py`, `embed` / `reduce` | ⬜ |
| W5 | Knowledge atomization | tách section → fact / concept; ≥2M instruction | thiết kế `docs/training/SUGGESTION_v2.md` | ⬜ |
| W6 | Mini-model injection | CPT vs SFT vs Hybrid; VMLU, lịch sử, địa lý | `cpt_segments`, `cpt_pack`, `docs/project/TASKS.md` (E-06..E-11) | ⬜ |
| W7 | Scale experiment | ≥10B clean token; tối ưu throughput | `--shard`, `make pipeline-cpu`, Ray | ⬜ |
| W8 | Release + decision | dataset có version, gói Knowledge SFT, go / no-go | — | ⬜ |

## Thứ tự triển khai code (đã đi tới đâu)

1. ✅ Đổi tên + tài liệu.
2. ✅ Source registry + schema + adapter SEA.
3. ✅ language / normalize / quality / dedup (2 vòng), knowledge unit, audit, report, bản đồ embedding.
4. ✅ Adapter stbook (OCR chuỗi VLM), giáo trình, VJOL, VISTA (trích text + bản ghi). 🟡 OCR GPU các trang `needs_ocr` của VJOL / VISTA và chạy cả kho còn chờ.
5. ✅ `cpt_segments` + `scripts/cpt_pack.py`; 🟡 `cpt_mix`, bộ chấm chất lượng bằng mô hình, quét nhiễm theo từng câu (TASKS L-01, L-05, L-06) chưa làm.
6. ⬜ Adapter ingest cho các nguồn đã chọn nhưng chưa có code: `phap_luat`, `anle`, `wiki_vi`, `fineweb2_vi`, `finepdfs_*`, `luanvan_vnu`.

Ghi chú: ảnh mục tiêu gốc còn cột "Acceptance / KPI" bị cắt ở bên phải ở các tuần đầu; số W3 (raw ≥30B / clean ≥15B) chưa xác nhận (R-10, TASKS K-08).
