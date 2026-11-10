# SUGGESTION V2: pipeline CPT + SFT nhẹ GPU, có chống quên

> **Lưu ý quy mô (T-10, 2026-11-04):** dữ liệu thu thập thật lên tới **hàng tỷ token**; "300M token" chỉ là mốc KPI W2 (mức sàn). Mọi con số trong tài liệu này tính trên corpus ~300M (GPU-giờ, số lần lặp, tỉ lệ replay, mở rộng từ vựng, nạp tri thức) phải xem lại theo quy mô thật. Chữ `SUGGESTION_V2` trong tài liệu này chỉ bản cũ (nay là `docs/SUGGESTION_V2_300M.md`); bản tổng hợp mới là `docs/SUGGESTION_v2.md`.


> **Trạng thái: đề xuất, chưa phải quyết định.** Thay cho phần "làm dữ liệu" (Tầng 0) và "huấn luyện" (Tầng 2, 3) của `docs/SUGGESTION.md`; phần thuật ngữ vẫn xem ở đó (mục 2).
> Lý do viết lại: sếp phản biện (2026-10-30) rằng bản cũ dùng model lớn ở gần mọi bước (trích sự kiện, sinh biến thể, Active Reading) nên rất lâu mới có dữ liệu, "thà CPT còn nhanh hơn", và hỏi đã tính catastrophic forgetting chưa.
> Ràng buộc chủ dự án chốt trong phiên này (T-01 ... T-04 trong `docs/DECISION_LOG.md`): chưa có ngân sách GPU cố định (nên đề xuất theo mức, có giá GPU-giờ); **chỉ dùng 4× A100 40GB nội bộ, không API model đóng**; so sánh công bằng CPT với cách khác rồi mới chọn; bắt buộc **không tụt khả năng làm theo chỉ thị / chat và lập luận / toán / code** (tiếng Anh không quan trọng); RAG do Claude chốt.
> Số GPU-giờ trong tài liệu này là **ước tính của tôi** (công thức 6ND và số đo trong các bài báo), chưa đo trên server. Ước tính tính cho model ~8B; nếu đích là ~27B (R-21) thì xem mục 8.

## 1. Tóm tắt một phút

**Sếp đúng về chi phí.** Bản cũ tốn GPU vì nó **sinh ra rất nhiều chữ**: mỗi sự kiện 3–5 biến thể + ~10 hỏi–đáp, Active Reading ~4 tỷ từ cho mỗi base. Chi phí sinh dữ liệu tỷ lệ với **số token sinh ra**, không phải số token đọc vào. Ước tính thô: bản cũ cần **~3.000 GPU-giờ (~1 tháng trên 4 GPU)**, trong khi một lượt CPT 300M token cho model 8B chỉ khoảng **45 GPU-giờ (~12 giờ)**.

**Nhưng CPT thô một mình có hai điểm yếu đã đo được:**
1. Nạp sự kiện kém: CPT thô trên tài liệu nhỏ có khi còn **thấp hơn** model gốc (EntiGraph: 38,15% so với base 39,49%; Mix-CPT: QA Wiki 26,10 → 22,20).
2. Gây quên mạnh nếu làm sai cách: CPT thô làm điểm Code của Llama3-8B tụt **58,75 → 15,56** (Mix-CPT); CPT thẳng lên model instruct làm MT-Bench tụt **7,89 → ~1,0** (FinDAP).

**Đề xuất mới: lấy CPT làm trục chính, vá hai điểm yếu bằng cách gần như không tốn GPU:**

```
Bước 0  Đo base (Known/Unknown, điểm gốc)                          ~10 GPU-giờ
Bước 1  Làm hỗn hợp CPT: corpus sạch + replay toán/code/chung      CPU, không model sinh
        + dữ liệu chỉ thị có sẵn + bài đọc hiểu sinh bằng regex
Bước 2  CPT full trên BASE (không phải instruct), LR thấp           ~90 GPU-giờ
Bước 3  Cộng "chat vector" (instruct − base) vào model đã CPT       CPU, vài phút
        rồi SFT ngắn bằng LoRA                                       ~5–10 GPU-giờ
Bước 4  RAG trên kho chunk đã có, cho sự kiện hiếm                  dùng lại embedding
Bước 5  Cổng eval ở MỌI checkpoint (IFEval, GSM8K, HumanEval, VMLU, Eval A/B)
```

Tổng mức rẻ nhất: **~120–150 GPU-giờ, khoảng 1,5–2 ngày trên 4 GPU**, nhỏ hơn bản cũ cỡ 20 lần. Sinh dữ liệu bằng model chỉ thêm vào **sau khi** đo thấy thiếu, và dùng model nhỏ (mục 6).

**Về catastrophic forgetting:** bản cũ có tính (LoRA, replay, đo từng checkpoint, không CPT lên instruct), nhưng chỉ cho đường SFT. Bản mới lấy CPT làm trục nên chống quên trở thành phần thiết kế chính, gồm 7 lớp có bằng chứng (mục 4).

## 2. So chi phí: bản cũ và bản mới

Hai con số gốc dùng để ước tính:

- **Sinh chữ bằng model 7–8B trên một A100: ~3M token/giờ.** WRAP đo đúng con số này với Mistral-7B. Số đo của chính dự án cũng khớp: OCR VLM 8B chạy ~1,55 trang/giây mỗi GPU (O-22), với ~500 token mỗi trang thì ra ~2,8M token/giờ.
- **Huấn luyện full model 8B: ~15 GPU-giờ cho 100M token.** Tính theo 6ND với MFU 35% ra 12 GPU-giờ; FinPythia (6,9B trên 8× A100) đo được 3.456 GPU-giờ cho 24B token, tức ~14 GPU-giờ cho 100M token.

| Bước của bản cũ | Token sinh ra (ước tính) | GPU-giờ |
|---|---|---|
| Trích sự kiện nhiều lượt trên 300M token | ~300M | ~100 |
| Biến thể + hỏi–đáp (×15 mỗi sự kiện) | 3–4,5 tỷ | 1.000–1.500 |
| Active Reading (bài gốc ~4 tỷ từ, **làm lại cho mỗi base**) | ~5 tỷ | ~1.700 |
| RAFT, lọc, SFT trên vài tỷ token | | vài trăm |
| **Tổng** | | **~3.000 (~1 tháng trên 4 GPU)** |

| Bước của bản mới (mức A) | GPU-giờ |
|---|---|
| Bước 0: đo 2–3 base | ~10 |
| Bước 1: làm hỗn hợp (CPU) | 0 |
| Bước 2: CPT ~600M token (corpus 300M, nguồn tốt lặp 2 lần, replay 25%, chỉ thị ~10%) | ~90 |
| Bước 3: merge (CPU) + SFT LoRA ~30–50M token | ~5–10 |
| Bước 5: eval ~6 checkpoint | ~12 |
| **Tổng** | **~120–150 (~1,5–2 ngày)** |

Lưu ý: số sinh chữ 3M token/giờ là của năm 2024; vLLM mới với batch lớn có thể nhanh hơn vài lần, nhưng không đổi được kết luận vì chênh lệch là ~20 lần.

## 3. Vì sao không CPT thô rồi thôi

| Bằng chứng | Số đo | Nghĩa là |
|---|---|---|
| EntiGraph (ICLR 2025), Llama 3 8B, tài liệu 1,3M token | CPT thô 38,15% < base 39,49%; CPT trên 455M token tổng hợp 56,22% | Tài liệu **càng nhỏ và ít lặp lại** thì CPT thô càng khó nạp sự kiện |
| Ovadia (EMNLP 2024), model 7B | Llama2 tin tức mới: base 0,353 → CPT thô 0,219 → CPT có paraphrase 0,392; RAG ~0,83–0,88 | Sự kiện phải xuất hiện **nhiều cách** mới học được; RAG thắng xa cho sự kiện mới |
| Mix-CPT (2024), Llama3-8B, ~4B token Wikipedia | CPT thô làm QA Wiki 26,10 → 22,20; Code 58,75 → 15,56. **Trộn dữ liệu chỉ thị vào CPT**: QA Wiki 44,09, Code giữ 57,20 | Trộn dữ liệu chỉ thị **có sẵn** vào CPT vừa giúp nạp kiến thức vừa chống quên, **không cần sinh gì** |
| AdaptLLM (ICLR 2024), LLaMA-7B | CPT thô làm điểm prompting y sinh 44,2 → 41,7; biến tài liệu thành bài đọc hiểu bằng **vài mẫu regex** + trộn chỉ thị: tài chính 63,4 (base 58,6, ngang BloombergGPT-50B) | Có cách biến văn bản thô thành dạng hỏi–đáp **chỉ bằng CPU** |
| Instruction Pre-Training (EMNLP 2024), Llama3-8B | Cùng ngân sách token: y sinh 61,3 so với CPT thô 58,4; tài chính 74,7 so với 72,0. Sinh hỏi–đáp ngắn bằng model 7B: **~24 GPU-giờ cho 1 tỷ token nguồn** | Sinh **ngắn** (~5 cặp × 52 token mỗi đoạn) thì rẻ; cái đắt là sinh **dài và nhiều bản** |

Điểm cần nói rõ với sếp: corpus của dự án (300M token, nhiều nguồn chồng nhau: sách, giáo trình, bài báo, web) **lớn và lặp lại tự nhiên hơn nhiều** so với các thí nghiệm "CPT thô thất bại" (1,3M token, tin tức mới). Nên CPT thô ở đây có thể tốt hơn các bài đó, **nhưng chưa ai đo cho tiếng Việt**. Vì vậy cần thí nghiệm so sánh ở mục 7 trước khi chốt.

## 4. Chống catastrophic forgetting: 7 lớp

Cái phải giữ: làm theo chỉ thị / chat và lập luận / toán / code. Tiếng Anh không bắt buộc, nhưng toán và code phần lớn là tiếng Anh, nên replay vẫn phải có toán và code.

| # | Cách làm | Bằng chứng | Chi phí |
|---|---|---|---|
| 1 | **CPT trên base, không CPT lên instruct** | CPT thẳng lên Llama3-8B-Instruct: IFEval 35,30 → 22,55 (1B token), GSM8K 75,06 → 68,01; MT-Bench 7,89 → ~1,0 (FinDAP). Chat Vector: CPT+SFT trên llama2-chat còn 6,46, trên base + chat vector 7,37 | 0 |
| 2 | **LR thấp** (1e-5 đến 2e-5), warmup ngắn, giảm dần về cuối | Ibrahim (2024): làm nóng lại LR là nguyên nhân lớn của quên; LR đỉnh cao = học nhiều hơn nhưng quên nhiều hơn. SEA-LION v2 dùng 1e-5, Sailor2 giai đoạn 2 dùng 1e-5 | 0 |
| 3 | **Replay 20–25%**: dữ liệu chung chất lượng cao + **toán + code** | Ibrahim: 1% replay đã giảm quên rõ, 5% cho lệch yếu, 25% cho lệch mạnh; 50% thì học kém đi. Sailor2 20% (bỏ code, nhưng ta cần code nên phải giữ); EntiGraph 10% | thêm ~25% token CPT |
| 4 | **Trộn dữ liệu chỉ thị có sẵn vào CPT** (SEA-Instruct tiếng Việt + bộ chỉ thị mở có toán và code) | Mix-CPT: Code 15,56 (CPT thô) → 57,20; FinDAP: học CPT và chỉ thị cùng lúc thắng làm lần lượt | ~10% token CPT, không sinh gì |
| 5 | **Cộng chat vector** sau CPT: `M = CPT_base + α × (Instruct − Base)`, rồi SFT ngắn | Instruction residual (2024): sau 1B token, IFEval 37,52 (base CPT + residual) so với 22,55 (CPT lên instruct). Chat Vector: CPT + vector + SFT tốt nhất (7,37) | CPU vài phút |
| 6 | **SFT ngắn bằng LoRA** có toán / code / chỉ thị tiếng Việt, để vá phần lớp 5 còn thiếu | Instruction residual kéo GSM8K về **49,51 trong khi instruct gốc 74,98**, tức merge một mình **không đủ cho toán**. Mix-CPT: chỉ cần ~10K mẫu SFT + 5K cặp ưu tiên sau CPT trộn | ~5–10 GPU-giờ |
| 7 | **Cổng eval ở mọi checkpoint**, có ngưỡng dừng | Đợt tụt có thể nằm giữa chừng rồi hồi lại; nhìn checkpoint cuối dễ bỏ sót | ~2 GPU-giờ mỗi checkpoint |

**Phương án dự phòng** khi cổng eval vẫn đỏ sau 7 lớp trên:

- **Source-Shielded Updates (SSU, ACL 2026):** đóng băng 50% cột trọng số quan trọng nhất cho khả năng cũ (chấm bằng 500 mẫu chỉ thị), rồi mới CPT. Với OLMo 2 Instruct 7B và 200M token (gần cỡ corpus này), mức tụt trung bình của chat / chỉ thị là **5,9%**, so với **34,1%** khi CPT full; HumanEval tụt 10,4% so với 49,7%. Tốn thêm rất ít GPU nhưng cần tự code.
- **CPT bằng LoRA:** AdaLoRA chỉ tụt 9,0% (cùng bài SSU), nhưng FinDAP cho thấy chuyển kiến thức từ CPT sang hiểu nhiệm vụ cần train full. Vì vậy chỉ dùng khi bắt buộc (ví dụ model 27B, mục 8).

**Cổng eval đề xuất** (ngưỡng là đề xuất của tôi, chủ dự án chốt): so với **model instruct gốc**, không tụt quá 2 điểm tuyệt đối ở IFEval, GSM8K, HumanEval/MBPP, MMLU; đồng thời VMLU và Eval A (sự kiện đã dạy) phải tăng, còn Eval B (tỉ lệ bịa) không được tăng.

## 5. Pipeline chi tiết (mức A, mặc định)

**Bước 0. Đo base.** Giữ như bản cũ (`docs/SUGGESTION.md` mục 5, Bước 0), thêm HumanEval/MBPP. Base phải có **cả bản base lẫn bản instruct công bố** cùng họ (cần cho chat vector ở Bước 3).

**Bước 1. Làm hỗn hợp CPT (chỉ CPU).**

| Thành phần | Tỉ lệ (đề xuất) | Nguồn |
|---|---|---|
| Corpus tiếng Việt sạch | ~60% | Output `finalize` hiện có. Nguồn chất lượng cao (stbook, giáo trình, VJOL) lặp 2 lần, giống SEA-LION v2 lặp Wikipedia tiếng Việt 4 lần; web SEA-PILE 1 lần |
| Bài đọc hiểu kiểu AdaptLLM | ~5% | Sinh từ chính corpus bằng mẫu regex (định nghĩa, "X là ...", tóm tắt theo tiêu đề, điền từ...), gắn ngay sau đoạn gốc |
| Replay toán + code + chung | 20–25% | Bộ mở (ví dụ FineWeb-Edu, FineWeb2 tiếng Việt, OpenWebMath / FineMath, một phần The Stack), cần kiểm giấy phép |
| Dữ liệu chỉ thị có sẵn | ~10% | SEA-Instruct-2602 tiếng Việt (đã nằm trong KU SFT) + bộ chỉ thị mở có toán / code, bỏ chat template như Mix-CPT |

Việc mới phải code: một stage `cpt_mix` sau `finalize` (đóng gói và trộn theo tỉ lệ, đếm token bằng tokenizer của base), và `reading_comp.py` (mẫu regex cho tiếng Việt). Không cần GPU.

**Bước 2. CPT full trên base.** ~1 epoch hỗn hợp (~600M token), LR 1e-5 đến 2e-5, lịch WSD hoặc cosine, độ dài 4.096. Model 8B full trên 4× A100 40GB rất sát bộ nhớ: cần FSDP / ZeRO-3 và optimizer 8-bit hoặc offload sang CPU. **Phải chạy thử 200 bước để đo tốc độ thật trước khi chạy hết.** Lưu checkpoint mỗi ~100M token.

**Bước 3. Trả lại khả năng chat.** Cộng chat vector, quét α = 0,5 / 0,75 / 1,0, có và không có lớp embedding / lm_head (Chat Vector thấy bỏ hai lớp này tốt hơn trên Mistral). Sau đó SFT ngắn bằng LoRA (rank 64–128, mọi lớp tuyến tính), ~10–50K mẫu gồm chỉ thị tiếng Việt, toán, code. Chọn bản tốt nhất qua cổng eval.

**Bước 4. RAG (Claude chốt: giữ).** Bằng chứng: với sự kiện ít phổ biến, RAG thắng fine-tune rất xa (Mistral-7B PopQA: FT 30,70, RAG 80,25; Soudani 2024). Chi phí gần như bằng 0 vì corpus đã cắt chunk, đã có `doc_id` và đã nhúng bằng Qwen3-Embedding. **Bỏ khỏi mặc định việc sinh tập RAFT** (tốn GPU). SFT dùng LoRA vì full fine-tune làm model kém dùng tài liệu truy xuất (Soudani: 32,50 so với 82,09). Chỉ làm RAFT khi đo thấy model dùng tài liệu kém.

**Bước 5. Cổng eval.** Như mục 4. Bộ Eval A / B vẫn cần (bản cũ, bước 7), nhưng chỉ ~1–2K câu, do model 8B nội bộ sinh rồi đọc tay một phần, tốn ~vài GPU-giờ.

## 6. Ba mức ngân sách

| Mức | Thêm gì so với mức trước | Tổng GPU-giờ (ước tính) | Thời gian trên 4 GPU |
|---|---|---|---|
| **A** (mặc định) | Mục 5: CPT trộn + chat vector + SFT LoRA + RAG, không sinh dữ liệu bằng model | ~120–150 | 1,5–2 ngày |
| **B** | Viết lại phần giá trị cao (~100M token sách / giáo trình / VJOL) bằng **model nhỏ 1,7–4B** thành 2 dạng có cấu trúc (hỏi–đáp FAQ, bài giảng / tóm tắt), trộn 1:1 với bản gốc; cộng hỏi–đáp ngắn kiểu Instruction Pre-Training | ~220–280 | ~3 ngày |
| **C** | EntiGraph / Active Reading **chỉ cho mảng hẹp** mà Eval A cho thấy còn thiếu (ví dụ ~10M token văn hoá–lịch sử), do chính model học tự sinh | ~400 | ~4–5 ngày |

Vì sao mức B dùng model nhỏ được: nhiều bài đo thấy model viết lại lớn hơn ~1–3B **gần như không thêm lợi**. FinePhrase (2026): model 1B hơn model 27B; từ 1B lên 12B chỉ thêm 0,21 điểm; model 1,7B chạy ~33M token/giờ trên H100. BeyondWeb: 1B / 3B / 8B được 47,3 / 48,8 / 49,2%, lợi ích phẳng từ 3B trở lên. WRAP: Qwen-1.8B ngang Mistral-7B và nhanh gấp 3. RePro: model 1B (có huấn luyện thêm) tốn 727 GPU-giờ so với 63.360 của model 70B. **Chỗ chưa chắc:** các số này là pretraining tiếng Anh từ đầu, chưa ai đo model 1,7–4B viết lại tiếng Việt có giữ đúng sự kiện không. Mức B phải có bước đọc tay ~200 mẫu và kiểm "sự kiện trong bản viết lại có trong bản gốc" trước khi chạy hết.

## 7. Thí nghiệm so sánh công bằng (trước khi chốt mức)

Chủ dự án chọn "so sánh công bằng" (T-03). Đề xuất: lấy một lát nhỏ (~30M token của một mảng, ví dụ lịch sử–văn hoá), dựng Eval A / B từ chính lát đó, rồi chạy 4 nhánh **với cùng ngân sách GPU-giờ** (tính cả giờ sinh dữ liệu):

| Nhánh | Nội dung |
|---|---|
| N1 | CPT thô + replay → chat vector → SFT ngắn |
| N2 | N1 + trộn chỉ thị + bài đọc hiểu regex (mức A) |
| N3 | N2 + viết lại bằng model nhỏ (mức B) |
| N4 | Kiểu bản cũ: sự kiện + biến thể + hỏi–đáp → SFT LoRA |

Đo: Eval A (4 dạng câu hỏi), Eval B (tỉ lệ bịa), VMLU, IFEval, GSM8K, HumanEval. Cỡ ~25–40 GPU-giờ mỗi nhánh, tổng ~1,5 ngày. Kết quả trả lời đúng câu hỏi của sếp ("thà CPT còn nhanh hơn") bằng số của chính dự án thay vì bằng bài báo tiếng Anh.

## 8. Rủi ro và chỗ chưa chắc

- **Cỡ model đích.** R-21 ghi "có vẻ ~27B". CPT full 27B **không vừa** 4× A100 40GB (riêng trạng thái optimizer đã ~400GB), và tốn gấp ~3,4 lần GPU-giờ. Khi đó Bước 2 phải là LoRA / QLoRA-CPT hoặc SSU trên một phần lớp, và học sẽ kém hơn. Cần chốt cỡ trước khi ước tính lại.
- **Chưa có số tiếng Việt** cho bất kỳ phương pháp nào ở đây; số tụt điểm là tiếng Anh, Trung, Tây Ban Nha.
- **Chat vector cần cặp base / instruct cùng họ**, và hệ số α không có công thức. Merge một mình làm tụt toán (GSM8K 49,51 so với 74,98), nên SFT ngắn là bắt buộc, không tuỳ chọn.
- **Giấy phép dữ liệu replay và chỉ thị mở** phải kiểm trước khi dùng; `rights_status` của corpus vẫn đi theo từng đoạn.
- **Base quá "no" (Qwen2.5 18T, Llama3.1 15T token)** để lại ít chỗ cho CPT (Sailor2 cảnh báo). Bước 0 nên so cả mức giảm perplexity tiếng Việt sau 200 bước CPT thử.

## 9. So với bản cũ: giữ gì, bỏ gì

| | Bản cũ | Bản mới |
|---|---|---|
| Trục nạp kiến thức | SFT LoRA trên dữ liệu tổng hợp | CPT trên base, trộn chỉ thị và replay |
| Model sinh dữ liệu | Teacher lớn ở B1, B3a, B3b, B3c | Không có ở mức A; model nhỏ 1,7–4B ở mức B; mức C chỉ cho mảng hẹp |
| Chống quên | LoRA, replay, đo checkpoint | 7 lớp ở mục 4, có ngưỡng dừng |
| RAG | Có, kèm RAFT | Có, RAFT chỉ khi đo thấy cần |
| Giữ nguyên | Bước 0, Eval A / B, chia theo họ trùng, quét nhiễm benchmark, `rights_status` | |

## 10. Cần chủ dự án / sếp quyết

1. Cỡ model đích: ~8B hay ~27B (đổi hẳn Bước 2 và số GPU-giờ).
2. Ngưỡng cổng eval: "tụt ≤ 2 điểm" có hợp không.
3. Có chạy thí nghiệm mục 7 (~1,5 ngày GPU) trước khi chạy mức A cho cả corpus không.
4. Bộ replay / chỉ thị mở nào được phép dùng (giấy phép).

## 11. Nguồn

Các nhận định trên được lấy từ đợt nghiên cứu ngày 2026-10-30 (21 nguồn gốc). Những kết luận suy rộng quá mức của nguồn đã bị loại ở bước kiểm chéo: "khuếch đại 4000× là bắt buộc nên quá đắt" (SPA), "pipeline nhiều bước là không cần" (SPA), "viết lại rẻ thì lợi ích có hạn" (Active Reading).

- Ibrahim et al. 2024, Simple and Scalable Strategies to Continually Pre-train LLMs, arXiv 2403.08763
- Instruction residual: Balancing Continuous Pre-Training and Instruction Fine-Tuning, arXiv 2410.10739
- Chat Vector, ACL 2024, arXiv 2310.04799
- Mix-CPT, arXiv 2407.10804
- FinDAP: Demystifying Domain-adaptive Post-training for Financial LLMs, EMNLP 2025, arXiv 2501.04961
- Source-Shielded Updates, ACL 2026, arXiv 2512.04844
- AdaptLLM, ICLR 2024, arXiv 2309.09530
- Instruction Pre-Training, EMNLP 2024, arXiv 2406.14491
- EntiGraph / Synthetic Continued Pretraining, ICLR 2025
- Active Reading, arXiv 2508.09494
- SPA, ICML 2026, arXiv 2603.22213
- Ovadia et al., Fine-Tuning or Retrieval?, EMNLP 2024
- Soudani et al., Fine Tuning vs. RAG for Less Popular Knowledge, arXiv 2403.01432
- WRAP, ACL 2024, arXiv 2401.16380
- RePro, arXiv 2510.10681
- BeyondWeb, arXiv 2508.10975
- FinePhrase: How Can We Synthesize High-Quality Pretraining Data?, arXiv 2604.13977
- FinPythia / Efficient Continual Pre-training for Domain LLMs, arXiv 2311.08545
- Sailor, arXiv 2404.03608; Sailor2, arXiv 2502.12982; SEA-LION v2 model card (GitHub aisingapore/sealion); OpenSeal, arXiv 2602.02266
