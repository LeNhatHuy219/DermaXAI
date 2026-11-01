# Hạn chế của ECBM trên Derm7pt

Phạm vi: ECBM categorical, cohort quy trình chung 50 epochs (`results/derm7pt/ecbm/efficientnet_b0_unified_exact_e50`), 3 seeds 42/123/2026.

- Số liệu **validation** (203 ca) được tính mới từ các checkpoint đã đóng băng.
- Số liệu **test** (395 ca) chỉ đọc lại từ kết quả đã lưu (`results/derm7pt/ecbm/test_modes/`, `results/derm7pt/{soft,hard}_cbm/.../intervention/`), không đọc lại ảnh test.
- Tái tạo: `python3 experiments/analyze_ecbm_limitations.py`. Số liệu đầy đủ trong [numbers.json](numbers.json).

Ký hiệu: "λ=1" là ECBM với trọng số suy luận 1/1/1 (model chính); "λ paper" là 1/1/0.01; "chỉ concept" là 0/1/1.

## Tóm tắt

| # | Hạn chế | Số liệu chính | Hình |
|---|---|---|---|
| 1 | Chẩn đoán tự động gần như chỉ dựa vào nhánh ảnh E_class | Với λ=1, độ lớn đóng góp vào log-odds: E_class 4.92, phần qua concept 1.35. Phần concept lớn hơn E_class ở 15.4% ca. Test: trùng nhãn với E_class 97.0% | 1 |
| 2 | Với λ của paper, can thiệp concept vô tác dụng | Test: sửa đủ 7 concept chỉ tăng Macro F1 +0.004 (λ paper), so với +0.048 (λ=1) | 2, 3 |
| 3 | Can thiệp làm lệch điểm hoạt động | Test, λ=1: specificity 88.5% → 94.2%, sensitivity 55.8% → 56.8%; ROC-AUC 0.853 → 0.918 | 4 |
| 4 | Concept dự đoán kém hơn CBM, và tăng λ_global làm concept kém thêm | Test concept accuracy: λ=1 58.7%, λ paper 63.9%, Hard CBM 66.7%, Soft CBM 65.8% | 5 |
| 5 | Không ổn định giữa các seed | Test concept accuracy với λ=1: 62.8 / 49.7 / 63.4%. Hard CBM: 66.9 / 66.8 / 66.4% | 6 |
| 6 | E_global được train trên concept gần đúng nhưng lúc chạy nhận concept sai nhiều | Validation ROC-AUC: E_global + concept đúng 0.928, E_global + concept dự đoán 0.818, E_class 0.831. Nhiễu lúc train 7.1%, lỗi concept dự đoán 36.7% | 7 |

---

## 1. Chẩn đoán tự động gần như chỉ dựa vào E_class

**Hiện tượng.** Với suy luận chính xác, log-odds melanoma tách đúng thành hai phần:

log p(y=1|x)/p(y=0|x) = [E_class(x,0) − E_class(x,1)] + [phần đi qua concept: E_concept + E_global]

Hình 1 vẽ hai phần này cho 609 điểm (203 ca × 3 seeds). Điểm nằm giữa hai đường chéo là ca mà phần concept lớn hơn E_class.

![Hình 1](fig1_logodds_decomposition.png)

**Bằng chứng (validation, trung bình 3 seeds).**

| | λ paper (0.01) | λ = 1 |
|---|---:|---:|
| Độ lớn trung bình phần E_class | 4.92 | 4.92 |
| Độ lớn trung bình phần qua concept | 0.016 | 1.35 |
| Tỷ lệ ca phần concept lớn hơn E_class | 0.2% | 15.4% |
| Tỷ lệ ca hai phần ngược chiều nhau | 19.1% | 41.1% |
| Trong các ca ngược chiều, concept thắng | 0% | 15.3% |

Trên test, chẩn đoán trùng nhãn với riêng E_class: λ paper 99.9%, λ=1 97.0%, chỉ concept 80.6%.

**Nguyên nhân.**
- E_class được train với trọng số 3, các năng lượng khác trọng số 1. Log-odds trung bình 4.9 tương ứng xác suất khoảng 0.99, tức nhánh ảnh rất "tự tin".
- E_global không nhìn ảnh, nên backbone chỉ học từ E_class và E_concept.
- Nhánh concept yếu hơn E_class khi nhận concept dự đoán (mục 6). Mô hình dựa vào E_class là hợp lý về mặt độ chính xác.

**Hệ quả.** Ở chế độ tự động, concept chỉ là lời giải thích đi kèm, không phải lý do của chẩn đoán. Không được viết "ECBM chẩn đoán dựa trên concept" cho chế độ này.

**Hướng cải tiến.** Giảm trọng số E_class khi train (3 → 1), đồng thời sửa mục 6 để nhánh concept đủ mạnh.

## 2. Với λ của paper, can thiệp concept vô tác dụng

**Hiện tượng.** Với λ_global = 0.01 như paper, sửa concept gần như không làm đổi nhãn nào.

![Hình 2](fig2_lambda_sweep.png)

![Hình 3](fig3_intervention_curves_test.png)

**Bằng chứng.**
- Validation (Hình 2): số ca đổi nhãn khi sửa đủ 7 concept tăng từ 0.3 (λ=0.01) lên 12.3 (λ=1) và 18.3 (λ=3).
- Validation: Macro F1 sau can thiệp cao nhất ở λ 0.3–1. Đến λ=3 thì giảm xuống 0.774, thấp hơn cả khi không sửa (0.791).
- Test (Hình 3), Macro F1 khi sửa đủ 7 concept:

| Mô hình | Không sửa → sửa đủ 7 | Mức tăng |
|---|---|---:|
| ECBM λ paper | 0.720 → 0.723 | +0.004 |
| ECBM λ=1 | 0.728 → 0.776 | +0.048 |
| ECBM chỉ concept | 0.711 → 0.775 | +0.064 |
| Hard CBM | 0.729 → 0.793 | +0.064 |
| Soft CBM | 0.733 → 0.747 | +0.014 |

**Nguyên nhân.** Ở λ=0.01, phần đi qua concept chỉ khoảng 0.016 log-odds (mục 1), quá nhỏ để đổi quyết định. Ở λ=3 thì ngược lại: E_global lấn át và xác suất dồn sát 0 hoặc 1 (ngưỡng chọn trên validation lên tới ~0.99).

**Hệ quả.** λ_global là siêu tham số nhạy và không mang nguyên từ CUB sang Derm7pt được. Dùng nguyên giá trị của paper thì mất luôn tính năng can thiệp, vốn là điểm mạnh chính của ECBM.

**Hướng cải tiến.** Chọn λ theo dữ liệu, như study đã làm, hoặc cho mô hình học λ.

## 3. Can thiệp làm lệch điểm hoạt động

**Hiện tượng.** Sau khi sửa concept, mô hình bớt báo động giả, nhưng không bắt được thêm melanoma.

![Hình 4](fig4_operating_point_test.png)

**Bằng chứng (test, ngưỡng đóng băng).**

| Mô hình | Sensitivity trước → sau | Specificity trước → sau | ROC-AUC trước → sau |
|---|---|---|---|
| ECBM λ=1 | 55.8 → 56.8% | 88.5 → 94.2% | 0.853 → 0.918 |
| ECBM chỉ concept | 65.7 → 59.1% | 80.2 → 92.6% | 0.833 → 0.924 |
| Hard CBM | 65.3 → 62.7% | 82.9 → 92.9% | 0.839 → 0.921 |

Với ECBM λ=1 và chỉ concept, khoảng 50% ca melanoma bị **giảm** xác suất melanoma sau khi sửa đủ 7 concept.

**Nguyên nhân.**
- Ngưỡng được chọn trên phân bố xác suất khi concept là dự đoán. Concept đúng làm phân bố đó dịch chuyển, nên ngưỡng cũ không còn hợp. AUC tăng mạnh chứng tỏ khả năng xếp hạng tốt lên; chỉ có ngưỡng là lệch.
- Derm7pt có 30.3% số ca thuộc các bộ concept giống nhau nhưng khác chẩn đoán (Nápoles et al., 2026). Với một phần ca melanoma, bộ concept đúng trông giống ca lành tính.

**Hệ quả.** Hạn chế này không riêng ECBM, Hard CBM cũng bị. Nhưng nó làm F1 sau can thiệp đánh giá thấp lợi ích thật của việc sửa concept.

**Hướng cải tiến.** Chọn ngưỡng hoặc hiệu chỉnh xác suất riêng cho chế độ "đã sửa concept". Luôn báo cáo AUC kèm F1.

## 4. Concept kém hơn CBM, và tăng λ_global làm concept kém thêm

**Hiện tượng.** Concept mà ECBM hiển thị cho bác sĩ đúng ít hơn concept của CBM, và càng kém hơn khi concept được phép ảnh hưởng chẩn đoán.

![Hình 5](fig5_concept_accuracy_by_group_test.png)

**Bằng chứng.**
- Test, concept accuracy: λ=1 58.7%, λ paper 63.9%, chỉ concept 56.0%, Hard CBM 66.7%, Soft CBM 65.8%.
- Validation, cùng checkpoint, chỉ đổi cách suy luận: E_concept riêng 63.3%, λ paper 63.4%, λ=1 58.6%, chỉ concept 56.2%.
- Giảm mạnh nhất ở Vascular (test: 59.3% với λ paper, 48.9% với λ=1, 66.0% với CBM) và Pigmentation (54.6%, 49.6%, 59.6% với Hard CBM).

**Nguyên nhân.**
- E_concept dùng chung một phép chiếu 32 chiều cho cả 7 nhóm. CBM dùng head riêng lấy thẳng từ 1280 chiều. Mức E_concept riêng (63.3%) đã thấp hơn CBM.
- Khi λ_global lớn, suy luận chính xác kéo xác suất concept về các bộ concept "hợp" với chẩn đoán. Chẩn đoán vì thế bóp méo ngược lại lời giải thích.

**Hệ quả.** Có một đánh đổi: để concept ảnh hưởng chẩn đoán (mục 2), mô hình chấp nhận để chẩn đoán làm sai lệch concept được hiển thị.

**Hướng cải tiến.** Tăng sức chứa của E_concept (chiều ẩn lớn hơn, phép chiếu riêng cho từng nhóm). Hoặc hiển thị concept từ E_concept trước bước suy luận chung.

## 5. Không ổn định giữa các seed

![Hình 6](fig6_seed_variability_test.png)

**Bằng chứng.**
- Test, ECBM λ=1: concept accuracy 62.8 / 49.7 / 63.4%, ngưỡng 0.499 / 0.865 / 0.790.
- Hard CBM: concept accuracy 66.9 / 66.8 / 66.4%.

**Nguyên nhân (gợi ý, mới có 3 seeds).** Checkpoint được chọn theo Macro F1 chẩn đoán, mà chẩn đoán do E_class chi phối (mục 1). Vì vậy có thể chọn trúng epoch mà nhánh concept chưa học xong.

| Seed | Best epoch | E_concept accuracy (validation) | λ=1 (validation) |
|---|---:|---:|---:|
| 123 | 10 | 59.8% | 50.9% |
| 42 | 17 | 63.0% | 60.8% |
| 2026 | 30 | 67.2% | 64.0% |

λ=1 khuếch đại điểm yếu này: ở seed 123, concept accuracy tụt 9 điểm.

**Hướng cải tiến.** Chọn checkpoint theo tiêu chí kết hợp chẩn đoán và concept, hoặc đặt số epoch tối thiểu.

## 6. E_global: train trên concept gần đúng, chạy trên concept sai nhiều

**Hiện tượng.** Nhánh concept → chẩn đoán rất mạnh khi nhận concept đúng, nhưng yếu hơn cả E_class khi nhận concept dự đoán.

![Hình 7](fig7_e_global_gt_vs_predicted.png)

**Bằng chứng (validation, trung bình 3 seeds).**

| Đầu vào | ROC-AUC | Macro F1 (ngưỡng chọn trên validation) |
|---|---:|---:|
| E_global + concept đúng (GT) | 0.928 | 0.849 |
| E_global + concept dự đoán (argmax E_concept) | 0.818 | 0.762 |
| E_class (nhánh ảnh) | 0.831 | 0.790 |

**Nguyên nhân.** Khi train, E_global nhận concept đúng có xáo trộn: 20% ca × 2/7 nhóm mỗi batch, tức **7.1%** ô concept bị sai. Lúc chạy, concept dự đoán sai **36.7%**, gấp khoảng 5 lần. E_global chưa từng học cách xử lý mức nhiễu đó.

**Hệ quả.** Đây là gốc của mục 1: nhánh concept yếu hơn nhánh ảnh khi chạy thật, nên mô hình dựa vào E_class.

**Hướng cải tiến.** Train E_global trên concept giống lúc chạy: concept dự đoán, lấy mẫu từ phân bố E_concept, hoặc xáo trộn với tỷ lệ khoảng 35%.

---

## Hướng đã thử nhưng không cải thiện: `ecbm_guard`

`ecbm_guard` (repo `ECBM_Derm7pt`) thêm ngân sách KL cho việc sửa concept lúc suy luận. Kết quả không cải thiện; pilot ghi trong README của repo đó cho thấy accuracy chẩn đoán giữ nguyên 80,79%.

Nhìn từ bảng trên, guard can thiệp vào bước tối ưu concept lúc suy luận, chưa nhắm vào mục 1 và mục 6. Đây có thể là lý do không cải thiện, nhưng chưa được kiểm chứng.

## Thứ tự cải tiến gợi ý

1. **Mục 6 cùng mục 1:** train E_global trên concept giống lúc chạy, và giảm trọng số E_class. Đây là gốc của việc concept không quyết định chẩn đoán.
2. **Mục 4 và 5:** tăng sức chứa E_concept, chọn checkpoint có tính tới concept.
3. **Mục 3:** ngưỡng riêng cho chế độ can thiệp. Việc này rẻ, không cần train lại.

## Giới hạn của phân tích

- Chỉ có 3 seeds. Validation có 203 ca và đã được dùng để chọn checkpoint, ngưỡng và λ.
- Tập test đã được xem từ trước. Mọi cải tiến phải chọn hết trên validation, ghi kế hoạch trước, và báo cáo là đánh giá trên tập test đã từng xem.
- Sensitivity và specificity chỉ dùng ở mục 3 để giải thích hạn chế. Bảng kết quả chính vẫn dùng 5 metric đã chốt.
