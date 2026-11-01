# Báo cáo tổng kết: quy trình chung trên Derm7pt

Cập nhật ngày 02/11/2026. Mọi số liệu được đọc lại từ kết quả đã lưu; báo cáo này không chạy thêm lần chấm test nào.

## 1. Những gì đã làm được

1. **Huấn luyện 4 mô hình theo cùng một quy trình:** Black-box, Soft CBM, Hard CBM và ECBM.
   - Cùng EfficientNet-B0, 50 epoch, AdamW, 3 seed (42/123/2026).
   - Checkpoint chọn theo validation Macro F1@0.5; ngưỡng chọn trên validation rồi giữ cố định cho test.
2. **Chọn λ_global cho ECBM trên validation** theo luật ghi trước: chọn λ=1, thay cho λ=0.01 của paper. Xem [kế hoạch](ecbm/lambda_global_study/study_plan.json) và [báo cáo](ecbm/lambda_global_study/report.md).
3. **Khóa kế hoạch rồi chấm test đúng một lần.** Checkpoint được khóa bằng SHA-256, kế hoạch test ghi trước trong [`test_plan.json`](test_plan.json).
4. **Đánh giá can thiệp concept:** sửa từ 0 đến 7 nhóm concept bằng nhãn thật, duyệt đủ 128 tập con, cho cả CBM và ECBM. So sánh bằng paired bootstrap 2000 lần.
5. **Phân tích hạn chế của ECBM:** 6 hạn chế, 7 hình. Xem [báo cáo hạn chế](ecbm/limitations/report.md).

## 2. Số liệu test

Test gồm 395 ca, trong đó 101 melanoma. Các giá trị là mean ± SD qua 3 seed.

### 2.1. Bảng chính

| Mô hình | Acc % | BAcc % | Macro F1 | ROC-AUC | Concept acc % |
|---|---:|---:|---:|---:|---:|
| Black-box | 80.84 ± 0.15 | 73.92 ± 2.25 | **0.7428** ± 0.0119 | 0.8433 ± 0.0071 | — |
| Soft CBM | **81.10** ± 0.58 | 71.92 ± 1.37 | 0.7333 ± 0.0119 | **0.8674** ± 0.0058 | 65.77 ± 0.67 |
| Hard CBM | 78.40 ± 1.39 | **74.11** ± 3.07 | 0.7285 ± 0.0216 | 0.8394 ± 0.0201 | **66.71** ± 0.27 |
| **ECBM λ=1 (chính)** | 80.17 ± 1.02 | 72.16 ± 3.70 | 0.7282 ± 0.0267 | 0.8535 ± 0.0170 | 58.65 ± 7.76 |
| ECBM λ paper (0.01) | 79.07 ± 1.92 | 71.86 ± 3.40 | 0.7198 ± 0.0212 | 0.8479 ± 0.0129 | 63.92 ± 4.23 |
| ECBM chỉ concept (phụ) | 76.46 ± 3.78 | 72.92 ± 1.21 | 0.7111 ± 0.0222 | 0.8329 ± 0.0225 | 56.02 ± 8.55 |

Các metric khác (sensitivity, specificity, concept F1...) nằm trong file kết quả của từng mô hình (mục 6), dùng cho phụ lục nếu cần.

### 2.2. Paired bootstrap Macro F1

2000 lần lặp, seed 20261005.

| So sánh | Chênh lệch | Khoảng 95% | Chứa 0 |
|---|---:|---:|:---:|
| ECBM λ=1 − Black-box | −0.0146 | [−0.0432, +0.0139] | có |
| ECBM λ=1 − Soft CBM | −0.0051 | [−0.0359, +0.0249] | có |
| ECBM λ=1 − Hard CBM | −0.0003 | [−0.0308, +0.0293] | có |
| ECBM chỉ concept − Soft CBM | −0.0222 | [−0.0579, +0.0160] | có |
| ECBM chỉ concept − Hard CBM | −0.0174 | [−0.0483, +0.0145] | có |

Mọi khoảng đều chứa 0, nên không có khác biệt có ý nghĩa thống kê.

### 2.3. Can thiệp concept

Macro F1 khi sửa concept bằng nhãn thật, ngưỡng giữ cố định.

| Mô hình | Không sửa | Sửa đủ 7 nhóm | Tăng | Curve AUC | Số ca đổi nhãn |
|---|---:|---:|---:|---:|---:|
| Soft CBM | 0.7333 | 0.7472 | +0.014 | 0.7534 | 68.7 |
| **Hard CBM** | 0.7285 | **0.7925** | **+0.064** | 0.7694 | 82.7 |
| ECBM λ=1 | 0.7282 | 0.7764 | +0.048 | 0.7506 | 24.3 |
| ECBM λ paper | 0.7198 | 0.7233 | +0.004 | 0.7214 | 1.0 |
| ECBM chỉ concept | 0.7111 | 0.7751 | +0.064 | **0.7713** | 91.3 |

"Số ca đổi nhãn" là số ca trung bình (trên 395) đổi nhãn khi sửa đủ 7 nhóm.

Tỷ lệ ca mà ECBM cho cùng nhãn với riêng nhánh ảnh E_class:
- λ=1: 96.96%
- λ paper: 99.92%
- chỉ concept: 80.59%

## 3. Ưu điểm

- **Phương pháp chặt chẽ.** Các mô hình được so sánh công bằng, mọi lựa chọn làm trên validation, kế hoạch ghi trước và test chỉ chấm một lần. Đây là điểm mạnh nhất hiện tại.
- **ECBM chẩn đoán ngang black-box.** Macro F1 chỉ chênh −0.015 và không có ý nghĩa thống kê. ROC-AUC 0.854 nhỉnh hơn black-box 0.843, nhưng chênh lệch này chưa được kiểm định.
- **Phát hiện mới về cấu hình của paper.** Với λ của paper, nhánh concept gần như không có tác dụng: sửa đủ 7 concept chỉ đổi nhãn trung bình 1 trên 395 ca. λ=1 (chọn trên validation) làm can thiệp có tác dụng: +0.048 so với +0.004.
- **Suy luận chính xác.** Mô hình liệt kê đủ 8,640 tổ hợp concept, không xấp xỉ.
- **Giải thích được bằng số.** Phân rã log-odds cho biết ở từng ca, bao nhiêu phần quyết định đến từ concept.

## 4. Hạn chế

### 4.1. Về kết quả

- **ECBM không thắng mô hình nào.** Mỗi metric có một mô hình khác đứng đầu, và không khác biệt nào có ý nghĩa thống kê.
- **Về can thiệp, Hard CBM tốt hơn ECBM λ=1:** 0.7925 so với 0.7764 khi sửa đủ, curve AUC 0.7694 so với 0.7506. Nghĩa là ECBM λ=1 chỉ can thiệp tốt hơn ECBM λ paper, chứ không tốt hơn CBM.
- **Concept của ECBM thấp nhất:** 58.65%, so với 65.77–66.71% của CBM. Tăng λ còn làm concept kém thêm (63.92 → 58.65).
- **Chẩn đoán vẫn chủ yếu do nhánh ảnh:** 97% ca trùng nhãn với E_class.
- **Dao động giữa các seed lớn.** Macro F1 của ECBM λ=1 đi từ 0.700 (seed 123) đến 0.754 (seed 42); concept accuracy có SD 7.8 điểm.
- **Sensitivity thấp ở mọi mô hình (53–66%).** Tức là bỏ sót khoảng một phần ba đến gần một nửa ca melanoma. Metric này không có trong bảng chính nhưng nên bàn trong phần thảo luận.

### 4.2. Nguyên nhân (từ phân tích hạn chế)

- Lúc train, E_global chỉ thấy concept sai 7.1%. Lúc chạy thật, concept dự đoán sai 36.7%.
  - Hệ quả: AUC của E_global giảm từ 0.928 (concept đúng) xuống 0.818 (concept dự đoán), thấp hơn E_class (0.831).
  - Vì nhánh concept yếu, mô hình dựa vào E_class.
- Can thiệp làm lệch điểm hoạt động: specificity tăng từ 88.5% lên 94.2%, còn sensitivity gần như đứng yên (55.8% → 56.8%).
- Seed nào có checkpoint tốt nhất ở epoch sớm thì concept yếu hơn: epoch 10 → 59.8%, 17 → 63.0%, 30 → 67.2% (validation).

Chi tiết và hình minh họa nằm trong [báo cáo hạn chế](ecbm/limitations/report.md).

### 4.3. Về thực nghiệm

- Chỉ có 3 seed và 395 ca test, nên khoảng tin cậy rộng khoảng ±0.03 Macro F1 và khó phát hiện khác biệt nhỏ.
- Bootstrap mới làm cho Macro F1, và chỉ cho các cặp có ECBM.
- Test đã bị xem. Mọi cải tiến sau này phải ghi rõ là "đánh giá trên test đã thấy".
- Chỉ dùng một dataset (Derm7pt) và một backbone.
- Hướng guard (repo `ECBM_Derm7pt`) không cải thiện, được ghi lại như một kết quả âm.

## 5. Ý nghĩa cho khóa luận

Với kết quả hiện tại, đóng góp của khóa luận là ba ý: đánh giá công bằng, chỉ ra ECBM theo paper không dùng tới concept, và phân tích vì sao. Chưa thể nói "ECBM tốt hơn".

Mục tiêu hợp lý cho bước cải tiến có hai phần, đều phải giữ được hiệu năng chẩn đoán:
- can thiệp ít nhất bằng Hard CBM;
- concept accuracy không thấp hơn CBM.

## 6. Nguồn số liệu

| Nội dung | File |
|---|---|
| Kết quả test Black-box, Soft CBM, Hard CBM | `efficientnet_b0/linear_e50_headlr0.001/`, `soft_cbm/efficientnet_b0_mlp128_e50_headlr0.001/`, `hard_cbm/efficientnet_b0_mlp128_e50_headlr0.001/` (`seed*.json`) |
| Can thiệp Soft/Hard CBM | thư mục `intervention/` của hai cohort CBM |
| Kết quả test ECBM theo 3 chế độ | `ecbm/test_modes/seed*.json`, `ecbm/test_modes/summary.json` |
| Bootstrap | `test_summary.json` |
| Phân tích hạn chế | `ecbm/limitations/numbers.json` |

Các đường dẫn tính từ `results/derm7pt/`. Accuracy, BAcc, ROC-AUC được tính lại từ `test_predictions` đã lưu.
