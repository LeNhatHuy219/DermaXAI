# TTA: trung bình dự đoán qua các bản lật của ảnh (validation, 3 seed)

**Hướng:** data-centric ở bước dự đoán, thuộc ý 2 trong định hướng của thầy (chất lượng thông tin ảnh).

**Câu hỏi:** mỗi ảnh được tạo thêm các bản lật. Mô hình dự đoán trên từng bản rồi lấy trung bình. Cách này có tốt hơn dự đoán trên một ảnh không?

**Cách làm.** Tiêu chí được ghi trong [plan.json](plan.json) trước khi chạy.

- **Không train lại**, chỉ suy luận trên validation bằng checkpoint đã có:
  - Black-box 224 chính thức;
  - Hard CBM, Soft CBM và ECBM bản D4a.
- **Hai bộ ảnh:**
  - **flip4 (chính):** ảnh gốc, lật ngang, lật dọc, lật cả hai chiều. Đây đúng là các phép lật mô hình đã gặp khi train.
  - **dihedral8 (tham khảo):** thêm 4 bản chuyển vị, tương đương xoay 90°.
- **Cách lấy trung bình**, để concept hiển thị khớp với chẩn đoán:
  - Black-box: trung bình xác suất melanoma.
  - Hard CBM và Soft CBM: trung bình xác suất concept trước, rồi mới đưa vào head chẩn đoán.
  - ECBM: trung bình năng lượng ảnh–nhãn và ảnh–concept, rồi suy luận một lần với trọng số 1/1/0.01.
- **Ngưỡng:** chọn lại trên validation theo Macro F1 cho từng nhánh.
- **Kiểm tra trước:** nhánh dùng một ảnh tái lập đúng xác suất validation đã lưu, sai khác lớn nhất 1.7e-6.
- **Không đọc test.**

## Kết quả (validation, trung bình ± độ lệch chuẩn qua seed 42/123/2026)

| Mô hình | TTA | Accuracy | BAcc | Macro F1 | ROC-AUC | Concept acc. (D4a) | Concept Macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Black-box | không | 80.3 ± 0.5% | 74.4 ± 1.5% | 0.754 ± 0.007 | 0.797 ± 0.011 | — | — |
| | **flip4** | 81.1 ± 1.2% | 76.1 ± 2.1% | 0.768 ± 0.018 | **0.824 ± 0.017** | — | — |
| | dihedral8 | 79.3 ± 2.1% | 76.5 ± 1.5% | 0.758 ± 0.013 | 0.817 ± 0.013 | — | — |
| Hard CBM (D4a) | không | 84.9 ± 1.7% | 80.5 ± 1.8% | 0.814 ± 0.019 | 0.855 ± 0.026 | 68.4 ± 0.3% | 0.613 ± 0.008 |
| | **flip4** | 85.7 ± 1.8% | 81.8 ± 2.6% | 0.826 ± 0.023 | **0.868 ± 0.012** | **69.6 ± 0.7%** | 0.625 ± 0.012 |
| | dihedral8 | 85.1 ± 2.0% | 79.8 ± 2.8% | 0.812 ± 0.027 | 0.862 ± 0.008 | 69.4 ± 0.8% | 0.619 ± 0.013 |
| Soft CBM (D4a) | không | 84.4 ± 1.6% | 77.2 ± 1.9% | 0.795 ± 0.020 | 0.831 ± 0.022 | 67.7 ± 1.8% | 0.611 ± 0.021 |
| | **flip4** | 83.4 ± 1.5% | 78.2 ± 0.1% | 0.794 ± 0.011 | **0.841 ± 0.014** | **68.8 ± 0.2%** | 0.620 ± 0.006 |
| | dihedral8 | 84.2 ± 0.5% | 77.8 ± 1.0% | 0.797 ± 0.006 | 0.837 ± 0.020 | 69.2 ± 0.8% | 0.620 ± 0.009 |
| ECBM (D4a) | không | 84.7 ± 0.5% | 79.3 ± 1.5% | 0.808 ± 0.010 | 0.858 ± 0.007 | 69.7 ± 1.3% | 0.620 ± 0.009 |
| | **flip4** | 85.1 ± 0.6% | 80.6 ± 0.6% | 0.816 ± 0.002 | **0.863 ± 0.009** | **70.4 ± 0.8%** | 0.627 ± 0.008 |
| | dihedral8 | 84.7 ± 1.3% | 80.7 ± 1.0% | 0.814 ± 0.013 | 0.864 ± 0.007 | 70.4 ± 1.2% | 0.620 ± 0.006 |

## Đối chiếu tiêu chí (flip4 so với không TTA)

| | ROC-AUC | Concept acc. | Chỉ số giảm nhiều nhất | Đạt |
|---|---|---|---|:---:|
| Black-box | +0.027, tăng ở 3/3 seed | — | không có | ✓ |
| Hard CBM | +0.013, tăng ở 2/3 seed (seed 2026 bằng) | +1.2 điểm, 3/3 seed | không có | ✓ |
| Soft CBM | +0.010, 3/3 seed | +1.0 điểm, 2/3 seed | Accuracy −0.0099 (sát ngưỡng −0.01) | ✓ |
| ECBM | +0.005, 3/3 seed (sát ngưỡng +0.005) | +0.7 điểm, 3/3 seed | không có | ✓ |

**Quyết định theo quy tắc chốt trước: dùng TTA flip4 trong cấu hình cuối cho cả 4 mô hình.**

## Nhận xét

- **TTA giúp rõ nhất ở Black-box**: ROC-AUC 0.797 → 0.824, Macro F1 0.754 → 0.768.
- **Với các mô hình concept, chẩn đoán và concept cùng tăng một chút.** Concept Macro F1 tăng ở cả 3 mô hình, nên mức tăng concept accuracy không phải do dồn dự đoán về trạng thái phổ biến.
- **Cách lấy trung bình giữ được tính giải thích.** Concept được lấy trung bình trước khi chẩn đoán, nên concept hiển thị cho bác sĩ đúng là concept mà chẩn đoán đã dùng.
- **dihedral8 không tốt hơn flip4.** Điều này khớp với dự đoán: xoay 90° chuyển phần viền đệm sang hai bên, kiểu ảnh mô hình chưa gặp khi train.
- **Chi phí:** suy luận chậm gấp 4 lần. Toàn bộ thí nghiệm chạy khoảng 2,5 phút trên máy cá nhân.

## Hạn chế

- **Hai chỗ sát ngưỡng:** ROC-AUC của ECBM (+0.0052, ngưỡng +0.005) và Accuracy của Soft CBM (−0.0099, ngưỡng −0.01). Mức tăng của ECBM nhỏ hơn độ lệch chuẩn giữa các seed.
- **Ngưỡng chẩn đoán chọn và chấm trên cùng validation**, giống nhau ở hai nhánh.
- **3 seed, không kiểm định thống kê.** Mức tăng thật trên test có thể nhỏ hơn.

**Số liệu và mã nguồn:**
- Số liệu đầy đủ: [summary.json](summary.json).
- Script: `experiments/run_tta.py`.
