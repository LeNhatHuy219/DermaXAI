# Kết quả concept intervention của Soft CBM

Seeds: [42, 123, 2026]; 395 ca test; 128 subsets cho mỗi seed.

Thay toàn bộ group được sửa bằng ground-truth one-hot. Giữ nguyên các groups chưa sửa, diagnosis head và threshold validation của từng seed; không huấn luyện lại.

Metrics được tính riêng cho mỗi subset trên toàn bộ test, rồi lấy trung bình theo m, sau đó tổng hợp mean ± sample SD giữa seeds. SD này không phải khoảng tin cậy theo mẫu.

| Số groups sửa | Balanced Accuracy (%) | Macro-F1 | Ca tốt lên (TB) | Ca xấu đi (TB) |
|---|---:|---:|---:|---:|
| 0 | 71.92 ± 1.37 | 0.7333 ± 0.0119 | 0.00 | 0.00 |
| 1 | 72.75 ± 1.62 | 0.7435 ± 0.0156 | 17.52 | 13.67 |
| 2 | 73.66 ± 2.34 | 0.7546 ± 0.0203 | 24.86 | 17.10 |
| 3 | 73.96 ± 3.22 | 0.7580 ± 0.0285 | 29.75 | 20.45 |
| 4 | 74.06 ± 4.05 | 0.7592 ± 0.0349 | 33.28 | 23.24 |
| 5 | 74.12 ± 4.37 | 0.7595 ± 0.0367 | 35.86 | 25.78 |
| 6 | 74.02 ± 4.93 | 0.7584 ± 0.0409 | 37.90 | 28.00 |
| 7 | 72.74 ± 5.27 | 0.7472 ± 0.0442 | 38.33 | 30.33 |

![Intervention curve](intervention_curve.png)

Ca tốt lên/xấu đi được so với m=0 của cùng seed; ở m=1…6 là trung bình qua subsets và seeds, không phải số ca duy nhất. Ở m=7, mỗi seed có một subset sửa đủ bảy groups.

| Sửa toàn bộ 7 groups | Balanced Accuracy (%) | Macro-F1 | Tốt lên | Xấu đi |
|---|---:|---:|---:|---:|
| Seed 42 | 66.96 | 0.6975 | 31 | 27 |
| Seed 123 | 77.26 | 0.7825 | 43 | 30 |
| Seed 2026 | 74.00 | 0.7614 | 41 | 34 |

Sửa đủ bảy groups làm Balanced Accuracy trung bình thay đổi +0.82 điểm phần trăm (71.92% → 72.74%). Đây là kết quả mô tả trên các checkpoints hiện có.

Diagnosis head Soft CBM được học từ soft probabilities; khi intervention, input của head chuyển sang ground-truth one-hot ở các groups được sửa. Sự khác biệt representation là một giả thuyết cần khảo sát để giải thích kết quả, chưa được xác lập là nguyên nhân. Các số liệu này chưa tự chứng minh concept leakage, ảnh hưởng của inconsistency hoặc ưu thế của ECBM.

Các file đi kèm: `summary.json`, `curve.csv`, `per_concept.csv`, `full_intervention_cases.csv` và JSON chi tiết cho từng seed.

`per_concept.csv` đo tác động sửa riêng từng group, không đo tầm quan trọng tổng quát hoặc quan hệ nhân quả. Với Soft CBM, groups chưa sửa giữ nguyên dự đoán nên accuracy của chúng không thay đổi khi giữ cùng group. Trung bình qua các group còn lại có thể thay đổi theo subset; tại m=7 chỉ số này không áp dụng.

Đây là mô phỏng hiệu chỉnh annotation. Kết quả sửa toàn bộ concepts dùng diagnosis head Soft CBM đã học và không đồng nhất với Oracle LR được huấn luyện riêng.
