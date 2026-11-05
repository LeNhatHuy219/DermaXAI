# ECBM hai chế độ (validation, 3 seed, checkpoint 224)

**Vấn đề.** Với một λ_global chung cho mọi việc, ECBM gặp một đánh đổi:

- λ nhỏ (0.01, như bài báo): concept hiển thị đúng hơn, nhưng bác sĩ sửa concept gần như không đổi được chẩn đoán.
- λ lớn (1): sửa concept có tác dụng, nhưng concept hiển thị bị chẩn đoán kéo lệch.

**Ý tưởng.** Tách thành hai chế độ, giống cách code gốc của tác giả chạy (`GradientInference.py`: tự động dùng 1:1:0.01, can thiệp đổi sang 0:0:3).

- **Tự động:** trọng số 1 : 1 : 0.01, dùng cho cả chẩn đoán và 7 concept hiển thị.
- **Can thiệp:** khi bác sĩ sửa concept thì chuyển sang bộ trọng số khác. Có 3 ứng viên, chọn trên validation theo [plan.json](plan.json) được chốt trước khi chạy.

Không train lại, không đọc test. Dùng checkpoint ECBM 224 của quy trình chung.

## Kết quả (mean ± SD qua seed 42/123/2026)

| | Hệ hiện tại (S0: λ = 1 cho mọi việc) | Hai chế độ |
|---|---:|---:|
| Concept accuracy hiển thị (tự động) | 58.6% ± 6.8 | **63.4% ± 3.6** |
| Macro F1 tự động | 0.793 ± 0.009 | 0.790 ± 0.012 |
| ROC-AUC tự động | 0.839 ± 0.014 | 0.831 ± 0.009 |
| Diện tích đường cong can thiệp (Macro F1) | 0.7963 ± 0.0114 | 0.7961 ± 0.0114 |
| Macro F1 khi sửa đủ 7 nhóm | 0.798 ± 0.021 | 0.798 ± 0.021 |

Concept hiển thị tốt hơn ở cả 3 seed: 60.8 → 63.1%, 50.9 → 60.0%, 64.0 → 67.2%. Độ dao động giữa các seed cũng giảm.

**Ba ứng viên cho chế độ can thiệp:**

| Ứng viên | Diện tích đường cong | F1 sửa đủ 7 | ROC-AUC sửa đủ 7 | Số ca đổi nhãn khi sửa đủ 7 |
|---|---:|---:|---:|---:|
| `joint_lambda1` (1:1:1), **được chọn** | **0.7961** | 0.798 | 0.906 | 13.0 |
| `concept_only` (0:1:1) | 0.7915 | **0.820** | 0.927 | 38.3 |
| `global_on_displayed` (gần code tác giả, 0:0:3) | 0.7589 | 0.749 | 0.928 | 37.3 |

## Đối chiếu tiêu chí

| Tiêu chí | Kết quả | Đạt |
|---|---|:---:|
| K1: concept hiển thị ≥ S0 + 2 điểm | 63.4% so với 58.6% (+4.9 điểm) | ✓ |
| K2: đường cong can thiệp ≥ S0 − 0.005 | 0.7961 so với 0.7963 | ✓ |
| K3: Macro F1 tự động ≥ S0 − 0.01 | 0.790 so với 0.793 | ✓ |

**Quyết định: ĐẠT.**

## Nhận xét

- **Tách hai chế độ bỏ được đánh đổi.** Concept hiển thị đúng hơn khoảng 5 điểm, còn hiệu quả can thiệp và chẩn đoán tự động gần như giữ nguyên.
- **Đây là cách tác giả thiết kế.** λ = 0.01 trong bài báo chỉ dành cho bước tự động.
- **Hai ứng viên kia có ROC-AUC cao hơn khi sửa đủ 7 nhóm (0.93), nhưng đường cong kém hơn.** Ngưỡng chọn khi concept còn là dự đoán không còn hợp khi concept đã đúng. Đây chính là hạn chế 5.3 trong báo cáo tổng kết.
- **Trên validation, mức tăng khi sửa concept nhỏ** (0.790 → 0.798). Trên test, với λ = 1, mức tăng trước đây là +0.048.
- **Hạn chế:** validation được dùng cả để chọn ngưỡng lẫn chọn ứng viên, nên số hơi lạc quan.

Kết quả từng seed: `seed42.json`, `seed123.json`, `seed2026.json`; tổng hợp: `summary.json`. Script: `experiments/run_ecbm_two_modes.py`.
