# Kết quả thí nghiệm

Thư mục này chứa kết quả do code sinh ra (JSON, CSV, hình). Phần diễn giải nằm trong [báo cáo tổng kết](../docs/reports/status_report.md) và [báo cáo các hướng cải tiến ECBM đã thử](../docs/reports/improvement_attempts.md).

## Bộ kết quả hiện hành: `derm7pt/` (quy trình chung, từ 29/10/2026)

| Đường dẫn | Nội dung |
|---|---|
| [`derm7pt/test_plan.json`](derm7pt/test_plan.json) | Kế hoạch chấm test: checkpoint (SHA-256), ngưỡng và chế độ suy luận đã khóa |
| [`derm7pt/test_summary.json`](derm7pt/test_summary.json) | Bảng test chính của bốn mô hình |
| [`derm7pt/efficientnet_b0/linear_e50_headlr0.001/`](derm7pt/efficientnet_b0/linear_e50_headlr0.001/) | Black-box, ba seed |
| [`derm7pt/soft_cbm/efficientnet_b0_mlp128_e50_headlr0.001/`](derm7pt/soft_cbm/efficientnet_b0_mlp128_e50_headlr0.001/) | Soft CBM, ba seed; `intervention/` là kết quả can thiệp concept |
| [`derm7pt/hard_cbm/efficientnet_b0_mlp128_e50_headlr0.001/`](derm7pt/hard_cbm/efficientnet_b0_mlp128_e50_headlr0.001/) | Hard CBM, ba seed; `intervention/` là kết quả can thiệp concept |
| [`derm7pt/ecbm/efficientnet_b0_unified_exact_e50/`](derm7pt/ecbm/efficientnet_b0_unified_exact_e50/) | ECBM, checkpoint dùng cho kết quả chính |
| `derm7pt/ecbm/efficientnet_b0_unified_gradient*_e50/` | ECBM chọn checkpoint theo suy luận gradient, để đối chiếu |
| [`derm7pt/ecbm/lambda_global_study/`](derm7pt/ecbm/lambda_global_study/report.md) | Chọn λ_global trên validation |
| [`derm7pt/ecbm/test_modes/`](derm7pt/ecbm/test_modes/report.md) | ECBM trên test với ba chế độ suy luận, kèm can thiệp concept |
| [`derm7pt/ecbm/limitations/`](derm7pt/ecbm/limitations/numbers.json) | Số liệu và hình phân tích hạn chế của ECBM |
| [`derm7pt/ecbm/align_pilot/`](derm7pt/ecbm/align_pilot/summary.json) | Pilot đồng nhất huấn luyện và suy luận (chỉ validation) |
| [`derm7pt/learning_curves_validation.png`](derm7pt/learning_curves_validation.png) | Đường học trên validation |

## Lưu trữ: `archive/`

Kết quả của bộ thí nghiệm trước quy trình chung (tên M0–M4, 10–20 epochs, chọn checkpoint theo BAcc hoặc Macro F1). **Không dùng làm kết quả chính.** Mô tả ở [`archive/README.md`](archive/README.md), cách chạy ở [nhật ký thí nghiệm cũ](../docs/experiment_log.md), đối chiếu tên cũ và mới ở [`NAME_MAPPING.md`](NAME_MAPPING.md).

## `local/`

Kết quả chạy thử cục bộ, không đưa lên Git.
