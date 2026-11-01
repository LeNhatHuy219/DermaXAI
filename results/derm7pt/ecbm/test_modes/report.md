# ECBM trên test (395 ca, 3 seeds), ba chế độ suy luận exact

Ngưỡng chốt trên validation trong [frozen_modes.json](frozen_modes.json), theo [test_plan.json](../../test_plan.json).

| Chế độ | Vai trò | Macro F1 | BAcc | ROC-AUC | Concept F1 | F1 sửa đủ 7 | Curve AUC F1 | Đổi nhãn m=7 | Trùng E_class |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ecbm_full | model chính | 0.7282 ± 0.0267 | 72.16% ± 3.70 | 0.8535 ± 0.0170 | 0.4805 ± 0.0435 | 0.7764 ± 0.0250 | 0.7506 ± 0.0224 | 24.3 | 96.96% ± 1.75 |
| ecbm_paper | tham chiếu: λ suy luận của paper | 0.7198 ± 0.0212 | 71.86% ± 3.40 | 0.8479 ± 0.0129 | 0.5013 ± 0.0233 | 0.7233 ± 0.0223 | 0.7214 ± 0.0217 | 1.0 | 99.92% ± 0.15 |
| ecbm_concept_only | báo cáo phụ: chẩn đoán chỉ qua concept (bỏ E_class) | 0.7111 ± 0.0222 | 72.92% ± 1.21 | 0.8329 ± 0.0225 | 0.4598 ± 0.0583 | 0.7751 ± 0.0090 | 0.7713 ± 0.0061 | 91.3 | 80.59% ± 2.11 |

Mean ± sample SD qua seeds 42/123/2026. Can thiệp thay nhóm concept bằng trạng thái GT, ngưỡng đóng băng.
