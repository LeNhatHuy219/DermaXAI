# Ánh xạ tên mô hình cũ (M0–M5) sang tên mới

Từ ngày 28/10/2026 code và run mới dùng tên thật. Kết quả, checkpoint và báo cáo đã ghi trước đó **giữ nguyên đường dẫn và nội dung**; bảng dưới dùng để đọc chúng theo tên mới. Code đọc được tên cũ qua `src/naming.py` (`canonical_model`).

| Tên cũ | Tên mới (code/thư mục) | Tên hiển thị | Vai trò |
|---|---|---|---|
| M0 Majority | — (đã bỏ) | — | Kết quả cũ: `results/m0_majority_results.json` |
| M1 EfficientNet-B0 | `efficientnet_b0` (model black-box mang tên backbone) | EfficientNet-B0 | Model chính |
| M2 Oracle LR | `oracle_lr` | Oracle LR | Tham chiếu: concept thật → chẩn đoán |
| M2 Oracle MLP | — (đã bỏ) | — | Kết quả cũ: `results/f1_macro/m2_mlp/` |
| M3 Soft Joint CBM | `soft_cbm` | Soft CBM | Model chính |
| M4-ST Hard Joint CBM (straight-through) | `hard_cbm` | Hard CBM | Model chính |
| M5 Categorical ECBM | `ecbm` | ECBM | Model chính |

## Đường dẫn và định danh

| Mô hình | Kết quả cũ (F1) | Checkpoint cũ (F1) | Trường `model` cũ | Runner cũ → mới |
|---|---|---|---|---|
| EfficientNet-B0 | `results/f1_macro/m1/<cấu hình>/` | `checkpoints/f1_macro/m1_efficientnet_b0_*_ckptf1macro_best.pth` | `M1_BlackBox_EfficientNetB0` | `run_m1.py` → `run_blackbox.py` |
| Oracle LR | `results/f1_macro/m2_lr/logistic_regression/` | `checkpoints/f1_macro/m2_oracle_lr*_ckptf1macro_best.joblib` | `M2_Oracle_LogisticRegression` | `run_m2_lr.py` → `run_oracle_lr.py` |
| Soft CBM | `results/f1_macro/m3/<cấu hình>/` | `checkpoints/f1_macro/m3_soft_joint_cbm_*_ckptf1macro_best.pth` | `M3_SoftJointCBM` | `run_m3.py` → `run_soft_cbm.py`; `run_m3_interventions.py` → `run_soft_cbm_interventions.py` |
| Hard CBM | `results/f1_macro/m4_st/<cấu hình>/` | `checkpoints/f1_macro/m4_hard_joint_cbm_*_ckptf1macro_best.pth` | `M4_HardJointCBM` | `run_m4_st.py` → `run_hard_cbm.py`; `run_m4_st_interventions.py` → `run_hard_cbm_interventions.py`; `m4_common.py` → `hard_cbm_common.py` |
| ECBM | `results/local/m5_paper/` (validation, bản gốc) | `checkpoints/local/m5_paper/` | `M5_CategoricalECBM` | `run_m5.py` → `run_ecbm.py` |

Bộ checkpoint chọn theo BAcc (`results/bacc/`, `checkpoints/bacc/`) theo cùng ánh xạ tên model. Các study cục bộ `results/local/m1_*` và `checkpoints/local/m1_*` thuộc EfficientNet-B0 black-box.

Run mới ghi tại `checkpoints/derm7pt/<model>/<cấu hình>/seed<seed>.pth` và `results/derm7pt/<model>/<cấu hình>/seed<seed>.json` (thêm `_ckptbacc` khi chọn checkpoint theo BAcc, `_thrbacc` khi chọn ngưỡng theo BAcc), với trường `model` là tên mới. Model black-box lấy id backbone làm `<model>` (`efficientnet_b0`); CBM/ECBM ghi backbone ở đầu `<cấu hình>` (`efficientnet_b0_mlp128_e50_headlr0.001`).
