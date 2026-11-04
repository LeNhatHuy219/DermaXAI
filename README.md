# DermaXAI: Explainable AI for Skin Lesion Classification

Hệ thống mã nguồn phục vụ nghiên cứu và đánh giá các mô hình Trí tuệ Nhân tạo Có Thể Giải Thích (Explainable AI - XAI) trong bài toán phân loại tổn thương da (Melanoma vs Non-Melanoma) trên tập dữ liệu Derm7pt:

| Mô hình | Vai trò | Trạng thái | Runner |
|---|---|---|---|
| EfficientNet-B0 (black-box) | Model chính | Quy trình chung: đã chấm test ba seeds | `experiments/run_blackbox.py` |
| Soft CBM | Model chính | Quy trình chung: đã chấm test và intervention ba seeds | `experiments/run_soft_cbm.py` |
| Hard CBM (straight-through) | Model chính | Quy trình chung: đã chấm test và intervention ba seeds | `experiments/run_hard_cbm.py` |
| ECBM (categorical) | Model chính | Quy trình chung, λ_global=1 chọn trên validation: đã chấm test và intervention ba seeds | `experiments/run_ecbm.py`, `experiments/run_ecbm_modes.py` |
| Oracle LR | Tham chiếu: concept thật → chẩn đoán, không xếp hạng | Đã có kết quả | `experiments/run_oracle_lr.py` |

Kết quả, ưu điểm và hạn chế nằm ở mục [Kết quả chính](#ket-qua-chinh) và trong [báo cáo tổng kết](docs/reports/status_report.md).

<a id="ket-qua-chinh"></a>

## Kết quả chính (quy trình chung, test 395 ca)

Mean ± SD qua ba seeds. ECBM dùng ba bộ trọng số năng lượng lúc suy luận (E_class/E_concept/E_global): **λ=1** là 1/1/1, chọn trên validation ([study λ_global](results/derm7pt/ecbm/lambda_global_study/report.md)); **λ paper** là 1/1/0.01; **chỉ concept** là 0/1/1, chỉ để đối chiếu.

| Mô hình | Acc % | BAcc % | Macro F1 | ROC-AUC | Concept acc % |
|---|---:|---:|---:|---:|---:|
| Black-box | 80.84 ± 0.15 | 73.92 ± 2.25 | 0.7428 ± 0.0119 | 0.8433 ± 0.0071 | — |
| Soft CBM | 81.10 ± 0.58 | 71.92 ± 1.37 | 0.7333 ± 0.0119 | 0.8674 ± 0.0058 | 65.77 ± 0.67 |
| Hard CBM | 78.40 ± 1.39 | 74.11 ± 3.07 | 0.7285 ± 0.0216 | 0.8394 ± 0.0201 | 66.71 ± 0.27 |
| ECBM λ=1 (chính) | 80.17 ± 1.02 | 72.16 ± 3.70 | 0.7282 ± 0.0267 | 0.8535 ± 0.0170 | 58.65 ± 7.76 |
| ECBM λ paper | 79.07 ± 1.92 | 71.86 ± 3.40 | 0.7198 ± 0.0212 | 0.8479 ± 0.0129 | 63.92 ± 4.23 |
| ECBM chỉ concept (phụ) | 76.46 ± 3.78 | 72.92 ± 1.21 | 0.7111 ± 0.0222 | 0.8329 ± 0.0225 | 56.02 ± 8.55 |

Intervention bằng concept GT (Macro F1, ngưỡng đóng băng):

| Mô hình | Không sửa | Sửa đủ 7 nhóm | Curve AUC |
|---|---:|---:|---:|
| Soft CBM | 0.7333 | 0.7472 | 0.7534 |
| Hard CBM | 0.7285 | 0.7925 | 0.7694 |
| ECBM λ=1 | 0.7282 | 0.7764 | 0.7506 |
| ECBM λ paper | 0.7198 | 0.7233 | 0.7214 |
| ECBM chỉ concept | 0.7111 | 0.7751 | 0.7713 |

Tóm tắt:

- Với λ paper, nhánh concept gần như không tác động: chẩn đoán trùng nhánh ảnh E_class ở 99.9% ca, sửa đủ 7 concept chỉ đổi nhãn trung bình 1/395 ca. λ=1 làm intervention có tác dụng, nhưng vẫn kém Hard CBM, và chẩn đoán tự động vẫn trùng E_class ở 97% ca.
- Concept accuracy của ECBM thấp hơn CBM, và giảm thêm khi tăng λ_global.

Báo cáo liên quan:

- [Báo cáo tổng kết](docs/reports/status_report.md): những gì đã làm, số liệu, ưu điểm, hạn chế, và phân tích 6 hạn chế của ECBM kèm 7 hình (mục 5).
- [Các hướng cải tiến ECBM đã thử](docs/reports/improvement_attempts.md): ý tưởng, thực nghiệm và lý do chưa thành công.
- [Kết quả ECBM theo chế độ suy luận](results/derm7pt/ecbm/test_modes/report.md).

## Cấu trúc thư mục

```text
src/           mô hình, dữ liệu, metric, quy tắc chọn checkpoint/ngưỡng, tiện ích chạy (runtime.py)
experiments/   runner train/test và intervention (run_*), phân tích, tổng hợp, audit dữ liệu
tests/         kiểm tra protocol và pipeline dữ liệu
data/          manifest.csv và label_mapping.json (không chứa ảnh)
notebooks/     EDA và notebook chạy lại trên Kaggle
results/       kết quả do code sinh ra: derm7pt/ là bộ hiện hành, archive/ là quy trình cũ
docs/          tài liệu đọc: reports/ (báo cáo), papers/ (tóm tắt paper), nhật ký thí nghiệm cũ
checkpoints/   weights, không đưa lên Git (trừ README và manifest)
Derm7pt/       dữ liệu gốc tải riêng, không đưa lên Git
```

Mục lục kết quả ở [`results/README.md`](results/archive/README.md).

## Cài đặt

Chạy các lệnh dưới đây trong thư mục repo. Bộ test hiện được kiểm tra với Python 3.11.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Trên Windows, kích hoạt môi trường bằng `.venv\Scripts\Activate.ps1` trong PowerShell.

**Dữ liệu được tải riêng** từ [trang chính thức Derm7pt](https://derm.cs.sfu.ca/). Giải nén để có `Derm7pt/release_v0/images/` và `Derm7pt/release_v0/meta/` bên trong thư mục repo. `data/manifest.csv` và `data/label_mapping.json` đã có sẵn; đường dẫn ảnh trong manifest là đường dẫn tương đối nên có thể dùng trên máy khác.

README đi kèm Derm7pt ghi rõ ảnh không được phân phối lại. Thư mục dữ liệu, checkpoint và log cục bộ nằm trong `.gitignore`. Output notebook chứa ảnh mẫu Derm7pt cũng cần được giữ ở máy cục bộ.

## Quy trình chung và cách chạy

Theo góp ý của giảng viên, mọi model ảnh dùng cùng một công thức train và chỉ khác kiến trúc. Đây là giá trị mặc định của các runner:

| Thành phần | Thiết lập |
|---|---|
| Ngân sách | 50 epochs cố định, không dừng sớm; batch 16 |
| Optimizer | AdamW, backbone LR 1e-4, phần head LR 1e-3, weight decay 0.01; CosineAnnealingLR về 1e-6 |
| Dữ liệu | `legacy_letterbox`, trọng số lớp và trọng số trạng thái concept tính từ train, `num_workers` 0 |
| Chọn checkpoint | Epoch có validation Macro F1@0.5 cao nhất |
| Chọn ngưỡng | Ngưỡng tối đa validation Macro F1, hòa chọn gần 0.5; đóng băng cho test |
| Kiến trúc | Black-box EfficientNet-B0: head Linear; Soft/Hard CBM: head MLP128; ECBM: ba năng lượng (`--recipe unified`) |

Run cũ chọn ngưỡng theo BAcc vẫn đọc được; `--threshold_metric balanced_accuracy` tái lập quy tắc cũ (thư mục cấu hình thêm `_thrbacc`). ECBM theo công thức gốc của paper chạy bằng `--recipe paper`.

```bash
# Train/validation (không đọc test), ba seeds mỗi model
for s in 42 123 2026; do python3 experiments/run_blackbox.py --seed "$s" --skip_test || break; done
for s in 42 123 2026; do python3 experiments/run_soft_cbm.py --seed "$s" --skip_test || break; done
for s in 42 123 2026; do python3 experiments/run_hard_cbm.py --seed "$s" --skip_test || break; done
for s in 42 123 2026; do python3 experiments/run_ecbm.py --seed "$s" --skip_test || break; done

# ECBM, chỉ validation: chọn λ_global theo luật ghi trước, rồi đóng băng ngưỡng của ba chế độ suy luận
python3 experiments/run_ecbm_lambda_study.py
python3 experiments/run_ecbm_modes.py --stage freeze

# Chấm test bằng checkpoint/ngưỡng đã đóng băng; --overwrite ghép phần test vào JSON và giữ phần validation
for m in blackbox soft_cbm hard_cbm; do
  for s in 42 123 2026; do python3 "experiments/run_$m.py" --mode test --seed "$s" --overwrite || break; done
done
python3 experiments/run_soft_cbm_interventions.py
python3 experiments/run_hard_cbm_interventions.py
python3 experiments/run_ecbm_modes.py --stage test    # ECBM: chẩn đoán và intervention cho ba chế độ
python3 experiments/summarize_unified_test.py         # tổng hợp bảng test → results/derm7pt/test_summary.json
python3 experiments/analyze_ecbm_limitations.py       # validation tính mới; phần test chỉ đọc kết quả đã lưu
python3 experiments/run_oracle_lr.py   # tham chiếu: train vài giây rồi tự chấm test
```

Kết quả ECBM chính được chấm bằng `run_ecbm_modes.py`, không dùng `run_ecbm.py --mode test`. Test của bộ kết quả này đã chấm một lần ngày 31/10/2026 theo [`test_plan.json`](results/derm7pt/test_plan.json); các lệnh chấm test trên chỉ dùng để tái tạo.

Tên mô hình cũ M0–M5 và đường dẫn tương ứng được đối chiếu trong [bảng ánh xạ tên](results/NAME_MAPPING.md).
Run mới ghi checkpoint tại `checkpoints/derm7pt/<model>/<cấu hình>/seed<seed>.pth` và JSON tại `results/derm7pt/<model>/<cấu hình>/seed<seed>.json`.

## Chạy lại trên Kaggle

Notebook [`notebooks/reproduce_derm7pt.ipynb`](notebooks/reproduce_derm7pt.ipynb) lấy code theo đúng commit của báo cáo tổng kết, nối Derm7pt và checkpoint từ phần Input, rồi chạy một trong ba chế độ:

- `check`: kiểm tra dữ liệu và hash checkpoint so với `test_plan.json`, không chấm test;
- `reproduce_test`: chấm lại test bằng checkpoint và ngưỡng đã khóa;
- `train`: train lại từ đầu các mô hình trong `TRAIN_MODELS` với các seed trong `TRAIN_SEEDS` (mặc định 4 mô hình × 3 seed, khoảng 2–4 giờ trên T4), chỉ train/validation, rồi so với kết quả validation đã lưu.

Dataset Derm7pt và checkpoint trên Kaggle để private. Notebook clone repo vào `/tmp` để Kaggle không lưu ảnh làm output.

## Chạy lại mà không ghi đè kết quả đã có

Các bảng trước quy trình chung được lưu trong `results/archive/` (xem [nhật ký thí nghiệm cũ](docs/experiment_log.md)). Để chạy lại và giữ nguyên các kết quả đã có, ghi JSON mới vào `results/local/`:

```bash
python experiments/run_oracle_lr.py --checkpoint_path checkpoints/local/oracle_lr_best.joblib --results_path results/local/oracle_lr_results.json
python experiments/run_blackbox.py --seed 42 --checkpoint_path checkpoints/local/efficientnet_b0_seed42_best.pth --results_path results/local/efficientnet_b0_seed42_results.json
python experiments/run_soft_cbm.py --seed 42 --checkpoint_path checkpoints/local/soft_cbm_seed42_best.pth --results_path results/local/soft_cbm_seed42_results.json
```

Các lệnh trên tải pretrained ImageNet weights khi cần và chạy train → validation selection → test. Checkpoint được tải về hoặc tạo tại máy người chạy; repo không kèm trọng số. Lệnh intervention cần checkpoint Soft CBM và test export tương ứng. Các đường dẫn tuyệt đối trong JSON cũ ghi lại nơi chạy thí nghiệm ban đầu, không phải đường dẫn cần tạo trên máy mới.

## Kiểm tra code

```bash
python -B -m unittest discover -s tests -p 'test_*.py' -v
python -B tests/test_data_pipeline.py
python -B experiments/audit_pipeline.py
```

Tests kiểm tra mapping/LF–CRLF, bảo vệ artifact, train → frozen test, giữ validation khi re-export, hard forward, gradient straight-through, `--skip_test`/`--no_save`, M4 checkpoint identity, hard export và concept metric integrity, 128 intervention subsets, tổng hợp ba seeds và regression report M3. Các smoke tests dùng mạng nhỏ trong thư mục tạm; không tạo kết quả thực nghiệm chính. Chạy trong môi trường đã cài dependencies của dự án.

Mỗi model có file test riêng: `tests/test_blackbox_protocol.py`, `tests/test_oracle_lr_protocol.py`, `tests/test_soft_cbm_protocol.py`, `tests/test_soft_cbm_upgrade.py`, `tests/test_hard_cbm_protocol.py`, `tests/test_ecbm_protocol.py`. Các kiểm tra chung giữa nhiều runners nằm trong `tests/test_protocol_safety.py`.

## Tài liệu

- [Bối cảnh đề tài, cơ sở y khoa, dữ liệu và kiến trúc](docs/project_background.md)
- [Báo cáo tổng kết](docs/reports/status_report.md) và [các hướng cải tiến ECBM đã thử](docs/reports/improvement_attempts.md)
- [Nhật ký thí nghiệm trước quy trình chung](docs/experiment_log.md)
- [Tóm tắt paper ECBM](docs/papers/ecbm_paper_card.md)

## Tài liệu tham khảo

1. Koh, P. W., et al. *"Concept Bottleneck Models."* International Conference on Machine Learning (ICML), 2020.
2. Kawahara, J., et al. *"Seven-Point Checklist and Skin Lesion Classification Using Multitask Multimodal Neural Nets."* IEEE Journal of Biomedical and Health Informatics (JBHI), 2019.
3. Nápoles, G., Grau, I. & Salgueiro, Y. *"Concept inconsistency in dermoscopic concept bottleneck models: a rough-set analysis of the Derm7pt dataset."* Scientific Reports (2026).
4. Patricio, C., et al. *"Coherent Concept-based Explanations for Skin Lesion Analysis."* (2023–2025).
