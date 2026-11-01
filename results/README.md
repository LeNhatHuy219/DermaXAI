# Kết quả thí nghiệm

Bảng kết quả chính đánh giá chất lượng phân loại chẩn đoán (Melanoma vs Non-Melanoma) và các khái niệm y khoa (Concepts) trên tập dữ liệu Derm7pt qua các mô hình đối chứng M0–M4.

Các thư mục được phân theo **tiêu chí chọn checkpoint**, đọc từ metadata của run. Mỗi JSON chứa cả BAcc và Macro F1 trong bảng metrics.

## Kết quả chính dùng cho khóa luận

**Bảng test chính là sáu model/cohort dưới đây**, theo các cấu hình đã chốt trong [Cấu hình đã chốt và thứ tự cải tiến](f1_macro/all_models_comparison.md#baseline-decisions). M3/M4 dùng cohort MLP128 20 epochs; các phiên bản Linear 10 epochs được lưu riêng để đối chiếu. Bảng này xác định cohort đã chốt, không chọn seed hoặc cấu hình theo điểm test cao nhất.

| Model | Cấu hình chính | Test Macro F1 | Nguồn số liệu |
|---|---|---:|---|
| M0 — Majority | Không huấn luyện | 0.4267 | [JSON nguồn](m0_majority_results.json) |
| M1 — Black-box | EfficientNet-B0, 10 epochs | 0.7078 ± 0.0118 | [JSON nguồn](f1_macro/m1/efficientnet_b0_e10/test_summary.json) |
| M2 — Oracle LR | Logistic Regression trên GT concepts | 0.8077 | [JSON nguồn](f1_macro/m2_lr/logistic_regression/seed42.json) |
| M2 — Oracle MLP | MLP trên GT concepts, 100 epochs | 0.8264 ± 0.0107 | [JSON nguồn](f1_macro/m2_mlp/mlp32_e100/test_summary.json) |
| M3 — Soft CBM | MLP128, 20 epochs | 0.7207 ± 0.0110 | [JSON nguồn](f1_macro/m3/mlp128_e20_headlr0.001/test_summary.json) |
| M4 — Hard CBM (straight-through) | MLP128, 20 epochs | 0.7131 ± 0.0071 | [JSON nguồn](f1_macro/m4_st/mlp128_e20_headlr0.001/test_summary.json) |

Test được đánh giá trên 395 ca. Các model có ba seeds báo cáo mean ± sample SD; M0 xác định và M2 LR chỉ có một run. M2 dùng nhãn concept thật nên là đối chứng oracle. Các checkpoint học được thuộc bộ F1; ngưỡng được chọn riêng trên validation bằng BAcc và giữ cố định khi test.

- **Metrics đầy đủ và nhận xét:** [all_models_comparison.md](f1_macro/all_models_comparison.md), đọc các dòng cohort chính ở bảng trên; báo cáo này còn trình bày phiên bản cũ để đối chiếu.
- **Phụ lục/tham khảo intervention:** [interventions_comparison.md](f1_macro/interventions_comparison.md). Kết quả có GT concept hỗ trợ, thuộc kiểm tra chức năng kế thừa; không dùng thay cho test tự động hoặc nhận là đóng góp phương pháp mới.
- **Kết quả lưu trữ/ablation:** `bacc/` và các cohort Linear 10 epochs. `f1_macro/all_models_comparison.md#legacy-f1` là bảng lịch sử của 16 runs đầu tiên.

## Năm báo cáo Markdown

| Cần đọc | File |
|---|---|
| Bảng chính và nguồn JSON | README này |
| Diagnosis, concept, train/validation, cấu hình đã chốt và cải tiến | [Báo cáo toàn model](f1_macro/all_models_comparison.md) |
| Intervention, từng seed/group và kết quả cũ | [Báo cáo intervention](f1_macro/interventions_comparison.md) |
| Study Linear/MLP128 của M4 | [Study head](bacc/ablation/m4_head_comparison/report.md) |

Các thông tin validation trước test và báo cáo riêng từng model được gộp vào các mục chi tiết có thể mở trong hai báo cáo chung. Số liệu JSON, CSV, PNG/SVG và weights giữ nguyên; đường dẫn JSON được cập nhật theo cấu trúc dưới đây. Runner intervention vẫn có thể tự sinh `report.md` khi chạy một thí nghiệm mới.

Quy tắc chọn checkpoint bằng validation diagnosis Macro F1@0.5 và threshold bằng validation BAcc được áp dụng nhất quán cho các mô hình đối chứng M0–M4. Bộ BAcc lịch sử và các kết quả đã chốt được bảo toàn.

## Cấu trúc và dữ liệu nguồn

| Thư mục | Nội dung hiện có | JSON |
|---|---|---:|
| [bacc/](#bacc) | Bộ BAcc lưu trữ/đối chiếu: baseline, summaries, intervention và ablation | 37 |
| [f1_macro/](#f1-macro) | Bộ kết quả chính: 19 runs có test, 6 summaries test, 2 summaries validation, 2 quyết định chốt và 8 JSON intervention M3/M4-ST MLP128 | 37 |

[M0 majority](m0_majority_results.json) dùng chung cho cả hai bộ vì không có checkpoint để chọn theo BAcc/F1.

[Bảng chính M0–M4](f1_macro/all_models_comparison.md) · [GT-assisted intervention M3 và M4-ST](f1_macro/interventions_comparison.md). Bảng checkpoint BAcc cũ được giữ riêng; cột BAcc vẫn có trong bảng chính như một metric bổ sung.

M1, M2 LR/MLP, M3 và M4-ST mặc định ghi JSON mới theo **tiêu chí → model → cấu hình → seed**. Truyền `--checkpoint_metric balanced_accuracy` để dùng `bacc/`. `--results_path` và `--output_dir` tùy chỉnh vẫn được ưu tiên.

```text
results/
├── m0_majority_results.json
├── f1_macro/
│   ├── m1/efficientnet_b0_e10/
│   ├── m2_lr/logistic_regression/
│   ├── m2_mlp/mlp32_e100/
│   ├── m3/
│   │   ├── linear_e10/
│   │   └── mlp128_e20_headlr0.001/
│   │       ├── seed42.json
│   │       ├── seed123.json
│   │       ├── seed2026.json
│   │       ├── test_summary.json
│   │       ├── validation_summary.json
│   │       ├── frozen_selection.json
│   │       ├── learning_curves.png / learning_curves.svg
│   │       └── intervention/
│   │           ├── seed42.json / seed123.json / seed2026.json
│   │           ├── summary.json
│   │           └── CSV, PNG và SVG
│   └── m4_st/                 # cùng quy tắc với m3
└── bacc/                      # cùng quy tắc; studies nằm trong ablation/
```

`mlp128` là diagnosis head MLP có hidden layer 128 units; `e20` là ngân sách train 20 epochs, không phải best epoch. `headlr0.001` là LR ban đầu của diagnosis head bằng 0.001; backbone/concept predictor dùng LR 0.0001 trong cohort này. Cấu hình khác augmentation, lambda concept loss, LR backbone hoặc weight decay có hậu tố riêng. Các model oracle có cấu hình riêng như `logistic_regression` và `mlp32_e100`.

| File | Dùng để đọc |
|---|---|
| `seed42.json`, `seed123.json`, `seed2026.json` | Kết quả train/validation/test của từng run |
| `test_summary.json` | Metrics test tổng hợp các seeds cùng cohort |
| `validation_summary.json` | Metrics validation của cohort đã chốt |
| `frozen_selection.json` | Quyết định cấu hình/checkpoint trước test |
| `intervention/seed*.json`, `intervention/summary.json` | Chi tiết và tổng hợp GT-assisted concept correction |

Intervention ba seeds nằm trong `<metric>/<model>/<cấu hình>/intervention/`. Các study BAcc cũ được lưu trong `bacc/ablation/<study>/<model>/<cấu hình>/`; plan, frozen selection và báo cáo chung của study ở tầng study. Khi gọi `summarize_seeds.py`, chọn inputs cùng protocol và đặt `--output_path` thành `test_summary.json` trong thư mục cấu hình tương ứng. M0 dùng file chung ở gốc; M2 LR chỉ có một run nên không có summary ba seeds.

Weights được chia tương ứng trong [checkpoints/bacc](../checkpoints/bacc/manifest.json) và [checkpoints/f1_macro](../checkpoints/f1_macro/README.md). Việc gom thư mục chỉ cập nhật đường dẫn tham chiếu; giữ nguyên metrics, predictions, cấu hình, thresholds và checkpoint hashes.


<a id="f1-macro"></a>
### Bộ F1 và cấu hình các nguồn kết quả

<details>
<summary>Mở bảng và ghi nhận chi tiết</summary>

Có **19 run JSONs**, tất cả đã hoàn tất train/validation/test: 13 baseline đầu tiên và ba seeds mỗi cohort M3/M4-ST MLP128 20 epochs. Có thêm sáu summaries test, hai summaries validation và hai quyết định chốt trước test: **29 JSON trong các thư mục model/cấu hình**. Hai thư mục intervention M3/M4-ST MLP128 có tổng cộng sáu run JSONs và hai summaries, đưa tổng số JSON kết quả lên **37**. Biến thể M4-SG đã được bỏ khỏi khóa luận ngày 27/10/2026.

- [Chốt baseline, phần cần cải thiện và thứ tự công việc](f1_macro/all_models_comparison.md#baseline-decisions) (05/10/2026).
- [Rà soát tất cả mô hình, khoảng bất định và đối chứng](f1_macro/all_models_comparison.md) (cập nhật 06/10/2026; gồm test/intervention M0–M4 đã chốt và các studies BAcc).
- [Bảng baseline đầu tiên và nhận xét](f1_macro/all_models_comparison.md#legacy-f1).
- [M3 MLP128: train, validation và test ba seeds](f1_macro/all_models_comparison.md#m3-report).
- [M3 MLP128: intervention ba seeds và diễn giải](f1_macro/interventions_comparison.md#m3-analysis).
- [So sánh GT-assisted intervention M3/M4-ST MLP128](f1_macro/interventions_comparison.md).
- [M4-ST MLP128: intervention ba seeds](f1_macro/interventions_comparison.md#m4-st-details).
- [M3 MLP128: validation ba seeds và quyết định trước test](f1_macro/all_models_comparison.md#m3-validation).
- [M4-ST MLP128: pilot validation ba seeds và frozen checkpoints](f1_macro/all_models_comparison.md#m4-st-validation).
- [M4-ST MLP128: train, validation và frozen test ba seeds](f1_macro/all_models_comparison.md#m4-st-report).

Các JSON nguồn dùng tên `seed<seed>.json`; tiêu chí F1 đã thể hiện trong thư mục `f1_macro/`, nên không lặp hậu tố trong tên JSON. Không gộp cohort MLP128 20 epochs với baseline Linear 10 epochs; summaries phải dùng inputs cùng model, cấu hình và tiêu chí checkpoint. SD giữa seeds dùng `ddof=1`.

Runs chọn checkpoint theo `validation_f1_macro_at_0.5`. Ngưỡng quyết định vẫn được chọn riêng trên validation bằng BAcc; đây là quy tắc độc lập với việc chọn checkpoint.

Cohort M3 MLP128 cùng ngân sách/head/LRs đã hoàn tất ba seeds: mean validation Macro F1@0.5 **0.7727 ± 0.0204**, mean test Macro F1 **0.7207 ± 0.0110**, BAcc **75.94% ± 2.07 điểm %**, AUC **0.8490 ± 0.0173**, concept F1 **0.4839 ± 0.0122**. Checkpoints epochs **19/7/18**, thresholds **0.072132/0.259394/0.090870**, weights và validation giữ nguyên so với quyết định trước test. Test chạy CPU; môi trường test và các đối chiếu được ghi trong [test summary](f1_macro/m3/mlp128_e20_headlr0.001/test_summary.json).

Intervention M3 mới đã hoàn tất **128 subsets × 395 ca × ba seeds**. Sửa đủ bảy groups bằng GT one-hot làm mean F1 **0.7207 → 0.6677**, BAcc **75.94% → 74.53%**; seed 123 giảm mạnh, seed 2026 tăng. Giữ kết quả này riêng với test tự động; weights và thresholds không đổi. [Báo cáo tự sinh](f1_macro/interventions_comparison.md#m3-details) · [Phân tích và đối chiếu](f1_macro/interventions_comparison.md#m3-analysis).

Những lệnh train mới dùng default F1 tự ghi JSON vào `f1_macro/<model>/<cấu hình>/`. Weights nằm trong [checkpoints/f1_macro](../checkpoints/f1_macro/README.md). Các hashes export validation trong quyết định chốt là snapshot trước test; bổ sung test hoặc đổi đường dẫn tham chiếu sẽ thay đổi bytes JSON nguồn nhưng phải giữ nguyên weights, best epoch, threshold và validation. Các hash snapshot trước test giữ nguyên để truy nguyên thời điểm chốt; chúng không phải checksum của JSON sau khi chuyển thư mục.

M4-ST MLP128 cùng head/LRs/ngân sách mới đã train đủ 20 epochs cho seeds 42/123/2026 trên MPS: mean validation F1@0.5 **0.7775 ± 0.0190**, F1 tại ngưỡng BAcc **0.7865 ± 0.0309**, AUC **0.7965 ± 0.0132**, concept F1 **0.5033 ± 0.0078**. Frozen test đã hoàn tất: Macro F1 **0.7131 ± 0.0071**, BAcc **72.66% ± 2.09 điểm %**, AUC **0.8302 ± 0.0092**, concept F1 **0.5069 ± 0.0015**. Best epochs **18/15/19**, thresholds **0.323596/0.427888/0.433949** và weights khớp [frozen selection](f1_macro/m4_st/mlp128_e20_headlr0.001/frozen_selection.json); toàn bộ JSON validation trước test tái tạo đúng SHA-256. Runtime train ST mới ghi PyTorch 2.12, M3 ghi 2.14; runner chưa lưu environment test riêng. Xem provenance và learning curves trong báo cáo validation, kiểm tra frozen test trong báo cáo mới.

Intervention M4-ST mới đã hoàn tất **128 subsets × 395 ca × ba seeds** trên frozen CPU heads. Full GT correction tăng mean F1 **0.7131 → 0.7862 ± 0.0279**, BAcc **72.66% → 77.44%**. Cả ba seeds đều tăng full F1; vẫn có ca/subset correction gây hại. GT-assisted metrics được giữ riêng với automatic test. Lượt kiểm tra đã xác minh head replay, official GT, no-op/remaining-group invariants, case transitions và aggregation. Hệ thống baseline M0–M4 đã hoàn thiện.

</details>


<a id="bacc"></a>
### Bộ BAcc lưu trữ và các nguồn dùng chung

<details>
<summary>Mở bảng và ghi nhận chi tiết</summary>

**Bộ lưu trữ/đối chiếu thử nghiệm bổ sung.** Bảng kết quả chính hiện tại dùng [checkpoint chọn theo Macro F1](f1_macro/all_models_comparison.md). Giữ nguyên bảng, JSON và weights BAcc để truy nguyên các studies LR/head; không gộp hai tiêu chí checkpoint vào cùng mean. Khi viết khóa luận, bảng BAcc có thể đặt ở phần ablation/phụ lục. BAcc vẫn là một metric phụ trong bảng kết quả chính F1.

37 JSON của các runs chọn checkpoint theo `validation_balanced_accuracy_at_0.5`, cùng metadata và phân tích đi kèm.

| Nhóm | Nội dung | JSON |
|---|---|---:|
| Các model/cấu hình baseline | 13 JSON baseline và 4 summaries | 17 |
| [m3/linear_e10/intervention/](f1_macro/interventions_comparison.md#bacc-m3) | M3: ba seeds và summary | 4 |
| [m4_st/linear_e10/intervention/](f1_macro/interventions_comparison.md#bacc-m4) | M4-ST: ba seeds và summary | 4 |
| [ablation/m4_head_comparison/](bacc/ablation/m4_head_comparison/report.md) | Thử Linear/MLP128 của M4-ST, plan, lựa chọn và phân tích | 11 |
| [ablation/m4_evaluation/](f1_macro/all_models_comparison.md#legacy-bacc) | Đánh giá M4 | 1 |

Baseline gồm M1, M2 LR, M2 MLP, M3 và M4-ST. Ba seeds là 42/123/2026; M2 LR có một run seed 42.

Study head dùng summary [M4-ST baseline](bacc/m4_st/linear_e10/test_summary.json); mỗi summary chỉ giữ một bản.

Weights nằm trong `../checkpoints/bacc/`; các trường `checkpoint_path`, `predictions_path` và `source_paths` trỏ đến vị trí hiện tại. CSV, report và hình của các nhóm con được giữ cùng nhóm.

Lưu ý có từ trước khi gom: JSON intervention M4 seed 2026 lưu hash của checkpoint cũ khác weights hiện tại. Kết quả và hash đó được giữ nguyên để bảo toàn nguồn gốc.

</details>
