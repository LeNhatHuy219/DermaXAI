# Train lại với schema concept đã gộp (D4a) — validation, 3 seed

**Hướng:** data-centric, giải quyết mất cân bằng ở tầng concept. Kèm theo là phép đo trần của dataset.

**Câu hỏi.** Một số trạng thái concept có rất ít ca train (0–5 ca). Ví dụ pigmentation "localized regular" không có ca train nào nhưng có 3 ca trong test. Câu hỏi đặt ra: gộp các trạng thái hiếm theo cách chấm 7-point checklist rồi train lại thì mô hình có giữ được chẩn đoán và nhận concept tốt hơn không?

**Cách làm.** Tiêu chí được ghi trong [plan.json](plan.json) trước khi train.

- **Schema D4a:**
  - pigmentation 5 → 3 trạng thái (không có / đều / không đều);
  - regression structures 4 → 2 (không có / có);
  - vascular structures 8 → 3 (không có / điển hình / không điển hình);
  - tổng cộng 28 → 19 trạng thái; ECBM chỉ còn duyệt 972 tổ hợp thay vì 8,640.
- **Train lại:** Hard CBM, Soft CBM và ECBM, ảnh 224, trên cùng máy với bộ chính thức. Chỉ đổi file nhãn (`data/d4a/`), mọi thiết lập khác giữ nguyên. Black-box không dùng concept nên không train lại.
- **Mốc so sánh:** chính các mô hình chính thức, lấy xác suất concept validation đã lưu rồi gộp sau khi train (cộng xác suất trong mỗi nhóm). Cả hai nhánh được chấm trên cùng schema D4a, bằng cùng một đoạn code.
- **Không đọc test.**

## Kết quả (validation, trung bình ± độ lệch chuẩn qua seed 42/123/2026)

| Mô hình | Cách làm | Accuracy | BAcc | Macro F1 | ROC-AUC | Concept acc. | Concept Macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Hard CBM | Gộp sau khi train | 82.9 ± 1.6% | 79.7 ± 1.1% | 0.797 ± 0.013 | 0.815 ± 0.014 | 67.2 ± 0.4% | 0.618 ± 0.003 |
| | **Train lại D4a** | 84.9 ± 1.7% | 80.5 ± 1.8% | 0.814 ± 0.019 | **0.855 ± 0.026** | **68.4 ± 0.3%** | 0.613 ± 0.008 |
| Soft CBM | Gộp sau khi train | 84.9 ± 2.0% | 78.6 ± 2.5% | 0.806 ± 0.026 | 0.837 ± 0.019 | 64.5 ± 0.3% | 0.596 ± 0.005 |
| | **Train lại D4a** | 84.4 ± 1.6% | 77.2 ± 1.9% | 0.795 ± 0.020 | 0.831 ± 0.022 | **67.7 ± 1.8%** | **0.611 ± 0.021** |
| ECBM | Gộp sau khi train | 83.1 ± 1.7% | 77.9 ± 1.4% | 0.790 ± 0.012 | 0.831 ± 0.008 | 61.9 ± 6.4% | 0.576 ± 0.046 |
| | **Train lại D4a** | 84.7 ± 0.5% | 79.3 ± 1.5% | 0.808 ± 0.010 | **0.858 ± 0.007** | **69.7 ± 1.3%** | **0.620 ± 0.009** |

**Từng nhóm concept: accuracy (Macro F1), gộp sau khi train → train lại D4a**

| Concept | Hard CBM | Soft CBM | ECBM | Luôn đoán trạng thái phổ biến nhất |
|---|---|---|---|---:|
| Pigmentation (gộp) | 61.1 → 63.5 (0.57 → 0.58) | 56.0 → 61.2 (0.53 → 0.56) | 54.8 → 62.7 (0.51 → 0.58) | 55.2 |
| Regression structures (gộp) | 67.5 → 70.8 (0.64 → 0.62) | 63.1 → 72.1 (0.61 → 0.64) | 58.3 → 74.4 (0.57 → 0.66) | 74.9 |
| Vascular structures (gộp) | 67.8 → 74.7 (0.51 → 0.54) | 64.7 → 70.9 (0.49 → 0.53) | 49.3 → 74.4 (0.39 → 0.51) | 80.3 |
| Pigment network | 59.4 → 60.9 | 60.1 → 61.9 | 59.4 → 59.1 | 41.4 |
| Streaks | 71.6 → 71.8 | 66.0 → 70.6 | 72.9 → 72.4 | 60.6 |
| Dots and globules | 58.8 → 55.2 | 57.5 → 56.3 | 58.8 → 61.9 | 48.3 |
| Blue-whitish veil | 84.4 → 82.1 | 84.4 → 81.1 | 79.6 → 82.9 | 77.3 |

## Đối chiếu tiêu chí

| Tiêu chí | Hard CBM | Soft CBM | ECBM |
|---|:---:|:---:|:---:|
| G1: chẩn đoán không giảm (AUC tối đa −0.01, Macro F1 tối đa −0.02) | ✓ | ✓ | ✓ |
| G2: concept accuracy và concept Macro F1 không giảm quá 0.01 | ✓ | ✓ | ✓ |
| I: cải thiện concept (+0.01 và tốt hơn ở ≥ 2/3 seed) | accuracy ✓, F1 ✗ | accuracy ✓, F1 ✓ | accuracy ✓, F1 ✓ |

**Quyết định theo quy tắc chốt trước: ĐẠT, dùng schema D4a cho các bước data-centric tiếp theo.**

## Trần của dataset (chỉ train và validation)

| | Schema gốc | D4a |
|---|---:|---:|
| Số hồ sơ concept khác nhau (train) | 176 | 145 |
| Ảnh có hồ sơ mâu thuẫn, train | 18.4% | 23.5% |
| Ảnh có hồ sơ mâu thuẫn, train + validation | 23.7% | 30.0% |
| Melanoma nằm trong hồ sơ mâu thuẫn, train + validation | 45.7% | 55.0% |
| Trần accuracy nếu chẩn đoán chỉ dựa vào concept, train + validation | 92.9% | 90.9% |

"Hồ sơ mâu thuẫn" là bộ 7 concept giống hệt nhau nhưng có cả ca melanoma lẫn ca không phải melanoma. Mô hình chỉ nhìn concept thì không thể đúng hết các ca này. Trần là số ca thuộc nhãn chiếm đa số trong mỗi hồ sơ chia cho tổng số ca. Bài của Nápoles và cộng sự tính trên toàn bộ dataset được 30.3% ảnh mâu thuẫn và trần 92.1%; ở đây không dùng nhãn test nên không tính lại con số đó.

## Nhận xét

- **Train trực tiếp trên schema gộp tốt hơn gộp sau khi train.**
  - Mức tăng tập trung ở 3 nhóm được gộp. Concept accuracy trung bình tăng +1.2 điểm (Hard CBM), +3.2 (Soft CBM) và +7.8 (ECBM).
  - Lý do: mô hình gốc phải gán trọng số loss rất lớn cho các trạng thái chỉ có 0–5 ca train, nên hay báo nhầm các trạng thái hiếm này. Gộp lại thì bỏ được trọng số đó.
  - Vascular structures thay đổi rõ nhất: ECBM tăng từ 49.3% lên 74.4%, và F1 cũng tăng ở cả 3 mô hình.
- **Không phải chỉ do dồn dự đoán về trạng thái phổ biến.** Với Soft CBM và ECBM, concept Macro F1 tăng cùng accuracy. Riêng Hard CBM thì concept Macro F1 không tăng (0.618 → 0.613), nên một phần mức tăng accuracy của Hard CBM có thể là do dồn về trạng thái phổ biến.
- **Chẩn đoán giữ được, và tăng ở hai mô hình.**
  - Hard CBM: ROC-AUC 0.815 → 0.855, tăng ở cả 3 seed.
  - ECBM: ROC-AUC 0.831 → 0.858, tăng ở cả 3 seed, và dao động giữa các seed nhỏ hơn hẳn.
  - Soft CBM giảm nhẹ (0.837 → 0.831), vẫn trong ngưỡng chặn.
  - Kế hoạch không đặt tiêu chí "chẩn đoán tăng", nên đây chỉ là mô tả.
- **ECBM ổn định hơn nhiều.** Độ lệch chuẩn của concept accuracy giảm từ 6.4 xuống 1.3 điểm. Mô hình gốc có một seed concept rất kém (55.2%).

## Hạn chế

- **Không so trực tiếp với số cũ.** Concept accuracy trên schema D4a không so được với số trên schema 28 trạng thái, vì ít lựa chọn hơn thì đoán đúng dễ hơn. Khóa luận cần báo cả hai.
- **Regression và vascular vẫn thấp hơn mốc "luôn đoán không có"** (74.9% và 80.3%). Đây là đánh đổi của trọng số cân bằng: mô hình cố phát hiện trường hợp "có" nên chấp nhận báo nhầm. Loại lỗi có/không này chính là mục tiêu của bước tiếp theo.
- **Trần dataset giảm khoảng 2 điểm** (92.9% → 90.9%), vì gộp nhãn làm nhiều hồ sơ trùng nhau hơn. Trần tính in-sample nên hơi lạc quan: nhiều hồ sơ chỉ có một ảnh.
- **Đánh giá trên validation.** Ngưỡng chẩn đoán và checkpoint đều chọn trên validation, giống nhau ở hai nhánh. Chỉ có 3 seed, không dùng kiểm định thống kê.

## Bước tiếp theo (data-centric)

1. Thêm nhánh dự đoán "có/không" cho từng concept trên schema D4a, nhắm vào loại lỗi nhầm có/không vẫn chiếm đa số.
2. Bổ sung nhãn có/không từ ISIC 2018 Task 2 (CC-0) và PH2. Phải kiểm tra ảnh trùng với Derm7pt trước khi dùng.

**Số liệu và mã nguồn:**
- Số liệu đầy đủ: [summary.json](summary.json).
- Script: `experiments/build_concept_schema_d4a.py` (tạo `data/d4a/`), `experiments/analyze_d4a_retrain.py`.
- Kết quả từng lần chạy: `results/derm7pt/{hard_cbm,soft_cbm,ecbm}/<cấu hình>_d4a/seed*.json`.
