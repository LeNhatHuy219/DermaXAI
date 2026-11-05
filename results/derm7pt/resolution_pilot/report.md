# Pilot độ phân giải: 224 → 512×768 (validation, seed 42)

**Câu hỏi.** Ảnh Derm7pt gốc khoảng 768×512 nhưng đang được thu về 224 px. Đưa ảnh vào ở độ phân giải gốc thì mô hình có nhận diện concept và chẩn đoán tốt hơn không?

**Thiết kế.** Ngoài kích thước ảnh, mọi thứ khác giữ nguyên quy trình chung: EfficientNet-B0, 50 epoch, batch 16, AdamW, cosine, cùng augmentation, cùng quy tắc chọn checkpoint và ngưỡng.

- Chạy Black-box và Hard CBM, seed 42, trên Kaggle Tesla T4 (khoảng 23 phút mỗi mô hình).
- **Chỉ dùng validation (203 ca), không chấm test.**
- Tiêu chí được ghi trong [plan.json](plan.json) trước khi chạy và so với 6 lần chạy 224 đã có (3 seed × máy cá nhân / Kaggle).

## Kết quả validation

| Mô hình | Ảnh | Accuracy | BAcc | Macro F1 | ROC-AUC | Concept acc. |
|---|---|---:|---:|---:|---:|---:|
| Black-box | 224 (T4, seed 42) | 80.8% | 73.2% | 0.750 | 0.789 | — |
| Black-box | **512×768** | 81.3% | 77.3% | **0.775** | **0.831** | — |
| | *224, 6 lần chạy* | | | *0.736–0.759* | *0.789–0.809* | |
| Hard CBM | 224 (T4, seed 42) | 82.3% | 77.5% | 0.783 | 0.831 | 65.1% |
| Hard CBM | **512×768** | 82.3% | 77.0% | 0.781 | 0.822 | 65.6% |
| | *224, 6 lần chạy* | | | *0.783–0.841* | *0.801–0.831* | *65.1–67.4%* |

## Đối chiếu tiêu chí

| Tiêu chí | Ngưỡng | Kết quả | Đạt |
|---|---|---|:---:|
| C1: concept accuracy của Hard CBM | ≥ 68.13% | 65.59% | ✗ |
| C2: ROC-AUC tăng rõ (Black-box hoặc Hard CBM) | BB ≥ 0.8193 hoặc HC ≥ 0.8408 | BB 0.8305 | ✓ |
| G: Macro F1 không giảm quá mức | BB ≥ 0.7164, HC ≥ 0.7628 | 0.7752 / 0.7805 | ✓ |

**Quyết định theo quy tắc chốt trước: ĐẠT, nhờ C2 và G.** Tuy vậy, chỉ Black-box cải thiện, còn Hard CBM thì không.

## Độ chính xác từng concept (Hard CBM, validation)

| Concept | 224 (T4, seed 42) | 512×768 | 224, 6 lần chạy |
|---|---:|---:|---:|
| Pigment network | 59.6% | **69.5%** | 58.1–61.1% |
| Regression structures | 62.1% | **67.0%** | 60.6–65.5% |
| Dots and globules | 55.7% | 57.1% | 55.2–61.6% |
| Vascular structures | 67.0% | 68.5% | 66.0–68.0% |
| Streaks | 68.5% | *66.5%* | 68.5–74.4% |
| Blue-whitish veil | 85.7% | *80.8%* | 81.3–86.7% |
| Pigmentation | 57.1% | *49.8%* | 57.1–64.0% |

## Nhận xét

- **Ảnh nét hơn giúp nhánh ảnh → chẩn đoán.** Black-box tăng Macro F1 từ 0.750 lên 0.775 và ROC-AUC từ 0.789 lên 0.831. Cả hai đều cao hơn mọi lần chạy ở 224.
- **Concept tốt lên ở một số nhóm và kém đi ở nhóm khác, nên trung bình gần như không đổi.**
  - Tăng rõ ở các cấu trúc dạng vân, chi tiết nhỏ: pigment network (+10 điểm, vượt hẳn mọi lần chạy 224) và regression structures.
  - Giảm ở các đặc điểm cần nhìn tổng thể cả tổn thương: pigmentation (phân bố sắc tố), blue-whitish veil, streaks.
  - Dots/globules, nhóm thầy nhắc tới, chưa tăng rõ.
- **Cách giải thích có thể** (chưa kiểm chứng): EfficientNet-B0 được pretrain ở 224. Ở 512×768, mỗi vùng mà mạng "nhìn" chỉ chiếm một phần nhỏ hơn của tổn thương, nên mạng thấy chi tiết rõ hơn nhưng khó nắm bố cục toàn cục.
- **Hạn chế:** chỉ 1 seed. Ở 224, đổi seed hoặc đổi máy cũng làm từng concept dao động 3–6 điểm. Cần đủ 3 seed trước khi kết luận.

## Hướng tiếp theo (sau pilot)

1. Theo quy tắc đã chốt: chạy đủ 3 seed và thêm Soft CBM, ECBM ở 512×768. Nhánh E_class của ECBM giống Black-box nên có thể hưởng lợi.
2. Nếu xu hướng "chi tiết tăng, toàn cục giảm" vẫn giữ qua 3 seed: thử kết hợp hai độ phân giải (một nhánh 224 nhìn toàn cục, một nhánh 512×768 nhìn chi tiết), hoặc dùng backbone được pretrain ở độ phân giải lớn hơn (EfficientNet-B4/B5).

---

# Chạy đủ: 4 mô hình × 3 seed ở 512×768 (validation)

**Cách chạy.** Kaggle T4, cùng commit, `--skip_test`; 10 lần chạy mới, cộng 2 lần chạy seed 42 của pilot. Thời gian khoảng 21 phút cho mỗi lần chạy Black-box, Hard CBM và Soft CBM; ECBM khoảng 34 phút.

- **Mốc so sánh chính:** 224 train lại trên **cùng máy Kaggle T4**, cùng 3 seed.
- Số 224 trên máy cá nhân (MPS) ở trong [summary_3seeds.json](summary_3seeds.json).
- Bước này không có tiêu chí chốt trước, vì quy tắc của pilot chỉ quyết định có chạy tiếp hay không. Phần dưới vì vậy chỉ mô tả.
- **Không chấm test.** Số ghi dạng trung bình ± độ lệch chuẩn qua 3 seed.

## Kết quả

| Mô hình | Ảnh | Accuracy | BAcc | Macro F1 | ROC-AUC | Concept acc. |
|---|---|---:|---:|---:|---:|---:|
| Black-box | 224 | 80.3 ± 1.3% | 73.1 ± 0.4% | 0.747 ± 0.009 | 0.796 ± 0.010 | — |
| Black-box | 512×768 | 81.3 ± 0.0% | **77.4 ± 1.2%** | **0.776 ± 0.005** | **0.837 ± 0.006** | — |
| Hard CBM | 224 | 84.9 ± 2.3% | 80.9 ± 3.2% | 0.816 ± 0.030 | 0.825 ± 0.010 | 65.9 ± 1.3% |
| Hard CBM | 512×768 | 82.9 ± 0.6% | 78.1 ± 0.9% | 0.790 ± 0.008 | 0.830 ± 0.012 | 65.7 ± 0.5% |
| Soft CBM | 224 | 85.6 ± 2.5% | 80.8 ± 2.9% | 0.820 ± 0.028 | 0.838 ± 0.015 | 63.7 ± 2.0% |
| Soft CBM | 512×768 | 82.9 ± 1.9% | 80.3 ± 1.6% | 0.799 ± 0.008 | 0.846 ± 0.023 | 63.1 ± 1.1% |
| ECBM | 224 | 84.2 ± 0.0% | 77.4 ± 0.3% | 0.795 ± 0.001 | 0.840 ± 0.007 | 66.1 ± 1.9% |
| ECBM | 512×768 | 82.1 ± 2.1% | **81.3 ± 1.1%** | 0.797 ± 0.017 | **0.859 ± 0.020** | 61.6 ± 2.7% |

**Độ chính xác từng concept (validation, trung bình 3 seed; 224 → 512×768):**

| Concept | Hard CBM | Soft CBM | ECBM | Chiều thay đổi |
|---|---:|---:|---:|---|
| Pigment network | 60.1 → **67.5** | 60.8 → **66.5** | 59.9 → **66.2** | tăng ở cả 3 mô hình |
| Streaks | 71.3 → 66.0 | 65.2 → 60.6 | 75.7 → 60.6 | giảm ở cả 3 |
| Blue-whitish veil | 83.4 → 81.3 | 82.1 → 80.3 | 81.6 → 77.3 | giảm ở cả 3 |
| Pigmentation | 60.9 → 50.9 | 51.9 → 51.6 | 57.0 → 45.3 | giảm ở 2/3 |
| Regression structures | 62.4 → 67.8 | 62.9 → 61.6 | 65.4 → 59.6 | không nhất quán |
| Dots and globules | 55.8 → 59.1 | 57.1 → 58.3 | 59.8 → 55.7 | không nhất quán |
| Vascular structures | 67.5 → 67.3 | 65.8 → 63.1 | 63.1 → 66.3 | không nhất quán |

## Nhận xét

- **Ảnh lớn giúp nhánh ảnh → chẩn đoán, và kết quả ổn định qua seed.**
  - Black-box: ROC-AUC 0.796 → 0.837, Macro F1 0.747 → 0.776. Cả 3 seed ở 512×768 đều cao hơn mọi lần chạy 224, trên cả hai máy.
  - ECBM: ROC-AUC 0.840 → 0.859, BAcc 77.4% → 81.3%. Chẩn đoán tự động của ECBM gần như là nhánh E_class, tức là một nhánh ảnh.
- **Concept accuracy trung bình không tăng.** Xu hướng của pilot vẫn đúng qua 3 seed và 3 mô hình:
  - Pigment network tăng 6–7 điểm ở mọi mô hình.
  - Streaks và blue-whitish veil giảm ở mọi mô hình; pigmentation giảm ở 2/3.
- **Hard CBM và Soft CBM không hưởng lợi.** Chẩn đoán của hai mô hình này đi qua concept, mà concept không tốt hơn.
  - Macro F1 giảm khoảng 0.02–0.03, nhưng ở 224 chỉ riêng việc đổi máy (MPS ↔ T4) đã làm F1 lệch tới 0.02. Vì vậy chưa thể nói 512×768 làm hai mô hình này kém đi.
- **Concept accuracy của ECBM giảm (66.1% → 61.6%) phần lớn do cách chọn checkpoint.**
  - ECBM chọn checkpoint theo F1 chẩn đoán trên validation. Ở 512×768, F1 đạt đỉnh rất sớm (seed 42 chọn epoch 5, seed 2026 chọn epoch 10), khi các head concept còn chưa học xong.
  - Ví dụ seed 42: concept F1 ở epoch 5 là 0.40, ở epoch 50 là 0.53.

## Ý nghĩa cho bước tiếp theo

1. **Không chọn được một độ phân giải tốt cho mọi thứ.** 512×768 tốt hơn cho chẩn đoán từ ảnh và cho concept chi tiết nhỏ (pigment network). 224 tốt hơn cho concept cần nhìn tổng thể (streaks, veil, pigmentation).
2. **Hướng rẻ nhất là kết hợp hai độ phân giải**, ví dụ lấy trung bình xác suất concept và chẩn đoán của bộ 224 và bộ 512×768.
   - Không cần train thêm vì đã có đủ checkpoint và dự đoán validation của cả hai bộ.
   - Đây cũng là bước rẻ nhất được đề xuất trong báo cáo nghiên cứu data-centric / model-centric.
3. **Gộp concept (D4a) có thể train ở 224**, vì ảnh lớn không làm concept tốt hơn trung bình mà tốn gấp khoảng 4 lần thời gian. Nếu bước 2 cho thấy kết hợp có ích thì train D4a ở cả hai độ phân giải.
4. **Nếu tiếp tục dùng ECBM ở 512×768**, cần chốt trước một quy tắc chọn checkpoint có tính tới concept, ví dụ chỉ xét các epoch đã qua một mốc concept F1. Nếu không, so sánh concept giữa hai độ phân giải sẽ bị lệch.

Số liệu đầy đủ (từng seed, từng concept, cả số 224 trên máy cá nhân): [summary_3seeds.json](summary_3seeds.json).

Kết quả từng lần chạy: `results/derm7pt/efficientnet_b0/linear_e50_headlr0.001_r512x768/seed42.json`, `results/derm7pt/hard_cbm/efficientnet_b0_mlp128_e50_headlr0.001_r512x768/seed42.json`. Trường `checkpoint_path` trong JSON là đường dẫn lúc chạy trên Kaggle.
