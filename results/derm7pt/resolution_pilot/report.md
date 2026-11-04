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

## Hướng tiếp theo

1. Theo quy tắc đã chốt: chạy đủ 3 seed và thêm Soft CBM, ECBM ở 512×768. Nhánh E_class của ECBM giống Black-box nên có thể hưởng lợi.
2. Nếu xu hướng "chi tiết tăng, toàn cục giảm" vẫn giữ qua 3 seed: thử kết hợp hai độ phân giải (một nhánh 224 nhìn toàn cục, một nhánh 512×768 nhìn chi tiết), hoặc dùng backbone được pretrain ở độ phân giải lớn hơn (EfficientNet-B4/B5).

Kết quả từng lần chạy: `results/derm7pt/efficientnet_b0/linear_e50_headlr0.001_r512x768/seed42.json`, `results/derm7pt/hard_cbm/efficientnet_b0_mlp128_e50_headlr0.001_r512x768/seed42.json`. Trường `checkpoint_path` trong JSON là đường dẫn lúc chạy trên Kaggle.
