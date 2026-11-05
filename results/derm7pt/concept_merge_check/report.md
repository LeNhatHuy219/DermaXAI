# Kiểm tra nhanh: gộp trạng thái concept hiếm (D4a)

**Câu hỏi.** Hai câu hỏi cần trả lời trước khi quyết định train lại:

1. Gộp các trạng thái concept hiếm có làm mất thông tin để chẩn đoán qua concept không?
2. Bao nhiêu lỗi concept hiện tại chỉ là nhầm giữa các trạng thái trong cùng một nhóm gộp?

**Cách gộp (D4a)** — 3 nhóm thay đổi:

- pigmentation: 5 → 3 trạng thái (không có / đều / không đều);
- regression structures: 4 → 2 (không có / có);
- vascular structures: 8 → 3 (không có / điển hình / không điển hình).

**Cách làm.** Không train backbone, không đọc test. Tiêu chí ở [plan.json](plan.json), ghi trước khi chạy.

- Dùng dự đoán validation đã lưu của Hard CBM, Soft CBM và ECBM (ảnh 224, 3 seed).
- Gộp xác suất concept theo từng nhóm gộp.
- Chẩn đoán qua concept bằng logistic regression fit trên concept thật (GT) của 413 ca train.

## Kết quả (validation)

**Chẩn đoán qua concept:**

| | Schema gốc | D4a |
|---|---:|---:|
| ROC-AUC với concept đúng (GT) | 0.925 | 0.929 |
| ROC-AUC với concept dự đoán (trung bình 9 mô hình × seed) | 0.818 | 0.816 |
| Macro F1 @0.5 với concept dự đoán | 0.734 | 0.716 |

**Concept accuracy ở 3 nhóm bị gộp:**

| Nhóm | Gốc | Sau khi gộp dự đoán |
|---|---:|---:|
| Pigmentation | 56.8% | 57.3% |
| Regression structures | 63.2% | 62.9% |
| Vascular structures | 63.6% | 60.6% |

Ở 3 nhóm này có 2128 lỗi concept; **chỉ 20% biến mất khi gộp**.

## Kết luận

- **R1 đạt.** Gộp không làm mất thông tin chẩn đoán: với concept đúng, AUC còn nhích lên (0.925 → 0.929); với concept dự đoán, AUC gần như giữ nguyên (−0.002). Theo quy tắc đã chốt, D4a **đáng train lại**.
- **Kỳ vọng phải thấp.** 80% lỗi concept là nhầm **có/không** (ví dụ thấy cấu trúc mạch máu khi thực ra không có), không phải nhầm giữa các trạng thái hiếm. Gộp nhãn không sửa được loại lỗi này. Nó phụ thuộc vào việc mô hình nhìn ảnh có tốt không (độ phân giải, backbone) và vào nhiễu nhãn.
- **Lợi ích chắc chắn khi train lại theo D4a** nằm ở chỗ khác:
  - Bỏ được trọng số cực lớn cho các trạng thái chỉ có 0–5 ca train.
  - ECBM chỉ cần duyệt 972 tổ hợp thay vì 8.640.
  - Phần giải thích khớp với cách chấm 7-point checklist.
- **Macro F1 @0.5 giảm (0.734 → 0.716)** là do dùng ngưỡng 0.5 cố định. AUC, chỉ số không phụ thuộc ngưỡng, gần như không đổi.

Số liệu chi tiết: [summary.json](summary.json). Script: `experiments/check_concept_merge.py`.
