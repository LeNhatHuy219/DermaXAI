# Checkpoints

Weights không đưa lên Git; Git chỉ lưu README và `manifest.json`. Mã hash SHA-256 của checkpoint dùng cho bảng kết quả chính được khóa trong [`results/derm7pt/test_plan.json`](../results/derm7pt/test_plan.json).

| Thư mục | Nội dung |
|---|---|
| `derm7pt/<model>/<cấu hình>/seed<seed>.pth` | Checkpoint quy trình chung (từ 29/10/2026), dùng cho kết quả chính |
| `derm7pt/ecbm/align_pilot/` | Fold model và model đủ train của pilot đồng nhất train–inference (chỉ validation) |
| [`archive/`](archive/README.md) | Checkpoint bộ thí nghiệm cũ M0–M4, chọn theo BAcc (`bacc/`) hoặc Macro F1 (`f1_macro/`) |
| `local/` | Checkpoint chạy thử cục bộ |
