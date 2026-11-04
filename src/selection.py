# Tiêu chí chọn checkpoint tốt nhất trên tập validation theo metric chỉ định.
import math
from pathlib import Path

from src.naming import DATASET
from src.transforms import image_size_hw

# Metric mặc định để chọn checkpoint (Macro-F1 hoặc Balanced Accuracy).

DEFAULT_CHECKPOINT_METRIC = "f1_macro"
CHECKPOINT_METRICS = ("f1_macro", "balanced_accuracy")


def selection_name(metric):
    # Kiểm tra tên metric trước khi tạo chuỗi mô tả dùng trong metadata checkpoint.
    if metric not in CHECKPOINT_METRICS:
        raise ValueError("checkpoint_metric must be f1_macro or balanced_accuracy")
    # @0.5 là ngưỡng tính metric lúc chọn epoch trên validation.
    # Ngưỡng dùng cho test/intervention được chọn riêng trên validation sau đó.
    return f"validation_{metric}_at_0.5"


def selection_score(metrics, metric):
    # Chọn theo diagnosis Macro F1, không nhầm với F1 melanoma/concept F1.
    # Xác nhận metric hợp lệ, rồi lấy điểm từ dict metrics của epoch đang xét.
    selection_name(metric)
    score = float(metrics[metric])
    # Loại bỏ NaN, vô cực và điểm ngoài [0, 1] để tránh lưu checkpoint sai.
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Checkpoint selection score must be finite and in [0, 1]")
    # Runner dùng điểm này để so sánh; nếu hòa thì giữ epoch tốt nhất đầu tiên.
    return score


def metric_from_config(config):
    # Artifact cũ có BAcc (hoặc thiếu metadata) vẫn được đọc theo protocol cũ.
    # Tiêu chí được đọc từ config đã lưu, thay vì lấy mặc định F1 của lần chạy mới.
    name = config.get("checkpoint_selection", selection_name("balanced_accuracy"))
    for metric in CHECKPOINT_METRICS:
        if name == selection_name(metric):
            # Nếu config có cả hai trường thì chúng phải mô tả cùng một tiêu chí.
            if config.get("checkpoint_metric", metric) != metric:
                raise ValueError("Checkpoint metric and selection metadata differ")
            return metric
    # Metadata lạ phải được kiểm tra lại, thay vì tự đoán tiêu chí chọn checkpoint.
    raise ValueError("Unknown checkpoint selection metadata")


# Tên thư mục cho một cấu hình; seed nằm trong tên JSON để ba runs ở cùng cohort.
def configuration_name(head, epochs, diagnosis_lr=None, *, augmentation_preset="legacy_letterbox",
                       concept_loss_weight=1., lr=1e-4, default_lr=1e-4, weight_decay=.01, image_size=224):
    name = f"{head}_e{epochs}"
    if diagnosis_lr is not None:
        name += f"_headlr{diagnosis_lr:g}"
    if augmentation_preset != "legacy_letterbox":
        name += f"_{augmentation_preset}"
    if concept_loss_weight != 1.:
        name += f"_lambda{concept_loss_weight:g}"
    if lr != default_lr:
        name += f"_lr{lr:g}"
    if weight_decay != .01:
        name += f"_wd{weight_decay:g}"
    # Kích thước khác 224 ghi vào tên (_r512, _r512x768) để không ghi đè kết quả 224.
    height, width = image_size_hw(image_size)
    if (height, width) != (224, 224):
        name += f"_r{height}" if height == width else f"_r{height}x{width}"
    return name


# Tiêu chí chọn ngưỡng quyết định trên validation; mặc định Macro F1, cùng tiêu chí chọn checkpoint.
THRESHOLD_METRICS = ("f1_macro", "balanced_accuracy")
DEFAULT_THRESHOLD_METRIC = "f1_macro"


def threshold_selection_name(metric):
    if metric not in THRESHOLD_METRICS:
        raise ValueError("threshold_metric must be f1_macro or balanced_accuracy")
    return f"maximize_validation_{metric}_tie_nearest_0.5"


def threshold_metric_from_config(config):
    # Config cũ chưa ghi tiêu chí ngưỡng đều đã chọn theo BAcc.
    name = config.get("threshold_selection", threshold_selection_name("balanced_accuracy"))
    for metric in THRESHOLD_METRICS:
        if name == threshold_selection_name(metric):
            return metric
    raise ValueError("Unknown threshold selection metadata")


# Thử mọi xác suất validation (cộng 0.5 và 1.0) làm ngưỡng; giữ điểm cao nhất, hòa ưu tiên gần 0.5.
def select_threshold(y_true, y_prob, metric=DEFAULT_THRESHOLD_METRIC):
    import numpy as np
    from src.metrics import compute_metrics  # nạp muộn: runner import torch trước sklearn
    threshold_selection_name(metric)
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) != len(y_prob) or len(np.unique(y_true)) != 2:
        raise ValueError("Threshold selection requires aligned validation scores from both classes")
    if not np.all(np.isfinite(y_prob)) or np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("Predicted probabilities must be finite and in [0, 1]")
    best_threshold, best_score = 0.5, -1.0
    for threshold in np.unique(np.concatenate([y_prob, [0.5, 1.0]])):
        score = compute_metrics(y_true, (y_prob >= threshold).astype(int))[metric]
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12
            and (abs(threshold - 0.5), threshold) < (abs(best_threshold - 0.5), best_threshold)
        ):
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, best_score


# Đường dẫn run mới: checkpoints|results/<dataset>/<model>/[<backbone>_]<cấu hình>[_ckptbacc][_thrbacc]/seed<seed>.
# Model black-box dùng id backbone làm <model> nên không lặp tiền tố; CBM/ECBM truyền backbone làm tiền tố.
# Kết quả cũ theo tên M1–M5 (results/archive/f1_macro, checkpoints/archive/f1_macro, ...) giữ nguyên; xem results/NAME_MAPPING.md.
def run_paths(project_root, metric, model, configuration, seed, checkpoint_suffix=".pth",
              threshold_metric=DEFAULT_THRESHOLD_METRIC, backbone=None):
    selection_name(metric)
    threshold_selection_name(threshold_metric)
    folder = configuration if backbone is None else f"{backbone}_{configuration}"
    if metric != DEFAULT_CHECKPOINT_METRIC:
        folder += "_ckptbacc"
    if threshold_metric != DEFAULT_THRESHOLD_METRIC:
        folder += "_thrbacc"
    root = Path(project_root)
    return (str(root / "checkpoints" / DATASET / model / folder / f"seed{seed}{checkpoint_suffix}"),
            str(root / "results" / DATASET / model / folder / f"seed{seed}.json"))


def add_selection_argument(parser):
    # Thêm tùy chọn chung cho CLI, ví dụ: --checkpoint_metric f1_macro.
    # argparse giới hạn hai tên hợp lệ; bỏ qua tùy chọn này thì mặc định dùng F1.
    parser.add_argument("--checkpoint_metric", choices=CHECKPOINT_METRICS,
        default=DEFAULT_CHECKPOINT_METRIC,
        help="Diagnosis metric trên validation @0.5 để chọn checkpoint; hòa giữ epoch đầu")


def add_threshold_argument(parser):
    parser.add_argument("--threshold_metric", choices=THRESHOLD_METRICS, default=DEFAULT_THRESHOLD_METRIC,
        help="Metric trên validation để chọn ngưỡng quyết định; balanced_accuracy để tái lập protocol cũ")
