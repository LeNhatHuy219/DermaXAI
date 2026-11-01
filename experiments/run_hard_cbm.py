# Runner Hard CBM: Hard Joint CBM với cơ chế Straight-Through estimator.


from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import hard_cbm_common as common
from experiments.hard_cbm_common import (DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LR,
    DEFAULT_WEIGHT_DECAY, DEFAULT_SEED, DEFAULT_AUGMENTATION_PRESET,
    DEFAULT_CONCEPT_LOSS_WEIGHT, _protect_inputs, evaluate)

HARD_CBM_MODEL = common.HARD_CBM_MODEL
HARD_CBM_PROTOCOL = common.HARD_CBM_PROTOCOL


# Tạo checkpoint/JSON paths theo cấu hình, seed và tiêu chí chọn checkpoint.
def _default_paths(seed=DEFAULT_SEED, augmentation_preset=DEFAULT_AUGMENTATION_PRESET,
                   concept_loss_weight=DEFAULT_CONCEPT_LOSS_WEIGHT,
                   checkpoint_metric=common.DEFAULT_CHECKPOINT_METRIC,
                   diagnosis_head=common.DEFAULT_DIAGNOSIS_HEAD, diagnosis_lr=common.DEFAULT_DIAGNOSIS_LR,
                   epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY,
                   threshold_metric=common.DEFAULT_THRESHOLD_METRIC):
    return common._default_paths(seed, augmentation_preset, concept_loss_weight, "straight_through",
                                 checkpoint_metric=checkpoint_metric, diagnosis_head=diagnosis_head,
                                 diagnosis_lr=diagnosis_lr, epochs=epochs, lr=lr,
                                 weight_decay=weight_decay, threshold_metric=threshold_metric)


# Kiểm tra config Hard CBM và yêu cầu variant ST trước khi dùng checkpoint.
def validate_hard_cbm_config(config):
    criteria = common.validate_hard_cbm_config(config)
    common._require_gradient_mode(config, "straight_through")
    return criteria


# Gọi train/validation chung nhưng cố định gradient_mode straight_through.
def run_hard_cbm_experiment(*args, **kwargs):
    return common.run_hard_cbm_experiment(*args, **kwargs, gradient_mode="straight_through")


# Gọi frozen test chung, yêu cầu checkpoint ST; không chọn lại epoch/ngưỡng.
def evaluate_hard_cbm_test(*args, **kwargs):
    return common.evaluate_hard_cbm_test(*args, **kwargs, expected_gradient_mode="straight_through")


# Kiểm tra export đúng ST rồi đối chiếu hard concepts và GT với manifest.
def validate_hard_cbm_test_export(export, manifest_path):
    validate_hard_cbm_config(export.get("hyperparameters"))
    return common.validate_hard_cbm_test_export(export, manifest_path)


# Gọi frozen intervention chung, yêu cầu checkpoint/export cùng variant ST.
def run_hard_cbm_intervention(*args, **kwargs):
    return common.run_hard_cbm_intervention(*args, **kwargs, expected_gradient_mode="straight_through")


# Chuyển tham số CLI sang hard_cbm_common.run_cli cho ST; logic train/test nằm ở file dùng chung.
def main(argv=None):
    return common.run_cli("straight_through", argv)


if __name__ == "__main__":
    main()
