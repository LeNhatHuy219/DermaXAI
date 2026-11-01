# Điều phối đánh giá can thiệp concept đa seed cho Hard CBM qua 128 tổ hợp nhóm.


import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hard_cbm_common import (HARD_CBM_VARIANTS, DIAGNOSIS_HEADS, DEFAULT_EPOCHS,
    DEFAULT_LR, DEFAULT_WEIGHT_DECAY, DEFAULT_DIAGNOSIS_HEAD, DEFAULT_DIAGNOSIS_LR, _default_paths,
    _protect_inputs, run_hard_cbm_intervention)
from experiments.run_soft_cbm_interventions import export_report, summarize_interventions
from src.selection import add_selection_argument, add_threshold_argument
import math


# Đọc CLI/cohort, kiểm tra paths, gọi intervention từng seed và xuất báo cáo chung.
def run_cli(gradient_mode, argv=None):
    if gradient_mode not in HARD_CBM_VARIANTS:
        raise ValueError("Unknown Hard CBM gradient mode")
    parser = argparse.ArgumentParser(description=f"Hard CBM {gradient_mode}: frozen intervention reports across seeds.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--manifest_path", default=str(ROOT / "data/manifest.csv"))
    parser.add_argument("--output_dir", help="Default: results/<metric>/<model>/<configuration>/intervention")
    parser.add_argument("--checkpoints", nargs="+", help="Checkpoint paths in the same order as --seeds")
    parser.add_argument("--predictions", nargs="+", help="Test exports in the same order as --seeds")
    parser.add_argument("--augmentation_preset", choices=["legacy_letterbox", "comparison"], default="legacy_letterbox")
    parser.add_argument("--concept_loss_weight", type=float, default=1.)
    parser.add_argument("--diagnosis_head", choices=list(DIAGNOSIS_HEADS), default=DEFAULT_DIAGNOSIS_HEAD)
    parser.add_argument("--diagnosis_lr", type=float, default=DEFAULT_DIAGNOSIS_LR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--overwrite", action="store_true")
    add_selection_argument(parser)
    add_threshold_argument(parser)
    args = parser.parse_args(argv)
    if len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use at least three distinct seeds")
    if not math.isfinite(args.concept_loss_weight) or args.concept_loss_weight < 0:
        parser.error("--concept_loss_weight must be finite and non-negative")
    for values in [args.checkpoints, args.predictions]:
        if values is not None and len(values) != len(args.seeds):
            parser.error("Provide one checkpoint/prediction path per seed")
    # Suy ra cohort từ head/LR/epochs/metric; không lấy nhầm baseline Linear/BAcc.
    defaults = [_default_paths(seed, args.augmentation_preset, args.concept_loss_weight, gradient_mode,
                              checkpoint_metric=args.checkpoint_metric,
                              diagnosis_head=args.diagnosis_head, diagnosis_lr=args.diagnosis_lr,
                              epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                              threshold_metric=args.threshold_metric)
                for seed in args.seeds]
    checkpoints = args.checkpoints or [pair[0] for pair in defaults]
    predictions = args.predictions or [pair[1] for pair in defaults]
    label = "hard_cbm"
    # Lấy thư mục cấu hình từ JSON train/test để tránh lệch naming giữa các runners.
    output = Path(args.output_dir or Path(defaults[0][1]).parent / "intervention").resolve()
    exports = [output / f"seed{seed}.json" for seed in args.seeds]
    artifacts = exports + [output / name for name in (
        "summary.json", "curve.csv", "per_concept.csv", "full_intervention_cases.csv",
        "intervention_curve.png", "intervention_curve.svg", "report.md")]
    _protect_inputs(artifacts, [*checkpoints, *predictions, args.manifest_path,
                               Path(args.manifest_path).resolve().parent / "label_mapping.json"])
    if not args.overwrite and any(path.exists() for path in artifacts):
        raise FileExistsError("Hard CBM intervention artifacts already exist; choose another --output_dir or use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    # Một JSON chi tiết cho mỗi seed; hàm con kiểm tra checkpoint/export đúng cặp.
    runs = []
    for seed, checkpoint, prediction, path in zip(args.seeds, checkpoints, predictions, exports):
        run = run_hard_cbm_intervention(checkpoint, prediction, args.manifest_path, str(path),
                                 args.overwrite, expected_seed=seed, expected_gradient_mode=gradient_mode)
        runs.append(run)
    # Gộp đường intervention giữa seeds; vẫn giữ SD giữa subsets và seeds riêng.
    summary = summarize_interventions(runs)
    summary["source_paths"] = [str(path) for path in exports]
    export_report(summary, runs, output, model_label=label)
    print(f"Saved {label} {len(runs)}-seed tables and figures to {output}")
    return summary
