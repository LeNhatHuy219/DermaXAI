# Runner Oracle-LR: Phân loại chẩn đoán từ concept Ground Truth bằng Logistic Regression.


import argparse
import os
import platform
import sys
from typing import Any, Dict, List, Tuple

# Thêm thư mục gốc vào đường dẫn hệ thống
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import src.dataset trước sklearn để tránh xung đột thư viện OpenMP trên một số nền tảng.

from src.dataset import CONCEPT_NAMES, CONCEPT_OFFSETS, TOTAL_CONCEPT_STATES
from src import runtime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression

from src.metrics import compute_metrics, print_metrics_table
from src.selection import (DEFAULT_CHECKPOINT_METRIC, selection_name, selection_score,
                           metric_from_config, add_selection_argument, run_paths, DEFAULT_THRESHOLD_METRIC,
                           add_threshold_argument, select_threshold, threshold_metric_from_config,
                           threshold_selection_name)
from src.protocol import (manifest_fingerprint, load_concept_schema, prepare_training_outputs,
                          save_experiment_results,
                          validate_manifest, validate_concept_schema)


# Baseline Oracle Logistic Regression từ 7 concept Ground Truth (one-hot 28 chiều).

DEFAULT_SEED = 42
DEFAULT_C_GRID: List[float] = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


# Tạo checkpoint/JSON paths theo cấu hình, seed và tiêu chí chọn checkpoint.
def _default_paths(seed: int = DEFAULT_SEED, checkpoint_metric: str = DEFAULT_CHECKPOINT_METRIC,
                   threshold_metric: str = DEFAULT_THRESHOLD_METRIC) -> Tuple[str, str]:
    return run_paths(PROJECT_ROOT, checkpoint_metric, "oracle_lr", "logistic_regression", seed,
                     checkpoint_suffix=".joblib", threshold_metric=threshold_metric)


# Ghép bảy GT concept groups thành one-hot 28 chiều; không đưa diagnosis vào features.
def build_oracle_features(df: pd.DataFrame, label_mapping: Dict[str, Dict[str, int]]) -> np.ndarray:
    n = len(df)
    features = np.zeros((n, TOTAL_CONCEPT_STATES), dtype=np.float32)
    for row_pos, (_, row) in enumerate(df.iterrows()):
        for c_name in CONCEPT_NAMES:
            c_idx = label_mapping[c_name][row[c_name]]
            features[row_pos, CONCEPT_OFFSETS[c_name] + c_idx] = 1.0
    return features


# Lấy tên 28 concept states theo đúng thứ tự các cột của vector đầu vào.
def concept_state_names(label_mapping: Dict[str, Dict[str, int]]) -> List[str]:
    # Tên đầy đủ "concept=state" cho từng cột của vector 28 chiều, phục vụ diễn giải hệ số hồi quy
    names = [""] * TOTAL_CONCEPT_STATES
    for c_name in CONCEPT_NAMES:
        for state_name, state_idx in label_mapping[c_name].items():
            names[CONCEPT_OFFSETS[c_name] + state_idx] = f"{c_name}={state_name}"
    return names


# Fit LR cho từng C trên train; chọn bằng validation metric@0.5, hòa giữ C đầu trong grid.
def _select_best_C(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    c_grid: List[float],
    seed: int,
    checkpoint_metric: str = DEFAULT_CHECKPOINT_METRIC,
) -> Tuple[float, Dict[float, float], LogisticRegression]:
    selection_name(checkpoint_metric)
    scores: Dict[float, float] = {}
    best_c = c_grid[0]
    best_score = -1.0
    best_model = None
    for c in c_grid:
        model = LogisticRegression(
            C=c, class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=seed,
        )
        model.fit(X_train, y_train)
        valid_probs = model.predict_proba(X_valid)[:, 1]
        valid_preds = (valid_probs >= 0.5).astype(int)
        score = selection_score(compute_metrics(y_valid, valid_preds), checkpoint_metric)
        scores[c] = score
        if score > best_score + 1e-12:
            best_score = score
            best_c = c
            best_model = model
    return best_c, scores, best_model


# Chuẩn bị GT features, chọn C/ngưỡng bằng validation, lưu classifier và validation JSON.
def run_oracle_lr_experiment(
    manifest_path: str = None,
    seed: int = DEFAULT_SEED,
    c_grid: List[float] = None,
    checkpoint_path: str = None,
    results_path: str = None,
    save_results: bool = True,
    overwrite: bool = False,
    checkpoint_metric: str = DEFAULT_CHECKPOINT_METRIC,
    threshold_metric: str = DEFAULT_THRESHOLD_METRIC,
) -> Dict[str, Any]:
    threshold_selection_name(threshold_metric)
    if manifest_path is None:
        manifest_path = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
    manifest_path = os.path.abspath(manifest_path)
    if c_grid is None:
        c_grid = DEFAULT_C_GRID
    if not c_grid or any(not np.isfinite(c) or c <= 0 for c in c_grid):
        raise ValueError("c_grid must contain finite positive values")

    checkpoint_path, results_path = prepare_training_outputs(
        _default_paths(seed, checkpoint_metric, threshold_metric), checkpoint_path, results_path, overwrite, save_results,
    )

    runtime.prepare_output(checkpoint_path, overwrite)
    if save_results:
        runtime.prepare_output(results_path, overwrite)

    print("Khởi động Oracle LR (Logistic Regression trên concept Ground Truth)")

    df = pd.read_csv(manifest_path, dtype={"is_inconsistent_profile": bool})
    schema = load_concept_schema(manifest_path)
    label_mapping = schema["label_mapping"]

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    valid_df = df[df["split"] == "valid"].reset_index(drop=True)

    assert len(train_df) == 413, "Train split phải có đúng 413 mẫu"
    assert len(valid_df) == 203, "Valid split phải có đúng 203 mẫu"

    X_train = build_oracle_features(train_df, label_mapping)
    y_train = train_df["diagnosis_binary"].to_numpy(dtype=int)
    X_valid = build_oracle_features(valid_df, label_mapping)
    y_valid = valid_df["diagnosis_binary"].to_numpy(dtype=int)

    print(f"Grid search C trên validation: {c_grid}")
    best_c, c_scores, best_model = _select_best_C(X_train, y_train, X_valid, y_valid, c_grid, seed, checkpoint_metric)
    print(f"C tốt nhất theo {checkpoint_metric} (threshold=0.5) trên validation: {best_c} (score={c_scores[best_c]:.4f})")

    valid_probs = best_model.predict_proba(X_valid)[:, 1]
    valid_metrics_at_05 = compute_metrics(y_valid, (valid_probs >= 0.5).astype(int), valid_probs)

    threshold, _ = select_threshold(y_valid, valid_probs, threshold_metric)
    valid_preds = (valid_probs >= threshold).astype(int)
    valid_metrics = compute_metrics(y_valid, valid_preds, valid_probs)
    valid_table = print_metrics_table(valid_metrics, title=f"Validation Set (C={best_c}, threshold={threshold:.6f})")

    manifest_sha256 = runtime.file_sha256(manifest_path)
    config = {
        "model_type": "LogisticRegression",
        "feature_space": "concept_onehot_28d_ground_truth",
        "class_weight": "balanced",
        "solver": "lbfgs",
        "max_iter": 2000,
        "seed": seed,
        "c_grid": c_grid,
        "best_C": best_c,
        f"c_grid_validation_{checkpoint_metric}": c_scores,
        "checkpoint_selection": selection_name(checkpoint_metric),
        "threshold_selection": threshold_selection_name(threshold_metric),
        "scikit_learn_version": str(sklearn.__version__),
        "numpy_version": str(np.__version__),
        "pandas_version": str(pd.__version__),
        "platform": platform.platform(),
    }

    concept_names_28 = concept_state_names(label_mapping)
    coefficients = {
        name: float(coef) for name, coef in zip(concept_names_28, best_model.coef_.ravel())
    }

    joblib.dump({
        "model": best_model,
        "config": config,
        "manifest_sha256": manifest_sha256,
        "manifest_fingerprint": manifest_fingerprint(manifest_path),
        "concept_schema": schema,
        "decision_threshold": threshold,
        "validation_metrics_at_threshold": valid_metrics,
        "concept_state_coefficients": coefficients,
        "intercept": float(best_model.intercept_[0]),
    }, checkpoint_path)
    print(f"Checkpoint đã lưu tại: {checkpoint_path}")

    val_sample_preds = runtime.sample_prediction_records(valid_df, valid_preds, valid_probs)
    summary_text = (
        f"Oracle LR (Logistic Regression, concept Ground Truth)\n"
        f"Training samples: 413, Valid: 203\n"
        f"Best C (validation BAcc@0.5): {best_c}\n"
        f"Threshold selected on validation: {threshold:.6f}\n\n"
        f"{valid_table}"
    )

    results = {
        "model": "oracle_lr",
        "mode": "train_validation",
        "hyperparameters": config,
        "manifest_sha256": manifest_sha256,
        "manifest_fingerprint": manifest_fingerprint(manifest_path),
        "concept_schema": schema,
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "decision_threshold": threshold,
        "summary": summary_text,
        "validation_metrics_at_0.5": valid_metrics_at_05,
        "validation_metrics": valid_metrics,
        "validation_predictions": val_sample_preds,
        "concept_state_coefficients": coefficients,
        "intercept": float(best_model.intercept_[0]),
    }

    if save_results:
        runtime.save_json(results_path, results)
        print(f"Kết quả chi tiết đã lưu tại: {results_path}")

    return results


# Đọc frozen classifier/schema/ngưỡng, tạo GT features test và chấm oracle test.
def evaluate_oracle_lr_test(
    checkpoint_path: str,
    manifest_path: str = None,
    results_path: str = None,
    save_results: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    if manifest_path is None:
        manifest_path = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
    manifest_path = os.path.abspath(manifest_path)

    checkpoint = joblib.load(checkpoint_path)
    if "decision_threshold" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Checkpoint has no frozen validation threshold/config; use a newly trained checkpoint")
    validate_manifest(checkpoint, manifest_path)
    schema = validate_concept_schema(checkpoint, manifest_path)

    config = checkpoint["config"]
    metric_from_config(config)
    threshold = float(checkpoint["decision_threshold"])
    if not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold in checkpoint")

    if results_path is None:
        _, results_path = _default_paths(config["seed"], metric_from_config(config), threshold_metric_from_config(config))
    if save_results:
        runtime.prepare_output(results_path, overwrite)

    df = pd.read_csv(manifest_path, dtype={"is_inconsistent_profile": bool})
    label_mapping = schema["label_mapping"]
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    assert len(test_df) == 395, "Test split phải có đúng 395 mẫu"

    X_test = build_oracle_features(test_df, label_mapping)
    y_test = test_df["diagnosis_binary"].to_numpy(dtype=int)

    model: LogisticRegression = checkpoint["model"]
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= threshold).astype(int)
    test_metrics = compute_metrics(y_test, test_preds, test_probs)
    test_table = print_metrics_table(test_metrics, title=f"Test Set (threshold={threshold:.6f})")

    results = {
        "model": "oracle_lr",
        "mode": "final_test",
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "hyperparameters": config,
        "manifest_sha256": checkpoint["manifest_sha256"],
        "manifest_fingerprint": manifest_fingerprint(manifest_path),
        "concept_schema": schema,
        "decision_threshold": threshold,
        "summary": test_table,
        "test_metrics": test_metrics,
        "test_predictions": runtime.sample_prediction_records(test_df, test_preds, test_probs),
    }

    if save_results:
        save_experiment_results(results_path, test=results, overwrite=overwrite)
        print(f"Kết quả test đã lưu tại: {results_path}")

    return results


# Đọc CLI: mặc định fit/chọn bằng train-valid rồi test; mode test dùng checkpoint có sẵn.
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="train: train rồi tự động test; test: đánh giá checkpoint đã lưu")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--results_path", type=str, default=None, help="Một file JSON chứa kết quả validation và test")
    parser.add_argument("--overwrite", action="store_true", help="Cho phép ghi đè artifact của cùng run")
    parser.add_argument("--no_save", action="store_true", help="Không lưu kết quả vào file")
    add_selection_argument(parser)
    add_threshold_argument(parser)
    args = parser.parse_args(argv)

    if args.mode == "train":
        checkpoint_path, results_path = prepare_training_outputs(
            _default_paths(args.seed, args.checkpoint_metric, args.threshold_metric), args.checkpoint_path, args.results_path,
            args.overwrite, not args.no_save,
        )
        validation = run_oracle_lr_experiment(
            manifest_path=args.manifest_path,
            seed=args.seed,
            checkpoint_path=checkpoint_path,
            save_results=False,
            overwrite=args.overwrite,
            checkpoint_metric=args.checkpoint_metric,
            threshold_metric=args.threshold_metric,
        )
        if not args.no_save:
            save_experiment_results(results_path, validation=validation, overwrite=args.overwrite)
        test = evaluate_oracle_lr_test(
            checkpoint_path=checkpoint_path, manifest_path=args.manifest_path,
            save_results=False, overwrite=args.overwrite,
        )
        if not args.no_save:
            save_experiment_results(results_path, validation=validation, test=test, overwrite=True)
    else:
        checkpoint_path = args.checkpoint_path or _default_paths(args.seed, args.checkpoint_metric, args.threshold_metric)[0]
        results_path = args.results_path
        if results_path is None:
            config = joblib.load(checkpoint_path)["config"]
            results_path = _default_paths(config["seed"], metric_from_config(config), threshold_metric_from_config(config))[1]
        if not args.no_save:
            runtime.prepare_output(results_path, args.overwrite)
        test = evaluate_oracle_lr_test(
            checkpoint_path=checkpoint_path,
            manifest_path=args.manifest_path,
            save_results=False,
            overwrite=args.overwrite,
        )
        if not args.no_save:
            save_experiment_results(results_path, test=test, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
