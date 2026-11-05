# Kiểm tra rẻ cho hướng gộp trạng thái concept hiếm (D4a), theo results/derm7pt/concept_merge_check/plan.json.
# Dùng dự đoán validation đã lưu của mô hình 224 và concept GT của train; không train backbone, không đọc test.


import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import src.dataset trước sklearn để tránh xung đột thư viện OpenMP trên một số nền tảng.
from src.dataset import CONCEPT_NAMES
from src import runtime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "derm7pt", "concept_merge_check")
COHORTS = {
    "Hard CBM": "results/derm7pt/hard_cbm/efficientnet_b0_mlp128_e50_headlr0.001",
    "Soft CBM": "results/derm7pt/soft_cbm/efficientnet_b0_mlp128_e50_headlr0.001",
    "ECBM": "results/derm7pt/ecbm/efficientnet_b0_unified_exact_e50",
}
SEEDS = (42, 123, 2026)


# Bảng ánh xạ chỉ số trạng thái gốc -> chỉ số trạng thái gộp cho từng nhóm (nhóm không gộp: ánh xạ đồng nhất).
def merge_tables(mapping, merge):
    tables, sizes = {}, {}
    for concept in CONCEPT_NAMES:
        states = mapping[concept]  # tên trạng thái -> chỉ số gốc
        target = merge.get(concept, {name: name for name in states})
        new_names = sorted(set(target.values()))
        tables[concept] = np.array([new_names.index(target[name]) for name, _ in sorted(states.items(), key=lambda x: x[1])])
        sizes[concept] = len(new_names)
    return tables, sizes


def one_hot(indices, sizes):
    return np.hstack([np.eye(sizes[c])[indices[:, g]] for g, c in enumerate(CONCEPT_NAMES)])


def fit_head(train_concepts, train_labels, sizes):
    model = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)
    return model.fit(one_hot(train_concepts, sizes), train_labels)


def diagnose(model, concepts, sizes, labels):
    prob = model.predict_proba(one_hot(concepts, sizes))[:, 1]
    return {"roc_auc": float(roc_auc_score(labels, prob)),
            "f1_macro_at_0.5": float(f1_score(labels, (prob >= 0.5).astype(int), average="macro"))}


def main():
    plan = runtime.load_json(os.path.join(OUT_DIR, "plan.json"))
    merge = {k: v for k, v in plan["merge_D4a"].items() if isinstance(v, dict)}
    mapping = runtime.load_json(os.path.join(PROJECT_ROOT, "data", "label_mapping.json"))
    tables, merged_sizes = merge_tables(mapping, merge)
    original_sizes = {c: len(mapping[c]) for c in CONCEPT_NAMES}
    collapse = lambda idx: np.stack([tables[c][idx[:, g]] for g, c in enumerate(CONCEPT_NAMES)], axis=1)

    manifest = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "manifest.csv"))
    train = manifest[manifest.split == "train"]
    train_idx = np.stack([train[c].map(mapping[c]).to_numpy() for c in CONCEPT_NAMES], axis=1)
    train_y = train.diagnosis_binary.to_numpy()
    heads = {"original": fit_head(train_idx, train_y, original_sizes),
             "D4a": fit_head(collapse(train_idx), train_y, merged_sizes)}
    sizes = {"original": original_sizes, "D4a": merged_sizes}

    merged_groups = list(merge)
    runs, oracle = [], None
    for model_name, folder in COHORTS.items():
        for seed in SEEDS:
            records = runtime.load_json(os.path.join(PROJECT_ROOT, folder, f"seed{seed}.json"))["validation_predictions"]
            labels = np.array([r["y_true"] for r in records])
            truth = np.array([[r["concept_true"][c] for c in CONCEPT_NAMES] for r in records])
            pred = np.array([[r["concept_pred"][c] for c in CONCEPT_NAMES] for r in records])
            # Gộp bằng cách cộng xác suất trong nhóm rồi argmax (không chỉ gộp nhãn argmax).
            merged_pred = np.stack([np.array([np.bincount(tables[c], weights=np.asarray(r["concept_probabilities"][c]),
                                                          minlength=merged_sizes[c]).argmax() for r in records])
                                    for c in CONCEPT_NAMES], axis=1)
            merged_truth = collapse(truth)
            if oracle is None:
                oracle = {schema: diagnose(heads[schema], t, sizes[schema], labels)
                          for schema, t in (("original", truth), ("D4a", merged_truth))}
            per_group = {}
            for g, c in enumerate(CONCEPT_NAMES):
                wrong = pred[:, g] != truth[:, g]
                per_group[c] = {"accuracy_original": float(np.mean(~wrong)),
                                "accuracy_D4a": float(np.mean(merged_pred[:, g] == merged_truth[:, g])),
                                "errors_original": int(wrong.sum()),
                                "errors_fixed_by_merge": int(np.sum(wrong & (merged_pred[:, g] == merged_truth[:, g])))}
            runs.append({"model": model_name, "seed": seed, "per_group": per_group,
                         "concept_accuracy_original": float(np.mean(pred == truth)),
                         "concept_accuracy_D4a": float(np.mean(merged_pred == merged_truth)),
                         "diagnosis_via_predicted_concepts": {
                             "original": diagnose(heads["original"], pred, original_sizes, labels),
                             "D4a": diagnose(heads["D4a"], merged_pred, merged_sizes, labels)}})

    mean = lambda values: float(np.mean(values))
    auc = {schema: mean([r["diagnosis_via_predicted_concepts"][schema]["roc_auc"] for r in runs]) for schema in sizes}
    f1 = {schema: mean([r["diagnosis_via_predicted_concepts"][schema]["f1_macro_at_0.5"] for r in runs]) for schema in sizes}
    errors = sum(r["per_group"][c]["errors_original"] for r in runs for c in merged_groups)
    fixed = sum(r["per_group"][c]["errors_fixed_by_merge"] for r in runs for c in merged_groups)
    r1 = (oracle["D4a"]["roc_auc"] >= oracle["original"]["roc_auc"] - 0.01) and (auc["D4a"] >= auc["original"] - 0.01)
    summary = {
        "test_read": False,
        "oracle_gt_concepts": oracle,
        "predicted_concepts_mean_9_runs": {"roc_auc": auc, "f1_macro_at_0.5": f1},
        "by_model": {m: {schema: {"roc_auc": mean([r["diagnosis_via_predicted_concepts"][schema]["roc_auc"] for r in runs if r["model"] == m]),
                                  "f1_macro_at_0.5": mean([r["diagnosis_via_predicted_concepts"][schema]["f1_macro_at_0.5"] for r in runs if r["model"] == m])}
                         for schema in sizes} for m in COHORTS},
        "merged_group_accuracy": {c: {k: mean([r["per_group"][c][k] for r in runs]) for k in ("accuracy_original", "accuracy_D4a")}
                                  for c in merged_groups},
        "concept_accuracy_mean": {k: mean([r[f"concept_accuracy_{k}"] for r in runs]) for k in ("original", "D4a")},
        "R2_errors_in_merged_groups": {"total": errors, "fixed_by_merge": fixed, "fraction": fixed / errors},
        "R1_information_kept": bool(r1),
        "decision": "Đáng train lại theo D4a" if r1 else "Không train lại: gộp làm mất thông tin chẩn đoán qua concept",
        "runs": runs,
    }
    runtime.save_json(os.path.join(OUT_DIR, "summary.json"), summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
