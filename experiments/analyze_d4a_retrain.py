# Phân tích theo results/derm7pt/d4a_retrain/plan.json: mô hình train lại với schema D4a so với mô hình gốc
# được gộp dự đoán sau khi train; kèm trần của dataset (chỉ train/validation). Không đọc test.


import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import src.dataset trước sklearn để tránh xung đột thư viện OpenMP trên một số nền tảng.
from src.dataset import CONCEPT_NAMES, compute_train_observed_states
from src import runtime
from src.metrics import compute_concept_metrics

import numpy as np
import pandas as pd

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "derm7pt", "d4a_retrain")
MERGE_PLAN = os.path.join(PROJECT_ROOT, "results", "derm7pt", "concept_merge_check", "plan.json")
COHORTS = {
    "hard_cbm": "results/derm7pt/hard_cbm/efficientnet_b0_mlp128_e50_headlr0.001",
    "soft_cbm": "results/derm7pt/soft_cbm/efficientnet_b0_mlp128_e50_headlr0.001",
    "ecbm": "results/derm7pt/ecbm/efficientnet_b0_unified_exact_e50",
}
SEEDS = (42, 123, 2026)
DIAGNOSIS = ("accuracy", "balanced_accuracy", "f1_macro", "roc_auc")


def load_json(*parts):
    return runtime.load_json(os.path.join(PROJECT_ROOT, *parts))


# Bảng chỉ số trạng thái gốc -> chỉ số D4a cho từng nhóm (nhóm không gộp: ánh xạ đồng nhất).
def collapse_tables(original_mapping, d4a_mapping, merge):
    tables = {}
    for concept in CONCEPT_NAMES:
        rule = merge.get(concept, {name: name for name in original_mapping[concept]})
        table = np.zeros(len(original_mapping[concept]), dtype=int)
        for name, index in original_mapping[concept].items():
            table[index] = d4a_mapping[concept][rule[name]]
        tables[concept] = table
    return tables


def concept_scores(truth, probabilities, counts, observed):
    pred = {c: probabilities[c].argmax(1) for c in CONCEPT_NAMES}
    metrics = compute_concept_metrics(truth, pred, counts, train_observed_states=observed)
    return {"concept_accuracy": metrics["overall_concept_accuracy"],
            "concept_f1_macro": metrics["overall_f1_macro_train_observed"],
            "per_concept_accuracy": {c: metrics["per_concept"][c]["accuracy"] for c in CONCEPT_NAMES},
            "per_concept_f1_macro": {c: metrics["per_concept"][c]["f1_macro_train_observed"] for c in CONCEPT_NAMES}}


def baseline_run(path, tables, counts, observed, d4a_index):
    result = load_json(path)
    records = result["validation_predictions"]
    truth, probs = {}, {}
    for concept in CONCEPT_NAMES:
        truth[concept] = np.array([tables[concept][r["concept_true"][concept]] for r in records])
        # Gộp sau khi train: cộng xác suất các trạng thái cùng nhóm rồi argmax.
        probs[concept] = np.stack([np.bincount(tables[concept], weights=np.asarray(r["concept_probabilities"][concept]),
                                               minlength=counts[concept]) for r in records])
    expected = np.array([d4a_index[r["case_num"]] for r in records])
    if not all(np.array_equal(truth[c], expected[:, g]) for g, c in enumerate(CONCEPT_NAMES)):
        raise ValueError("Collapsed ground truth differs from data/d4a/manifest.csv")
    return {**{k: result["validation_metrics"][k] for k in DIAGNOSIS},
            **concept_scores(truth, probs, counts, observed)}


def d4a_run(path, counts, observed):
    result = load_json(path)
    if result["concept_schema"]["total_states"] != 19 or "test_metrics" in result:
        raise ValueError(f"{path} is not a validation-only D4a run")
    records = result["validation_predictions"]
    truth = {c: np.array([r["concept_true"][c] for r in records]) for c in CONCEPT_NAMES}
    probs = {c: np.array([r["concept_probabilities"][c] for r in records]) for c in CONCEPT_NAMES}
    return {**{k: result["validation_metrics"][k] for k in DIAGNOSIS},
            **concept_scores(truth, probs, counts, observed), "best_epoch": result.get("best_epoch")}


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "per_seed": values.tolist()}


def decide(base, new):
    m = lambda arm, key: float(np.mean([run[key] for run in arm]))
    g1 = m(new, "roc_auc") >= m(base, "roc_auc") - 0.01 and m(new, "f1_macro") >= m(base, "f1_macro") - 0.02
    g2 = (m(new, "concept_accuracy") >= m(base, "concept_accuracy") - 0.01 and
          m(new, "concept_f1_macro") >= m(base, "concept_f1_macro") - 0.01)
    improved = {}
    for key in ("concept_accuracy", "concept_f1_macro"):
        wins = sum(n[key] > b[key] for n, b in zip(new, base))
        improved[key] = bool(m(new, key) >= m(base, key) + 0.01 and wins >= 2)
    return {"G1_diagnosis": bool(g1), "G2_concept_not_worse": bool(g2), "I_improvement": improved,
            "I_any": any(improved.values())}


# Trần của dataset: tỉ lệ ảnh có hồ sơ mâu thuẫn và accuracy tối đa nếu chỉ nhìn hồ sơ concept cứng (in-sample).
def dataset_ceiling(manifest_path, splits):
    frame = pd.read_csv(manifest_path)
    frame = frame[frame["split"].isin(splits)]
    profile = frame[CONCEPT_NAMES].astype(str).agg(" | ".join, axis=1)
    groups = frame.groupby(profile)["diagnosis_binary"]
    inconsistent = groups.transform("nunique") > 1
    majority = groups.agg(lambda y: max((y == 0).sum(), (y == 1).sum())).sum()
    melanoma = frame["diagnosis_binary"] == 1
    return {"n": int(len(frame)), "unique_profiles": int(profile.nunique()),
            "inconsistent_image_fraction": float(inconsistent.mean()),
            "inconsistent_melanoma_fraction": float(inconsistent[melanoma].mean()),
            "accuracy_ceiling_hard_concepts": float(majority / len(frame))}


def main():
    argparse.ArgumentParser().parse_args()
    merge = {k: v for k, v in load_json(MERGE_PLAN)["merge_D4a"].items() if isinstance(v, dict)}
    original_mapping = load_json("data", "label_mapping.json")
    d4a_mapping = load_json("data", "d4a", "label_mapping.json")
    counts = {c: len(d4a_mapping[c]) for c in CONCEPT_NAMES}
    tables = collapse_tables(original_mapping, d4a_mapping, merge)
    d4a_manifest = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "d4a", "manifest.csv"))
    observed = compute_train_observed_states(d4a_manifest[d4a_manifest["split"] == "train"], d4a_mapping)
    d4a_index = {int(row["case_num"]): [d4a_mapping[c][row[c]] for c in CONCEPT_NAMES]
                 for _, row in d4a_manifest[d4a_manifest["split"] == "valid"].iterrows()}

    summary = {"plan": "plan.json", "test_read": False, "models": {}}
    for model, folder in COHORTS.items():
        base = [baseline_run(f"{folder}/seed{s}.json", tables, counts, observed, d4a_index) for s in SEEDS]
        new = [d4a_run(f"{folder}_d4a/seed{s}.json", counts, observed) for s in SEEDS]
        keys = DIAGNOSIS + ("concept_accuracy", "concept_f1_macro")
        block = {arm: {k: mean_sd([r[k] for r in runs]) for k in keys}
                 for arm, runs in (("baseline_merged_after_training", base), ("d4a_retrained", new))}
        for arm, runs in (("baseline_merged_after_training", base), ("d4a_retrained", new)):
            for key in ("per_concept_accuracy", "per_concept_f1_macro"):
                block[arm][key] = {c: mean_sd([r[key][c] for r in runs]) for c in CONCEPT_NAMES}
        block["d4a_retrained"]["best_epoch"] = [r["best_epoch"] for r in new]
        block["d4a_retrained"]["devices"] = sorted({load_json(f"{folder}_d4a/seed{s}.json")["hyperparameters"]["device"]
                                                    for s in SEEDS})
        block["decision"] = decide(base, new)
        summary["models"][model] = block
    adopt = all(b["decision"]["G1_diagnosis"] and b["decision"]["G2_concept_not_worse"]
                for b in summary["models"].values())
    summary["decision"] = ("Dùng schema D4a cho các bước data-centric tiếp theo" if adopt else
                           "Giữ schema gốc làm chính; D4a là kết quả âm/trung tính")
    # Mốc "luôn đoán trạng thái phổ biến nhất của train" trên validation, để đọc accuracy từng nhóm cho đúng.
    valid = d4a_manifest[d4a_manifest["split"] == "valid"]
    train = d4a_manifest[d4a_manifest["split"] == "train"]
    summary["majority_state_accuracy_valid"] = {
        c: float((valid[c] == train[c].mode()[0]).mean()) for c in CONCEPT_NAMES}
    summary["dataset_ceiling"] = {
        schema: {name: dataset_ceiling(os.path.join(PROJECT_ROOT, *path), splits)
                 for name, splits in (("train", ["train"]), ("valid", ["valid"]), ("train+valid", ["train", "valid"]))}
        for schema, path in (("original", ("data", "manifest.csv")), ("d4a", ("data", "d4a", "manifest.csv")))}
    runtime.save_json(os.path.join(OUT_DIR, "summary.json"), summary)

    for model, block in summary["models"].items():
        for arm in ("baseline_merged_after_training", "d4a_retrained"):
            row = block[arm]
            print(f"{model:9s} {arm:31s} " + "  ".join(
                f"{k}={row[k]['mean']:.3f}±{row[k]['sd']:.3f}" for k in DIAGNOSIS + ("concept_accuracy", "concept_f1_macro")))
        print(f"{model:9s} decision {json.dumps(block['decision'])}")
    print(summary["decision"])
    print(json.dumps(summary["dataset_ceiling"], indent=1))


if __name__ == "__main__":
    main()
