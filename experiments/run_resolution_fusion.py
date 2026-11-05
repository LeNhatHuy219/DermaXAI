# Kết hợp mô hình 224 với mô hình 512x768 theo results/derm7pt/resolution_fusion/plan.json.
# Chỉ dùng dự đoán validation đã lưu; không train, không chạy mô hình, không đọc test.


import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import src.dataset trước sklearn để tránh xung đột thư viện OpenMP trên một số nền tảng.
from src.dataset import CONCEPT_NAMES
from src import runtime
from src.metrics import compute_metrics
from src.selection import select_threshold

import numpy as np

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "derm7pt", "resolution_fusion")
CONFIGS = {
    "blackbox": "efficientnet_b0/linear_e50_headlr0.001",
    "hard_cbm": "hard_cbm/efficientnet_b0_mlp128_e50_headlr0.001",
    "soft_cbm": "soft_cbm/efficientnet_b0_mlp128_e50_headlr0.001",
    "ecbm": "ecbm/efficientnet_b0_unified_exact_e50",
}
SEEDS = (42, 123, 2026)
METRICS = ("accuracy", "balanced_accuracy", "f1_macro", "roc_auc", "sensitivity", "specificity")
ARMS = ("224", "512x768", "fusion")


def load_records(path):
    records = runtime.load_json(path)["validation_predictions"]
    return {r["case_num"]: r for r in records}


def evaluate(y_true, y_prob, concept_true=None, concept_prob=None):
    threshold, _ = select_threshold(y_true, y_prob, "f1_macro")
    metrics = compute_metrics(y_true, (y_prob >= threshold).astype(int), y_prob)
    row = {k: float(metrics[k]) for k in METRICS}
    row["threshold"] = threshold
    if concept_true is not None:
        per = {c: float(np.mean(concept_prob[c].argmax(1) == concept_true[c])) for c in CONCEPT_NAMES}
        row["concept_accuracy"] = float(np.mean(list(per.values())))
        row["per_concept_accuracy"] = per
    return row


def fuse_pair(model, low_path, high_path):
    low, high = load_records(low_path), load_records(high_path)
    if set(low) != set(high):
        raise ValueError(f"{model}: hai file không cùng tập ca validation")
    cases = sorted(low)
    y_true = np.array([low[c]["y_true"] for c in cases])
    if not np.array_equal(y_true, [high[c]["y_true"] for c in cases]):
        raise ValueError(f"{model}: nhãn không khớp")
    prob = {"224": np.array([low[c]["y_prob"] for c in cases]),
            "512x768": np.array([high[c]["y_prob"] for c in cases])}
    prob["fusion"] = (prob["224"] + prob["512x768"]) / 2
    has_concepts = "concept_probabilities" in low[cases[0]]
    out = {}
    for arm in ARMS:
        if not has_concepts:
            out[arm] = evaluate(y_true, prob[arm])
            continue
        truth = {g: np.array([low[c]["concept_true"][g] for c in cases]) for g in CONCEPT_NAMES}
        cp = {"224": {g: np.array([low[c]["concept_probabilities"][g] for c in cases]) for g in CONCEPT_NAMES},
              "512x768": {g: np.array([high[c]["concept_probabilities"][g] for c in cases]) for g in CONCEPT_NAMES}}
        cp["fusion"] = {g: (cp["224"][g] + cp["512x768"][g]) / 2 for g in CONCEPT_NAMES}
        out[arm] = evaluate(y_true, prob[arm], truth, cp[arm])
    return out


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)), "per_seed": values.tolist()}


def decide(runs, has_concepts):
    def best_single(metric):
        return max(("224", "512x768"), key=lambda arm: np.mean([r[arm][metric] for r in runs]))

    fused = lambda metric: np.array([r["fusion"][metric] for r in runs])
    single = lambda arm, metric: np.array([r[arm][metric] for r in runs])
    ref_auc = best_single("roc_auc")
    k1 = bool(fused("roc_auc").mean() >= single(ref_auc, "roc_auc").mean()
              and np.sum(fused("roc_auc") >= single(ref_auc, "roc_auc")) >= 2)
    ref_f1 = best_single("f1_macro")
    guard = bool(fused("f1_macro").mean() >= single(ref_f1, "f1_macro").mean() - 0.02)
    result = {"K1_reference": ref_auc, "K1_diagnosis": k1, "G_reference": ref_f1, "G_guard": guard,
              "useful_for_diagnosis": k1 and guard}
    if has_concepts:
        ref_c = best_single("concept_accuracy")
        k2 = bool(fused("concept_accuracy").mean() >= single(ref_c, "concept_accuracy").mean() + 0.01
                  and np.sum(fused("concept_accuracy") > single(ref_c, "concept_accuracy")) >= 2)
        result.update({"K2_reference": ref_c, "K2_concept": k2, "useful_for_concepts": k2 and guard})
    return result


def summarize(pairs):
    summary = {}
    for model, seed_paths in pairs.items():
        runs = [fuse_pair(model, low, high) for low, high in seed_paths]
        has_concepts = "concept_accuracy" in runs[0]["224"]
        keys = METRICS + (("concept_accuracy",) if has_concepts else ())
        block = {arm: {k: mean_sd([r[arm][k] for r in runs]) for k in keys} for arm in ARMS}
        if has_concepts:
            for arm in ARMS:
                block[arm]["per_concept_accuracy"] = {
                    c: mean_sd([r[arm]["per_concept_accuracy"][c] for r in runs]) for c in CONCEPT_NAMES}
        block["decision"] = decide(runs, has_concepts)
        block["thresholds"] = {arm: [r[arm]["threshold"] for r in runs] for arm in ARMS}
        summary[model] = block
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t4_224_dir", default=None,
                        help="Thư mục output Kaggle v3 (train/<model>_seed<seed>/result.json) cho cặp phụ")
    args = parser.parse_args()
    results = lambda model, suffix, seed: os.path.join(PROJECT_ROOT, "results", "derm7pt",
                                                       CONFIGS[model] + suffix, f"seed{seed}.json")
    primary = {m: [(results(m, "", s), results(m, "_r512x768", s)) for s in SEEDS] for m in CONFIGS}
    output = {"plan": "plan.json", "test_read": False, "primary": summarize(primary)}
    if args.t4_224_dir:
        secondary = {m: [(os.path.join(args.t4_224_dir, "train", f"{m}_seed{s}", "result.json"),
                          results(m, "_r512x768", s)) for s in SEEDS] for m in CONFIGS}
        output["secondary_same_device_t4"] = summarize(secondary)
    runtime.save_json(os.path.join(OUT_DIR, "summary.json"), output)
    for name in ("primary", "secondary_same_device_t4"):
        if name not in output:
            continue
        print(f"== {name}")
        for model, block in output[name].items():
            for arm in ARMS:
                row = block[arm]
                cells = [f"{k}={row[k]['mean']:.3f}±{row[k]['sd']:.3f}" for k in
                         ("accuracy", "balanced_accuracy", "f1_macro", "roc_auc", "concept_accuracy") if k in row]
                print(f"{model:9s} {arm:8s} " + "  ".join(cells))
            print(f"{model:9s} decision {json.dumps(block['decision'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
