# Bảng test chính của quy trình chung 50 epochs và các so sánh paired bootstrap đã khai báo trong test_plan.json.
# Chỉ đọc kết quả đã lưu; không chạy model, không chọn lại ngưỡng.


import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from sklearn.metrics import f1_score

from src.metrics import compute_metrics
from src import runtime

RESULTS = os.path.join(PROJECT_ROOT, "results", "derm7pt")
PLAN_PATH = os.path.join(RESULTS, "test_plan.json")
SEEDS = ("42", "123", "2026")
NUM_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261005


# Dự đoán test từng seed (sắp theo case_num) và metrics tổng hợp của mỗi model/chế độ.
def collect():
    plan = runtime.load_json(PLAN_PATH)
    runs = {}
    for model in ("efficientnet_b0", "soft_cbm", "hard_cbm"):
        cohort = os.path.join(PROJECT_ROOT, plan["models"][model]["cohort"])
        seeds = {}
        for seed in SEEDS:
            run = runtime.load_json(os.path.join(cohort, f"seed{seed}.json"))
            records = sorted(run["test_predictions"], key=lambda r: r["case_num"])
            concept = run.get("test_concept_metrics", {}).get("overall_f1_macro_all_defined")
            seeds[seed] = {"case_ids": [r["case_num"] for r in records], "y_true": [r["y_true"] for r in records],
                           "y_pred": [r["y_pred"] for r in records], "y_prob": [r["y_prob"] for r in records],
                           "concept_f1": concept}
        intervention = None
        if model != "efficientnet_b0":
            summary = runtime.load_json(os.path.join(cohort, "intervention", "summary.json"))
            intervention = {"m7_f1_macro": summary["curve"][-1]["f1_macro"]["values"],
                            "curve_auc_f1_macro": summary["intervention_curve_auc"]["f1_macro"]["absolute"]["values"]}
        runs[model] = {"seeds": seeds, "intervention": intervention}
    ecbm_dir = os.path.join(RESULTS, "ecbm", "test_modes")
    ecbm_runs = {seed: runtime.load_json(os.path.join(ecbm_dir, f"seed{seed}.json")) for seed in SEEDS}
    for mode in plan["models"]["ecbm"]["modes"]:
        seeds, m7, auc = {}, [], []
        for seed, run in ecbm_runs.items():
            block = run["modes"][mode]
            records = sorted(block["test_predictions"], key=lambda r: r["case_num"])
            seeds[seed] = {"case_ids": [r["case_num"] for r in records], "y_true": [r["y_true"] for r in records],
                           "y_pred": [r["y_pred"] for r in records], "y_prob": [r["y_prob"] for r in records],
                           "concept_f1": block["test_concept_metrics"]["overall_f1_macro_all_defined"]}
            m7.append(block["test_metrics_m7"]["f1_macro"])
            auc.append(block["intervention_curve_auc"]["f1_macro"]["absolute"])
        runs[mode] = {"seeds": seeds, "intervention": {"m7_f1_macro": m7, "curve_auc_f1_macro": auc}}
    reference = runs["efficientnet_b0"]["seeds"][SEEDS[0]]
    for name, run in runs.items():
        for seed, data in run["seeds"].items():
            if data["case_ids"] != reference["case_ids"] or data["y_true"] != reference["y_true"]:
                raise ValueError(f"{name} seed {seed}: test cases or labels differ")
            if len(data["case_ids"]) != 395:
                raise ValueError("Test split phải có đúng 395 mẫu")
    return plan, runs


def table(runs):
    rows = {}
    for name, run in runs.items():
        metrics = [compute_metrics(d["y_true"], d["y_pred"], d["y_prob"]) for d in run["seeds"].values()]
        row = {key: runtime.mean_sd([m[key] for m in metrics])
               for key in ("f1_macro", "balanced_accuracy", "roc_auc", "sensitivity", "specificity")}
        concept = [d["concept_f1"] for d in run["seeds"].values()]
        row["concept_f1_macro"] = runtime.mean_sd(concept) if all(c is not None for c in concept) else None
        if run["intervention"]:
            row["m7_f1_macro"] = runtime.mean_sd(run["intervention"]["m7_f1_macro"])
            row["curve_auc_f1_macro"] = runtime.mean_sd(run["intervention"]["curve_auc_f1_macro"])
        rows[name] = row
    return rows


# Paired stratified case bootstrap: lấy mẫu có hoàn lại trong từng lớp, cùng ca cho mọi model/seed,
# tính Macro F1 từng seed rồi lấy mean; không ensemble predictions.
def bootstrap(runs, pairs):
    y = np.asarray(runs["efficientnet_b0"]["seeds"][SEEDS[0]]["y_true"])
    classes = [np.flatnonzero(y == c) for c in (0, 1)]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = [np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in classes])
               for _ in range(NUM_BOOTSTRAP)]
    preds = {name: [np.asarray(d["y_pred"]) for d in run["seeds"].values()] for name, run in runs.items()}
    mean_f1 = lambda name, idx: float(np.mean([f1_score(y[idx], p[idx], average="macro", zero_division=0)
                                               for p in preds[name]]))
    full = np.arange(len(y))
    out = []
    for a, b in pairs:
        deltas = np.array([mean_f1(a, idx) - mean_f1(b, idx) for idx in samples])
        low, high = np.percentile(deltas, [2.5, 97.5])
        out.append({"a": a, "b": b, "delta_mean_test_f1": mean_f1(a, full) - mean_f1(b, full),
                    "ci95_percentile": [float(low), float(high)], "contains_zero": bool(low <= 0 <= high)})
    return out


def write(rows, comparisons):
    with open(os.path.join(RESULTS, "test_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"table": rows, "comparisons": comparisons, "num_bootstrap": NUM_BOOTSTRAP,
                   "bootstrap_seed": BOOTSTRAP_SEED}, handle, ensure_ascii=False, indent=2)


def main():
    plan, runs = collect()
    rows = table(runs)
    comparisons = bootstrap(runs, plan["comparisons"]["pairs"])
    write(rows, comparisons)
    for c in comparisons:
        print(f"{c['a']} - {c['b']}: {c['delta_mean_test_f1']:+.4f} "
              f"[{c['ci95_percentile'][0]:+.4f}, {c['ci95_percentile'][1]:+.4f}]")


if __name__ == "__main__":
    main()
