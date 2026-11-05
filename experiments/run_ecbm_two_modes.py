# ECBM hai chế độ trên validation, theo results/derm7pt/ecbm/two_mode/plan.json.
# Chế độ tự động dùng trọng số 1/1/0.01 (chẩn đoán + concept hiển thị); khi bác sĩ sửa concept thì
# chuyển sang một bộ trọng số khác. So với hệ hiện tại (S0: một bộ 1/1/1 cho cả hai việc). Không đọc test.


import argparse
import hashlib
import json
import os
import sys
from itertools import combinations

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import src.dataset trước sklearn để tránh xung đột thư viện OpenMP trên một số nền tảng.
from src.dataset import CONCEPT_NAMES
from src import runtime

import numpy as np
import torch

from experiments.run_ecbm_lambda_study import _stored_validation, _trapezoid, evaluate_weights, exact_outputs, load_seed
from src.metrics import compute_concept_metrics, compute_metrics
from src.selection import select_threshold

STUDY_DIR = os.path.join(PROJECT_ROOT, "results", "derm7pt", "ecbm", "two_mode")
PLAN_PATH = os.path.join(STUDY_DIR, "plan.json")
AUTO_WEIGHTS = (1.0, 1.0, 0.01)
S0_WEIGHTS = (1.0, 1.0, 1.0)
CANDIDATES = ("joint_lambda1", "concept_only", "global_on_displayed")
GLOBAL_SCALE = 3.0   # λ_global trong bước can thiệp của GradientInference.py gốc
COUNT = len(CONCEPT_NAMES)
RATES = [m / COUNT for m in range(COUNT + 1)]


# Concept hiển thị (argmax marginal từng nhóm) và concept accuracy dưới một bộ trọng số, khi chưa sửa gì.
def displayed_concepts(data, weights):
    _, probabilities = exact_outputs(data["model"], data["features"], weights)
    predicted = np.stack([p.argmax(-1) for p in probabilities], axis=1)
    truth = data["concepts"].cpu().numpy()
    metrics = compute_concept_metrics({c: truth[:, g] for g, c in enumerate(CONCEPT_NAMES)},
                                      {c: predicted[:, g] for g, c in enumerate(CONCEPT_NAMES)},
                                      {c: k for c, k in zip(CONCEPT_NAMES, data["model"].state_counts)})
    return predicted, metrics


# Ứng viên global_on_displayed: p(y|c) = softmax(-3·E_global(c,·)), c = GT ở nhóm đã sửa, concept hiển thị ở nhóm còn lại.
def global_on_displayed(data, displayed):
    model, labels = data["model"], data["labels"]
    truth = data["concepts"]
    shown = torch.as_tensor(displayed, device=truth.device, dtype=truth.dtype)

    def scores_for(subset):
        profile = shown.clone()
        for group in subset:
            profile[:, group] = truth[:, group]
        with torch.no_grad():
            energies = model.global_class_energies(profile)
        return (-GLOBAL_SCALE * energies).softmax(-1)[:, 1].cpu().numpy().astype(float)

    baseline = scores_for(())
    threshold, _ = select_threshold(labels, baseline, "f1_macro")
    curve, full_scores = [], None
    for m in range(COUNT + 1):
        values = []
        for subset in combinations(range(COUNT), m):
            scores = scores_for(subset)
            values.append(compute_metrics(labels, (scores >= threshold).astype(int), scores)["f1_macro"])
            if m == COUNT:
                full_scores = scores
        curve.append(float(np.mean(values)))
    return {"decision_threshold": float(threshold), "curve": {"f1_macro": curve},
            "m0_metrics": compute_metrics(labels, (baseline >= threshold).astype(int), baseline)}, full_scores


def _two_mode_curve(auto_f1_m0, candidate_curve):
    curve = [auto_f1_m0] + list(candidate_curve[1:])
    return curve, _trapezoid(curve, RATES)


def run_seed(seed, entry, manifest_path, device):
    data = load_seed(entry, manifest_path, device)
    labels = data["labels"]
    with torch.no_grad():
        auto, _ = evaluate_weights(data, AUTO_WEIGHTS)
        s0, s0_full = evaluate_weights(data, S0_WEIGHTS)
        concept_only, concept_only_full = evaluate_weights(data, (0.0, 1.0, 1.0))
        auto_displayed, auto_concepts = displayed_concepts(data, AUTO_WEIGHTS)
        _, s0_concepts = displayed_concepts(data, S0_WEIGHTS)
        global_shown, global_full = global_on_displayed(data, auto_displayed)

    # Chế độ tự động phải tái hiện scores validation đã lưu của cohort (λ = 0.01).
    stored_scores, stored_f1 = _stored_validation(entry, data["case_ids"])
    max_error = float(np.max(np.abs(np.asarray(auto["baseline_scores"]) - stored_scores)))
    if max_error > 1e-4 or abs(auto["validation_metrics_m0"]["f1_macro"] - stored_f1) > 1e-9:
        raise ValueError(f"Seed {seed}: chế độ tự động không tái hiện validation đã lưu (max |Δp| {max_error:.2e})")

    auto_pred = (np.asarray(auto["baseline_scores"]) >= auto["decision_threshold"]).astype(int)
    auto_f1 = auto["validation_metrics_m0"]["f1_macro"]
    candidates = {
        "joint_lambda1": (s0["decision_threshold"], s0["curve"]["f1_macro"], s0_full),
        "concept_only": (concept_only["decision_threshold"], concept_only["curve"]["f1_macro"], concept_only_full),
        "global_on_displayed": (global_shown["decision_threshold"], global_shown["curve"]["f1_macro"], global_full),
    }
    two_mode = {}
    for name, (threshold, curve, full_scores) in candidates.items():
        combined, area = _two_mode_curve(auto_f1, curve)
        full_pred = (np.asarray(full_scores) >= threshold).astype(int)
        two_mode[name] = {"intervention_threshold": float(threshold), "curve_f1_macro": combined,
                          "curve_auc_f1_macro": area, "f1_macro_m7": combined[-1],
                          "flips_m7_vs_auto": int(np.sum(full_pred != auto_pred)),
                          "roc_auc_m7": compute_metrics(labels, full_pred, full_scores)["roc_auc"]}
    s0_area = s0["intervention_curve_auc"]["f1_macro"]["absolute"]
    s0_pred = (np.asarray(s0["baseline_scores"]) >= s0["decision_threshold"]).astype(int)
    result = {
        "seed": int(seed), "checkpoint": entry, "case_ids": data["case_ids"],
        "baseline_check": {"max_abs_score_error_vs_stored": max_error, "stored_validation_f1_macro": stored_f1},
        "auto": {"weights": list(AUTO_WEIGHTS), "decision_threshold": auto["decision_threshold"],
                 "metrics_m0": auto["validation_metrics_m0"],
                 "concept_accuracy": auto_concepts["overall_concept_accuracy"],
                 "concept_f1_macro": auto_concepts["overall_f1_macro_all_defined"],
                 "per_concept_accuracy": {c: v["accuracy"] for c, v in auto_concepts["per_concept"].items()}},
        "s0": {"weights": list(S0_WEIGHTS), "decision_threshold": s0["decision_threshold"],
               "metrics_m0": s0["validation_metrics_m0"], "curve_f1_macro": s0["curve"]["f1_macro"],
               "curve_auc_f1_macro": s0_area, "f1_macro_m7": s0["validation_metrics_m7"]["f1_macro"],
               "flips_m7": s0["flips_m7"],
               "concept_accuracy": s0_concepts["overall_concept_accuracy"],
               "concept_f1_macro": s0_concepts["overall_f1_macro_all_defined"],
               "per_concept_accuracy": {c: v["accuracy"] for c, v in s0_concepts["per_concept"].items()},
               "agreement_with_auto_m0": float(np.mean(s0_pred == auto_pred))},
        "two_mode": two_mode,
    }
    print(f"seed {seed}: concept tự động {100 * result['auto']['concept_accuracy']:.1f}% vs S0 "
          f"{100 * result['s0']['concept_accuracy']:.1f}% | F1 m0 {auto_f1:.4f} vs S0 "
          f"{result['s0']['metrics_m0']['f1_macro']:.4f} | curve S0 {s0_area:.4f} | " +
          ", ".join(f"{n} {v['curve_auc_f1_macro']:.4f}" for n, v in two_mode.items()), flush=True)
    return result


def summarize(plan_text, per_seed):
    seeds = list(per_seed)
    stat = lambda values: runtime.mean_sd(values)
    rows = {"auto": {"concept_accuracy": stat([per_seed[s]["auto"]["concept_accuracy"] for s in seeds]),
                     "f1_macro_m0": stat([per_seed[s]["auto"]["metrics_m0"]["f1_macro"] for s in seeds]),
                     "roc_auc_m0": stat([per_seed[s]["auto"]["metrics_m0"]["roc_auc"] for s in seeds])},
            "s0": {"concept_accuracy": stat([per_seed[s]["s0"]["concept_accuracy"] for s in seeds]),
                   "f1_macro_m0": stat([per_seed[s]["s0"]["metrics_m0"]["f1_macro"] for s in seeds]),
                   "roc_auc_m0": stat([per_seed[s]["s0"]["metrics_m0"]["roc_auc"] for s in seeds]),
                   "curve_auc_f1_macro": stat([per_seed[s]["s0"]["curve_auc_f1_macro"] for s in seeds]),
                   "f1_macro_m7": stat([per_seed[s]["s0"]["f1_macro_m7"] for s in seeds]),
                   "flips_m7": stat([per_seed[s]["s0"]["flips_m7"] for s in seeds])},
            "two_mode": {name: {key: stat([per_seed[s]["two_mode"][name][key] for s in seeds])
                                for key in ("curve_auc_f1_macro", "f1_macro_m7", "flips_m7_vs_auto", "roc_auc_m7")}
                         for name in CANDIDATES}}
    best = max(rows["two_mode"][n]["curve_auc_f1_macro"]["mean"] for n in CANDIDATES)
    chosen = next(n for n in CANDIDATES if best - rows["two_mode"][n]["curve_auc_f1_macro"]["mean"] < 1e-4)
    k1 = rows["auto"]["concept_accuracy"]["mean"] >= rows["s0"]["concept_accuracy"]["mean"] + 0.02
    k2 = rows["two_mode"][chosen]["curve_auc_f1_macro"]["mean"] >= rows["s0"]["curve_auc_f1_macro"]["mean"] - 0.005
    k3 = rows["auto"]["f1_macro_m0"]["mean"] >= rows["s0"]["f1_macro_m0"]["mean"] - 0.01
    return {"plan_sha256": hashlib.sha256(plan_text.encode("utf-8")).hexdigest(), "seeds": seeds,
            "test_read": False, "summary": rows, "chosen_intervention_mode": chosen,
            "criteria": {"K1_concept": k1, "K2_curve": k2, "K3_auto_f1": k3},
            "decision": "ĐẠT" if k1 and k2 and k3 else "KHÔNG ĐẠT"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="ECBM hai chế độ trên validation (không đọc test)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    summary_path = os.path.join(STUDY_DIR, "summary.json")
    if os.path.exists(summary_path) and not args.overwrite:
        raise FileExistsError(f"{summary_path} đã tồn tại; dùng --overwrite nếu chủ động chạy lại")
    plan_text = open(PLAN_PATH, encoding="utf-8").read()
    plan = json.loads(plan_text)
    manifest_path = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
    device = runtime.get_device(args.device)
    per_seed = {}
    for seed, entry in plan["fixed"]["checkpoints"].items():
        per_seed[seed] = run_seed(seed, entry, manifest_path, device)
        runtime.save_json(os.path.join(STUDY_DIR, f"seed{seed}.json"), per_seed[seed])
    summary = summarize(plan_text, per_seed)
    runtime.save_json(summary_path, summary)
    print(json.dumps({"chosen": summary["chosen_intervention_mode"], **summary["criteria"],
                      "decision": summary["decision"]}, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
