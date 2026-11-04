# Chấm ba chế độ suy luận exact của ECBM theo results/derm7pt/test_plan.json.
# --stage freeze: chỉ đọc validation, chốt ngưỡng từng chế độ/seed vào frozen_modes.json.
# --stage test: đọc test một lần với ngưỡng đã chốt, kèm can thiệp concept GT đủ 128 tổ hợp.


import argparse
import hashlib
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import src.dataset trước sklearn để tránh xung đột thư viện OpenMP trên một số nền tảng.
from src.dataset import CONCEPT_NAMES, CONCEPT_NUM_CLASSES
from src.dataset import compute_train_observed_states
from src import runtime

import numpy as np
import pandas as pd
import torch

from experiments.run_ecbm_lambda_study import REPORT_KEYS, evaluate_weights, exact_outputs, load_seed
from src.metrics import compute_concept_metrics, compute_metrics
from src.selection import select_threshold

PLAN_PATH = os.path.join(PROJECT_ROOT, "results", "derm7pt", "test_plan.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "derm7pt", "ecbm", "test_modes")
FROZEN_PATH = os.path.join(OUTPUT_DIR, "frozen_modes.json")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
THRESHOLD_METRIC = "f1_macro"


def _file_sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _plan():
    plan = json.load(open(PLAN_PATH, encoding="utf-8"))
    ecbm = plan["models"]["ecbm"]
    checkpoint_dir = ecbm["cohort"].replace("results/", "checkpoints/")
    entries = {seed: {"path": f"{checkpoint_dir}/seed{seed}.pth", "sha256": sha256}
               for seed, sha256 in ecbm["checkpoints"].items()}
    return plan, ecbm["modes"], entries


def _e_class_scores(data):
    with torch.no_grad():
        return (-data["model"].class_energies(data["features"])).softmax(-1)[:, 1].cpu().numpy().astype(float)


# Chỉ đọc validation: lấy ngưỡng từ nguồn đã khai báo, kiểm tra tái hiện, chọn ngưỡng cho concept-only.
def freeze(device, overwrite=False):
    if os.path.exists(os.path.join(OUTPUT_DIR, "summary.json")):
        raise RuntimeError("Test đã được chấm; không chốt lại ngưỡng sau khi đọc test")
    if os.path.exists(FROZEN_PATH) and not overwrite:
        raise FileExistsError(f"{FROZEN_PATH} đã tồn tại; dùng --overwrite nếu chủ động chốt lại trước khi chấm test")
    plan, modes, entries = _plan()
    selection_path = os.path.join(PROJECT_ROOT, modes["ecbm_full"]["threshold_source"])
    selection = json.load(open(selection_path, encoding="utf-8"))
    if [1.0, 1.0, selection["chosen_lambda_global"]] != modes["ecbm_full"]["energy_weights"]:
        raise ValueError("ecbm_full weights differ from the λ chosen in lambda_global_study")
    frozen = {name: {"role": mode["role"], "energy_weights": mode["energy_weights"], "decision_thresholds": {},
                     "validation_metrics": {}} for name, mode in modes.items()}
    e_class_thresholds = {}
    for seed, entry in entries.items():
        data = load_seed(entry, MANIFEST_PATH, device, split="valid")
        declared = {"ecbm_full": selection["chosen_decision_thresholds"][seed],
                    "ecbm_paper": data["checkpoint"]["decision_thresholds"]["exact"],
                    "ecbm_concept_only": None}
        for name, mode in modes.items():
            scores, _ = exact_outputs(data["model"], data["features"], tuple(mode["energy_weights"]))
            selected, _ = select_threshold(data["labels"], scores, THRESHOLD_METRIC)
            if declared[name] is not None and abs(selected - declared[name]) > 1e-5:
                raise ValueError(f"Seed {seed} {name}: validation threshold {selected} differs from declared {declared[name]}")
            threshold = selected if declared[name] is None else float(declared[name])
            metrics = compute_metrics(data["labels"], (scores >= threshold).astype(int), scores)
            frozen[name]["decision_thresholds"][seed] = float(threshold)
            frozen[name]["validation_metrics"][seed] = {key: metrics[key] for key in REPORT_KEYS}
        e_class_thresholds[seed] = float(select_threshold(data["labels"], _e_class_scores(data), THRESHOLD_METRIC)[0])
        print(f"seed {seed}: " + " | ".join(
            f"{name} thr {frozen[name]['decision_thresholds'][seed]:.4f} F1 {frozen[name]['validation_metrics'][seed]['f1_macro']:.4f}"
            for name in modes))
    payload = {"created_from": os.path.relpath(PLAN_PATH, PROJECT_ROOT), "test_plan_sha256": _file_sha256(PLAN_PATH),
               "lambda_selection_sha256": _file_sha256(selection_path), "threshold_selection":
               f"maximize_validation_{THRESHOLD_METRIC}_tie_nearest_0.5", "checkpoints": entries,
               "modes": frozen, "e_class_only_thresholds": e_class_thresholds, "test_read": False}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    runtime.save_json(FROZEN_PATH, payload)
    print(f"Đã chốt ngưỡng: {FROZEN_PATH}")


# Đọc test một lần với ngưỡng đã chốt; không chọn lại gì trên test.
def test(device, overwrite=False):
    if not os.path.exists(FROZEN_PATH):
        raise FileNotFoundError("Chưa có frozen_modes.json; chạy --stage freeze trước")
    frozen = json.load(open(FROZEN_PATH, encoding="utf-8"))
    if frozen["test_plan_sha256"] != _file_sha256(PLAN_PATH):
        raise ValueError("test_plan.json changed after thresholds were frozen")
    outputs = [os.path.join(OUTPUT_DIR, f"seed{seed}.json") for seed in frozen["checkpoints"]]
    outputs += [os.path.join(OUTPUT_DIR, name) for name in ("summary.json", "report.md")]
    if any(os.path.exists(path) for path in outputs) and not overwrite:
        raise FileExistsError("Kết quả test ECBM đã tồn tại; dùng --overwrite nếu chủ động chạy lại")
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"is_inconsistent_profile": bool})
    train_df = manifest[manifest["split"] == "train"].reset_index(drop=True)
    runs = {}
    for seed, entry in frozen["checkpoints"].items():
        data = load_seed(entry, MANIFEST_PATH, device, split="test")
        if len(data["labels"]) != 395:
            raise ValueError("Test split phải có đúng 395 mẫu")
        observed = compute_train_observed_states(train_df, data["schema"]["label_mapping"])
        concept_true = {name: data["concepts"][:, i].cpu().numpy() for i, name in enumerate(CONCEPT_NAMES)}
        e_class_pred = (_e_class_scores(data) >= frozen["e_class_only_thresholds"][seed]).astype(int)
        modes = {}
        for name, mode in frozen["modes"].items():
            threshold = mode["decision_thresholds"][seed]
            result, full_scores = evaluate_weights(data, tuple(mode["energy_weights"]), threshold=threshold, prefix="test")
            scores, concept_probabilities = exact_outputs(data["model"], data["features"], tuple(mode["energy_weights"]))
            pred = (scores >= threshold).astype(int)
            concept_pred = {n: p.argmax(1) for n, p in zip(CONCEPT_NAMES, concept_probabilities)}
            result["test_concept_metrics"] = compute_concept_metrics(concept_true, concept_pred, CONCEPT_NUM_CLASSES,
                                                                    train_observed_states=observed)
            result["agreement_with_e_class"] = float(np.mean(pred == e_class_pred))
            result["test_predictions"] = [
                {"case_num": c, "y_true": int(y), "y_prob": float(p), "y_pred": int(q), "y_prob_full_intervention": float(f)}
                for c, y, p, q, f in zip(data["case_ids"], data["labels"], scores, pred, full_scores)]
            result.pop("baseline_scores")
            modes[name] = {"role": mode["role"], "energy_weights": mode["energy_weights"], **result}
            print(f"seed {seed} {name}: test F1 {result['test_metrics_m0']['f1_macro']:.4f} | "
                  f"m7 {result['test_metrics_m7']['f1_macro']:.4f} | curve AUC "
                  f"{result['intervention_curve_auc']['f1_macro']['absolute']:.4f}")
        runs[seed] = {"model": "ecbm", "mode": "final_test", "seed": int(seed), "checkpoint": entry,
                      "best_epoch": data["best_epoch"], "frozen_modes_sha256": _file_sha256(FROZEN_PATH), "modes": modes}
        runtime.save_json(os.path.join(OUTPUT_DIR, f"seed{seed}.json"), runs[seed])
    summary = {}
    for name in frozen["modes"]:
        rows = [run["modes"][name] for run in runs.values()]
        summary[name] = {"role": rows[0]["role"], "energy_weights": rows[0]["energy_weights"],
                         **{f"m0_{key}": runtime.mean_sd([r["test_metrics_m0"][key] for r in rows]) for key in REPORT_KEYS},
                         "m7_f1_macro": runtime.mean_sd([r["test_metrics_m7"]["f1_macro"] for r in rows]),
                         "curve_auc_f1_macro": runtime.mean_sd([r["intervention_curve_auc"]["f1_macro"]["absolute"] for r in rows]),
                         "concept_f1_macro": runtime.mean_sd([r["test_concept_metrics"]["overall_f1_macro_all_defined"] for r in rows]),
                         "flips_m7": runtime.mean_sd([r["flips_m7"] for r in rows]),
                         "agreement_with_e_class": runtime.mean_sd([r["agreement_with_e_class"] for r in rows])}
    runtime.save_json(os.path.join(OUTPUT_DIR, "summary.json"), {"frozen_modes_sha256": _file_sha256(FROZEN_PATH),
                                                                 "num_cases": 395, "modes": summary})
    _write_report(summary)


def _write_report(summary):
    fmt = lambda s, pct=False: (f"{100 * s['mean']:.2f}% ± {100 * s['sd']:.2f}" if pct else f"{s['mean']:.4f} ± {s['sd']:.4f}")
    lines = ["# ECBM trên test (395 ca, 3 seeds), ba chế độ suy luận exact", "",
             "Ngưỡng chốt trên validation trong [frozen_modes.json](frozen_modes.json), theo [test_plan.json](../../test_plan.json).", "",
             "| Chế độ | Vai trò | Macro F1 | BAcc | ROC-AUC | Concept F1 | F1 sửa đủ 7 | Curve AUC F1 | Đổi nhãn m=7 | Trùng E_class |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in summary.items():
        lines.append(f"| {name} | {s['role']} | {fmt(s['m0_f1_macro'])} | {fmt(s['m0_balanced_accuracy'], True)} | "
                     f"{fmt(s['m0_roc_auc'])} | {fmt(s['concept_f1_macro'])} | {fmt(s['m7_f1_macro'])} | "
                     f"{fmt(s['curve_auc_f1_macro'])} | {s['flips_m7']['mean']:.1f} | {fmt(s['agreement_with_e_class'], True)} |")
    lines += ["", "Mean ± sample SD qua seeds 42/123/2026. Can thiệp thay nhóm concept bằng trạng thái GT, ngưỡng đóng băng."]
    with open(os.path.join(OUTPUT_DIR, "report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="ECBM: chốt ngưỡng ba chế độ trên validation, rồi chấm test một lần")
    parser.add_argument("--stage", choices=["freeze", "test"], required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    device = runtime.get_device(args.device)
    (freeze if args.stage == "freeze" else test)(device, args.overwrite)


if __name__ == "__main__":
    main()
