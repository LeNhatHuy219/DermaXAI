# Study λ_global suy luận của ECBM trên validation, theo results/derm7pt/ecbm/lambda_global_study/study_plan.json.
# Với mỗi λ: chọn ngưỡng ở m=0, đóng băng, rồi clamp concept GT cho đủ 128 tổ hợp nhóm (exact inference).
# Không đọc test; ghi kết quả từng seed, frozen_selection.json và report.md vào thư mục study.


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
from src.dataset import CONCEPT_NAMES, Derm7ptDataset

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments import run_soft_cbm as shared
from experiments.run_ecbm import _build_model, validate_ecbm_config
from src.ecbm import exact_inference
from src.metrics import compute_metrics
from src.protocol import validate_concept_schema, validate_manifest
from src.selection import select_threshold
from src.transforms import get_transforms

STUDY_DIR = os.path.join(PROJECT_ROOT, "results", "derm7pt", "ecbm", "lambda_global_study")
PLAN_PATH = os.path.join(STUDY_DIR, "study_plan.json")
REPORT_KEYS = ("f1_macro", "balanced_accuracy", "roc_auc", "sensitivity", "specificity")


def _sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _trapezoid(values, x):
    # NumPy 2.4 bỏ trapz; giữ tương thích NumPy 1.x như src/intervention.py.
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(values, x=x))


# Feature ảnh của một split, nhãn và concept GT; kiểm tra hash checkpoint, manifest và concept schema.
def load_seed(entry, manifest_path, device, split="valid"):
    checkpoint_path = os.path.join(PROJECT_ROOT, entry["path"])
    if _sha256(checkpoint_path) != entry["sha256"]:
        raise ValueError(f"Checkpoint {entry['path']} differs from the frozen hash")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    validate_manifest(checkpoint, manifest_path)
    schema = validate_concept_schema(checkpoint, manifest_path)
    config = checkpoint["config"]
    validate_ecbm_config(config)
    model = _build_model(config, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = Derm7ptDataset(manifest_path=manifest_path, project_root=PROJECT_ROOT, split=split,
                             label_mapping=schema["label_mapping"],
                             transform=get_transforms("valid", target_size=config["target_size"], augment=False))
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    features, labels, concepts = [], [], []
    with torch.no_grad():
        for batch in loader:
            features.append(model.extract_features(batch["image"].to(device)))
            labels.append(batch["label"])
            concepts.append(batch["concept_indices"])
    return {"model": model, "features": torch.cat(features), "labels": torch.cat(labels).numpy().astype(int),
            "concepts": torch.cat(concepts).to(device), "case_ids": dataset.df["case_num"].astype(int).tolist(),
            "best_epoch": int(checkpoint["epoch"]), "checkpoint": checkpoint, "schema": schema, "df": dataset.df}


# Scores validation và Macro F1 đã lưu trong JSON cohort, để kiểm tra tái hiện ở λ=0.01.
def _stored_validation(entry, case_ids):
    stored = json.load(open(os.path.join(PROJECT_ROOT, entry["path"].replace("checkpoints/", "results/")
                                         .replace(".pth", ".json")), encoding="utf-8"))
    scores = {int(r["case_num"]): float(r["y_prob"]) for r in stored["validation_predictions"]}
    return np.array([scores[c] for c in case_ids]), float(stored["validation_metrics"]["f1_macro"])


def exact_outputs(model, features, weights, clamp=None):
    output = exact_inference(model, features, weights, clamp=clamp)
    scores = np.clip(output["class_probabilities"][:, 1].cpu().numpy().astype(float), 0.0, 1.0)
    return scores, [p.cpu().numpy() for p in output["concept_probabilities"]]


def _scores(model, features, weights, clamp=None):
    return exact_outputs(model, features, weights, clamp)[0]


# Một bộ trọng số năng lượng trên một seed: ngưỡng chọn ở m=0 (hoặc truyền ngưỡng đã đóng băng),
# giữ cố định cho 128 tổ hợp can thiệp. Trả thêm scores khi sửa đủ bảy nhóm.
def evaluate_weights(data, weights, threshold=None, threshold_metric="f1_macro", prefix="validation"):
    model, features, labels, concepts = data["model"], data["features"], data["labels"], data["concepts"]
    baseline_scores = _scores(model, features, weights)
    if threshold is None:
        threshold, _ = select_threshold(labels, baseline_scores, threshold_metric)
    baseline_pred = (baseline_scores >= threshold).astype(int)
    count = len(CONCEPT_NAMES)
    subsets = []
    for m in range(count + 1):
        for subset in combinations(range(count), m):
            scores = (baseline_scores if m == 0 else
                      _scores(model, features, weights, clamp={g: concepts[:, g] for g in subset}))
            pred = (scores >= threshold).astype(int)
            metrics = compute_metrics(labels, pred, scores)
            subsets.append({"m": m, "groups": [CONCEPT_NAMES[g] for g in subset],
                            **{key: metrics[key] for key in ("f1_macro", "balanced_accuracy")}})
            if m == count:
                full_scores, full_pred, full_metrics = scores, pred, metrics
    rates = [m / count for m in range(count + 1)]
    curve = {key: [float(np.mean([s[key] for s in subsets if s["m"] == m])) for m in range(count + 1)]
             for key in ("f1_macro", "balanced_accuracy")}
    auc = {key: {"absolute": _trapezoid(curve[key], rates), "gain_over_baseline": _trapezoid(curve[key], rates) - curve[key][0]}
           for key in curve}
    baseline_metrics = compute_metrics(labels, baseline_pred, baseline_scores)
    result = {"decision_threshold": float(threshold),
              f"{prefix}_metrics_m0": {key: baseline_metrics[key] for key in REPORT_KEYS},
              f"{prefix}_metrics_m7": {key: full_metrics[key] for key in REPORT_KEYS},
              "flips_m7": int(np.sum(full_pred != baseline_pred)),
              "baseline_scores": baseline_scores.tolist(), "curve": curve, "intervention_curve_auc": auc,
              "subsets": subsets}
    return result, full_scores


def _evaluate_lambda(data, lambda_global, threshold_metric):
    result, _ = evaluate_weights(data, (1.0, 1.0, float(lambda_global)), threshold_metric=threshold_metric)
    return {"lambda_global": float(lambda_global), **result}


def run_study(device_name="auto", overwrite=False):
    plan_text = open(PLAN_PATH, encoding="utf-8").read()
    plan = json.loads(plan_text)
    rule = plan["selection_rule"]
    grid = [float(x) for x in plan["candidates"]["lambda_global"]]
    selection_path = os.path.join(STUDY_DIR, "frozen_selection.json")
    if os.path.exists(selection_path) and not overwrite:
        raise FileExistsError(f"{selection_path} đã tồn tại; dùng --overwrite nếu chủ động chạy lại")
    manifest_path = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
    device = shared.get_device(device_name)
    threshold_metric = "f1_macro"
    per_seed = {}
    for seed, entry in plan["fixed"]["checkpoints"].items():
        data = load_seed(entry, manifest_path, device)
        data["stored_scores"], data["stored_f1"] = _stored_validation(entry, data["case_ids"])
        model = data["model"]
        with torch.no_grad():
            class_scores = (-model.class_energies(data["features"])).softmax(-1)[:, 1].cpu().numpy()
        class_threshold, _ = select_threshold(data["labels"], class_scores, threshold_metric)
        class_pred = (class_scores >= class_threshold).astype(int)
        results = []
        for lambda_global in grid:
            with torch.no_grad():
                result = _evaluate_lambda(data, lambda_global, threshold_metric)
            scores = np.asarray(result["baseline_scores"])
            result["agreement_with_e_class"] = float(np.mean(
                (scores >= result["decision_threshold"]).astype(int) == class_pred))
            results.append(result)
            print(f"seed {seed} λ_global={lambda_global:g}: F1 m0 {result['validation_metrics_m0']['f1_macro']:.4f} "
                  f"m7 {result['validation_metrics_m7']['f1_macro']:.4f} | curve AUC "
                  f"{result['intervention_curve_auc']['f1_macro']['absolute']:.4f} | flips {result['flips_m7']}")
        # Kiểm tra tái hiện cohort đã lưu ở λ=0.01 trước khi dùng kết quả.
        reference = results[grid.index(0.01)]
        max_error = float(np.max(np.abs(np.asarray(reference["baseline_scores"]) - data["stored_scores"])))
        if max_error > 1e-4 or abs(reference["validation_metrics_m0"]["f1_macro"] - data["stored_f1"]) > 1e-9:
            raise ValueError(f"Seed {seed}: λ=0.01 không tái hiện validation đã lưu (max |Δp| {max_error:.2e})")
        payload = {"seed": int(seed), "checkpoint": entry, "best_epoch": data["best_epoch"],
                   "case_ids": data["case_ids"], "y_true": data["labels"].tolist(),
                   "baseline_check": {"max_abs_score_error_vs_stored": max_error,
                                      "stored_validation_f1_macro": data["stored_f1"]},
                   "e_class_only": {"decision_threshold": float(class_threshold),
                                    "validation_f1_macro": compute_metrics(data["labels"], class_pred, class_scores)["f1_macro"]},
                   "threshold_selection": "maximize_validation_f1_macro_tie_nearest_0.5",
                   "results": results}
        shared._save_json(os.path.join(STUDY_DIR, f"lambda_seed{seed}.json"), payload)
        per_seed[seed] = results

    # Áp dụng quy tắc đã chốt trong study_plan.json: ràng buộc F1 m=0, rồi tối đa curve AUC; hòa chọn λ nhỏ hơn.
    seeds = list(per_seed)
    summary = []
    for index, lambda_global in enumerate(grid):
        rows = [per_seed[s][index] for s in seeds]
        entry = {"lambda_global": lambda_global}
        for key in REPORT_KEYS:
            entry[f"m0_{key}"] = float(np.mean([r["validation_metrics_m0"][key] for r in rows]))
            entry[f"m7_{key}"] = float(np.mean([r["validation_metrics_m7"][key] for r in rows]))
        entry["m0_f1_macro_sd"] = float(np.std([r["validation_metrics_m0"]["f1_macro"] for r in rows], ddof=1))
        entry["curve_auc_f1_macro"] = float(np.mean([r["intervention_curve_auc"]["f1_macro"]["absolute"] for r in rows]))
        entry["curve_auc_f1_macro_sd"] = float(np.std([r["intervention_curve_auc"]["f1_macro"]["absolute"] for r in rows], ddof=1))
        entry["flips_m7_mean"] = float(np.mean([r["flips_m7"] for r in rows]))
        entry["agreement_with_e_class_mean"] = float(np.mean([r["agreement_with_e_class"] for r in rows]))
        entry["decision_thresholds"] = {s: r["decision_threshold"] for s, r in zip(seeds, rows)}
        summary.append(entry)
    floor = summary[grid.index(0.01)]["m0_f1_macro"] - 0.01
    feasible = [e for e in summary if e["m0_f1_macro"] >= floor - 1e-12]
    best = max(e["curve_auc_f1_macro"] for e in feasible)
    chosen = min((e for e in feasible if best - e["curve_auc_f1_macro"] < 1e-4), key=lambda e: e["lambda_global"])
    for e in summary:
        e["feasible"] = e in feasible
    selection = {"created_from": os.path.relpath(PLAN_PATH, PROJECT_ROOT),
                 "study_plan_sha256": hashlib.sha256(plan_text.encode("utf-8")).hexdigest(),
                 "selection_rule": rule, "constraint_floor_m0_f1_macro": floor,
                 "chosen_lambda_global": chosen["lambda_global"],
                 "chosen_decision_thresholds": chosen["decision_thresholds"],
                 "inference_energy_weights": [1.0, 1.0, chosen["lambda_global"]],
                 "test_read": False, "summary": summary}
    shared._save_json(selection_path, selection)
    _write_report(summary, chosen, floor)
    print(f"\nλ_global được chọn: {chosen['lambda_global']:g} (ngưỡng ràng buộc F1 m0 ≥ {floor:.4f})")
    return selection


def _write_report(summary, chosen, floor):
    lines = ["# Study λ_global suy luận ECBM (validation, 3 seeds)", "",
             f"Ràng buộc: mean F1 m=0 ≥ {floor:.4f}. λ được chọn: **{chosen['lambda_global']:g}**. "
             "Không đọc test. Quy tắc trong [study_plan.json](study_plan.json).", "",
             "| λ_global | F1 m=0 | BAcc m=0 | AUC m=0 | F1 m=7 | Curve AUC F1 | Đổi nhãn m=7 | Trùng E_class | Thỏa ràng buộc |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|"]
    for e in summary:
        mark = "**" if e is chosen else ""
        lines.append(f"| {mark}{e['lambda_global']:g}{mark} | {e['m0_f1_macro']:.4f} ± {e['m0_f1_macro_sd']:.4f} | "
                     f"{100 * e['m0_balanced_accuracy']:.2f}% | {e['m0_roc_auc']:.4f} | {e['m7_f1_macro']:.4f} | "
                     f"{e['curve_auc_f1_macro']:.4f} ± {e['curve_auc_f1_macro_sd']:.4f} | {e['flips_m7_mean']:.1f} | "
                     f"{100 * e['agreement_with_e_class_mean']:.1f}% | {'✓' if e['feasible'] else '✗'} |")
    lines += ["", "Mean ± sample SD qua seeds 42/123/2026; số đổi nhãn là trung bình trên 203 ca. "
              "Best epoch đã chọn với λ_global=0.01, và validation dùng cho cả ngưỡng lẫn λ nên điểm của λ được chọn lạc quan."]
    with open(os.path.join(STUDY_DIR, "report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Study λ_global suy luận ECBM trên validation (không đọc test)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    run_study(args.device, args.overwrite)


if __name__ == "__main__":
    main()
