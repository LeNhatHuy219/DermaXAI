# Phân tích hạn chế của ECBM (cohort unified exact 50 epochs) cho chương "Hạn chế của ECBM".
# Tính mới chỉ trên validation (nạp checkpoint đã đóng băng); phần test chỉ đọc lại kết quả đã lưu, không đọc lại ảnh test.
# Ghi numbers.json và các hình fig*.png/svg vào results/derm7pt/ecbm/limitations/.


import json
import math
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import src.dataset trước sklearn để tránh xung đột thư viện OpenMP trên một số nền tảng.
from src.dataset import CONCEPT_NAMES

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from experiments import run_soft_cbm as shared
from experiments.run_ecbm_lambda_study import exact_outputs, load_seed
from experiments.run_ecbm_modes import MANIFEST_PATH, _plan
from src.ecbm import exact_inference
from src.metrics import compute_metrics
from src.selection import select_threshold

RESULTS = os.path.join(PROJECT_ROOT, "results", "derm7pt")
OUTPUT_DIR = os.path.join(RESULTS, "ecbm", "limitations")
SEEDS = ("42", "123", "2026")
CBM_COHORTS = {"hard_cbm": "hard_cbm/efficientnet_b0_mlp128_e50_headlr0.001",
               "soft_cbm": "soft_cbm/efficientnet_b0_mlp128_e50_headlr0.001"}
ECBM_MODES = ("ecbm_full", "ecbm_paper", "ecbm_concept_only")
LABELS = {"ecbm_full": "ECBM λ=1 (chính)", "ecbm_paper": "ECBM λ paper", "ecbm_concept_only": "ECBM chỉ concept",
          "hard_cbm": "Hard CBM", "soft_cbm": "Soft CBM"}
# Màu cố định theo mô hình ở mọi hình (thứ tự palette đã kiểm tra bằng validate_palette.js trên nền trắng).
COLORS = {"ecbm_full": "#2a78d6", "ecbm_paper": "#eb6834", "ecbm_concept_only": "#1baf7a",
          "hard_cbm": "#eda100", "soft_cbm": "#e87ba4"}
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GROUP_LABELS = {"pigment_network": "Pigment network", "streaks": "Streaks", "pigmentation": "Pigmentation",
                "regression_structures": "Regression", "dots_and_globules": "Dots/globules",
                "blue_whitish_veil": "Blue-whitish veil", "vascular_structures": "Vascular"}


def _load(path):
    return json.load(open(path, encoding="utf-8"))


def _summary(values):
    values = [float(v) for v in values]
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)), "values": values}


# ---------- Tính trên validation ----------

# Tách log-odds của exact inference thành phần E_class và phần đi qua concept (E_concept + E_global).
def _decomposition(model, features, lambda_global):
    e_class = model.class_energies(features)
    concept_sum = sum(e[:, model.profiles[:, g]] for g, e in enumerate(model.concept_energies(features)))
    e_global = model.global_class_energies(model.profiles)
    log_mass = torch.logsumexp(-concept_sum[:, :, None] - lambda_global * e_global[None], dim=1)
    class_term = (e_class[:, 0] - e_class[:, 1]).cpu().numpy()
    concept_term = (log_mass[:, 1] - log_mass[:, 0]).cpu().numpy()
    p = exact_inference(model, features, (1.0, 1.0, lambda_global))["class_probabilities"].cpu().numpy()
    exact_log_odds = np.log(p[:, 1]) - np.log(p[:, 0])
    if np.max(np.abs(exact_log_odds - (class_term + concept_term))) > 1e-3:
        raise ValueError("Log-odds decomposition does not reproduce exact inference")
    return class_term, concept_term


def _training_corruption_rate(num_train=413, batch_size=16, sample_fraction=0.2, concept_fraction=0.2, groups=7):
    # perturb_concepts đổi ceil(B·0.2) ca × ceil(7·0.2) nhóm mỗi batch sang trạng thái khác (drop_last=False).
    sizes = [batch_size] * (num_train // batch_size) + ([num_train % batch_size] if num_train % batch_size else [])
    cells = sum(math.ceil(b * sample_fraction) for b in sizes) * math.ceil(groups * concept_fraction)
    return cells / (num_train * groups)


def validation_analysis(device):
    _, _, entries = _plan()
    out = {"decomposition": {}, "e_global": {}, "concept_accuracy_by_inference": {}}
    points = {0.01: ([], []), 1.0: ([], [])}
    for seed, entry in entries.items():
        data = load_seed(entry, MANIFEST_PATH, device, split="valid")
        model, features, labels, concepts = data["model"], data["features"], data["labels"], data["concepts"]
        with torch.no_grad():
            for lambda_global in (0.01, 1.0):
                class_term, concept_term = _decomposition(model, features, lambda_global)
                points[lambda_global][0].extend(class_term.tolist())
                points[lambda_global][1].extend(concept_term.tolist())
                disagree = np.sign(class_term) != np.sign(concept_term)
                total = class_term + concept_term
                out["decomposition"].setdefault(str(lambda_global), {})[seed] = {
                    "mean_abs_class_term": float(np.mean(np.abs(class_term))),
                    "mean_abs_concept_term": float(np.mean(np.abs(concept_term))),
                    "fraction_concept_larger": float(np.mean(np.abs(concept_term) > np.abs(class_term))),
                    "fraction_terms_disagree": float(np.mean(disagree)),
                    "fraction_concept_wins_when_disagree": (float(np.mean(np.sign(total[disagree]) == np.sign(concept_term[disagree])))
                                                            if disagree.any() else None)}
            e_concepts = model.concept_energies(features)
            predicted = torch.stack([e.argmin(-1) for e in e_concepts], dim=1)
            scores = {"e_global_gt_concepts": (-model.global_class_energies(concepts)).softmax(-1)[:, 1],
                      "e_global_predicted_concepts": (-model.global_class_energies(predicted)).softmax(-1)[:, 1],
                      "e_class": (-model.class_energies(features)).softmax(-1)[:, 1]}
            for name, score in scores.items():
                score = score.cpu().numpy().astype(float)
                threshold, f1 = select_threshold(labels, score, "f1_macro")
                out["e_global"].setdefault(name, {})[seed] = {"roc_auc": float(roc_auc_score(labels, score)),
                                                              "f1_macro_at_validation_threshold": float(f1)}
            out["e_global"].setdefault("predicted_concept_error_rate", {})[seed] = float(1 - (predicted == concepts).float().mean())
            truth = concepts.cpu().numpy()
            accuracy = {"e_concept_argmax": float((predicted == concepts).float().mean())}
            for name, weights in (("ecbm_paper", (1.0, 1.0, 0.01)), ("ecbm_full", (1.0, 1.0, 1.0)),
                                  ("ecbm_concept_only", (0.0, 1.0, 1.0))):
                _, marginals = exact_outputs(model, features, weights)
                accuracy[name] = float(np.mean([np.mean(p.argmax(1) == truth[:, g]) for g, p in enumerate(marginals)]))
            out["concept_accuracy_by_inference"][seed] = accuracy
    out["training_concept_corruption_rate"] = _training_corruption_rate()
    return out, points


# ---------- Đọc kết quả test đã lưu ----------

def test_analysis():
    ecbm = {s: _load(os.path.join(RESULTS, "ecbm", "test_modes", f"seed{s}.json")) for s in SEEDS}
    out = {"curves": {}, "operating_point": {}, "per_group_concept_accuracy": {}, "per_seed": {}, "agreement_with_e_class": {}}
    for mode in ECBM_MODES:
        blocks = [ecbm[s]["modes"][mode] for s in SEEDS]
        out["curves"][mode] = [list(map(float, b["curve"]["f1_macro"])) for b in blocks]
        ops, seeds = [], []
        for b in blocks:
            p = b["test_predictions"]
            y = np.array([r["y_true"] for r in p]); before = np.array([r["y_prob"] for r in p])
            after = np.array([r["y_prob_full_intervention"] for r in p]); t = b["decision_threshold"]
            m0, m7 = compute_metrics(y, before >= t, before), compute_metrics(y, after >= t, after)
            ops.append({"sensitivity": [m0["sensitivity"], m7["sensitivity"]], "specificity": [m0["specificity"], m7["specificity"]],
                        "roc_auc": [m0["roc_auc"], m7["roc_auc"]],
                        "melanoma_prob_decreased": float(np.mean(after[y == 1] < before[y == 1]))})
            seeds.append({"f1_macro": m0["f1_macro"], "concept_accuracy": b["test_concept_metrics"]["overall_concept_accuracy"],
                          "decision_threshold": t})
        out["operating_point"][mode] = ops
        out["per_seed"][mode] = seeds
        out["per_group_concept_accuracy"][mode] = {g: float(np.mean([b["test_concept_metrics"]["per_concept"][g]["accuracy"]
                                                                     for b in blocks])) for g in CONCEPT_NAMES}
        out["agreement_with_e_class"][mode] = _summary([b["agreement_with_e_class"] for b in blocks])
    for model, cohort in CBM_COHORTS.items():
        runs = [_load(os.path.join(RESULTS, cohort, f"seed{s}.json")) for s in SEEDS]
        summary = _load(os.path.join(RESULTS, cohort, "intervention", "summary.json"))
        curve = summary["curve"]
        out["curves"][model] = [[float(curve[m]["f1_macro"]["values"][i]) for m in range(len(curve))] for i in range(len(SEEDS))]
        out["operating_point"][model] = [{key: [curve[0][key]["values"][i], curve[-1][key]["values"][i]]
                                          for key in ("sensitivity", "specificity", "roc_auc")} for i in range(len(SEEDS))]
        out["per_seed"][model] = [{"f1_macro": r["test_metrics"]["f1_macro"],
                                   "concept_accuracy": r["test_concept_metrics"]["overall_concept_accuracy"],
                                   "decision_threshold": r["decision_threshold"]} for r in runs]
        out["per_group_concept_accuracy"][model] = {g: float(np.mean([r["test_concept_metrics"]["per_concept"][g]["accuracy"]
                                                                      for r in runs])) for g in CONCEPT_NAMES}
    selection = _load(os.path.join(RESULTS, "ecbm", "lambda_global_study", "frozen_selection.json"))
    out["lambda_sweep_validation"] = [{k: e[k] for k in ("lambda_global", "m0_f1_macro", "m7_f1_macro", "curve_auc_f1_macro",
                                                          "flips_m7_mean")} for e in selection["summary"]]
    return out


# ---------- Hình ----------

def _style(plt):
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
                         "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
                         "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
                         "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
                         "axes.titlelocation": "left"})


def _save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.{ext}"), dpi=200, bbox_inches="tight")


def figures(val, points, test):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    dot = dict(s=34, edgecolors="white", linewidths=1.2, zorder=3)

    # Hình 1: tách log-odds (validation) — phần concept gần 0 ở λ paper, vẫn nhỏ hơn E_class ở λ=1.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharex=True, sharey=True)
    lim = max(np.abs(np.array(v)).max() for pair in points.values() for v in pair) * 1.05
    for ax, lam, title in zip(axes, (0.01, 1.0), ("λ_global = 0.01 (paper)", "λ_global = 1 (đã chọn)")):
        x, y = np.array(points[lam][0]), np.array(points[lam][1])
        ax.plot([-lim, lim], [-lim, lim], color=AXIS, lw=0.8, zorder=1)
        ax.plot([-lim, lim], [lim, -lim], color=AXIS, lw=0.8, zorder=1)
        ax.scatter(x, y, s=14, color=COLORS["ecbm_full"], alpha=0.55, edgecolors="none", zorder=2)
        share = np.mean(np.abs(y) > np.abs(x)) * 100
        ax.set_title(title)
        ax.text(0.03, 0.96, f"Phần concept lớn hơn E_class: {share:.1f}% ca", transform=ax.transAxes, va="top",
                color=INK_2, fontsize=8)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("Đóng góp của E_class vào log-odds melanoma")
    axes[0].set_ylabel("Đóng góp qua concept (E_concept + E_global)")
    fig.suptitle("Hình 1. Ai quyết định chẩn đoán? (validation, 3 seeds, 609 điểm)", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(); _save(fig, "fig1_logodds_decomposition"); plt.close(fig)

    # Hình 2: quét λ_global trên validation.
    sweep = test["lambda_sweep_validation"]
    lam = [e["lambda_global"] for e in sweep]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    ax.plot(lam, [e["m0_f1_macro"] for e in sweep], color=MUTED, lw=1.5, marker="o", ms=5, mec="white", mew=1.2, label="Không sửa concept")
    ax.plot(lam, [e["m7_f1_macro"] for e in sweep], color=COLORS["ecbm_full"], lw=1.5, marker="o", ms=5, mec="white", mew=1.2,
            label="Sửa đủ 7 concept")
    ax.set_xscale("log"); ax.set_xlabel("λ_global lúc suy luận"); ax.set_ylabel("Macro F1 (validation)")
    ax.set_title("Macro F1 trước và sau can thiệp"); ax.legend(loc="lower left", fontsize=8)
    ax = axes[1]
    ax.plot(lam, [e["flips_m7_mean"] for e in sweep], color=COLORS["ecbm_full"], lw=1.5, marker="o", ms=5, mec="white", mew=1.2)
    ax.set_xscale("log"); ax.set_xlabel("λ_global lúc suy luận"); ax.set_ylabel("Số ca đổi nhãn (trên 203)")
    ax.set_title("Số ca đổi nhãn khi sửa đủ 7 concept")
    for a in axes:
        a.set_xticks(lam); a.set_xticklabels([f"{v:g}" for v in lam]); a.minorticks_off()
    fig.suptitle("Hình 2. Ảnh hưởng của λ_global (validation, trung bình 3 seeds)", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(); _save(fig, "fig2_lambda_sweep"); plt.close(fig)

    # Hình 3: đường can thiệp trên test (đã lưu).
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for model in ("ecbm_full", "ecbm_paper", "ecbm_concept_only", "hard_cbm", "soft_cbm"):
        mean = np.mean(test["curves"][model], axis=0)
        ax.plot(range(8), mean, color=COLORS[model], lw=1.5, marker="o", ms=5, mec="white", mew=1.2, label=LABELS[model])
    ax.set_xlabel("Số nhóm concept được sửa về nhãn đúng (m)"); ax.set_ylabel("Macro F1 (test)")
    ax.set_xticks(range(8)); ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Hình 3. Đường can thiệp concept trên test (trung bình 3 seeds)")
    fig.tight_layout(); _save(fig, "fig3_intervention_curves_test"); plt.close(fig)

    # Hình 4: sensitivity/specificity trước và sau khi sửa đủ 7 concept (test, đã lưu).
    order = ["ecbm_full", "ecbm_paper", "ecbm_concept_only", "hard_cbm", "soft_cbm"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, key, title in zip(axes, ("sensitivity", "specificity"), ("Sensitivity (bắt melanoma)", "Specificity")):
        for i, model in enumerate(order):
            before = np.mean([o[key][0] for o in test["operating_point"][model]]) * 100
            after = np.mean([o[key][1] for o in test["operating_point"][model]]) * 100
            ax.plot([before, after], [i, i], color=AXIS, lw=1.2, zorder=1)
            ax.scatter([before], [i], color=MUTED, **dot)
            ax.scatter([after], [i], color=COLORS[model], **dot)
        ax.set_title(title); ax.set_xlabel("% (test)")
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels([LABELS[m] for m in order]); axes[0].invert_yaxis()
    handles = [axes[0].scatter([], [], color=MUTED, s=34, label="Trước can thiệp"),
               axes[0].scatter([], [], color=INK_2, s=34, label="Sau khi sửa đủ 7 concept (chấm mang màu của mô hình)")]
    fig.suptitle("Hình 4. Can thiệp tăng mạnh specificity, sensitivity gần như không đổi hoặc giảm (ngưỡng đóng băng)", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=2, fontsize=8)
    _save(fig, "fig4_operating_point_test"); plt.close(fig)

    # Hình 5: concept accuracy từng nhóm (test, đã lưu).
    models = ["ecbm_full", "ecbm_paper", "hard_cbm", "soft_cbm"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for j, model in enumerate(models):
        values = [test["per_group_concept_accuracy"][model][g] * 100 for g in CONCEPT_NAMES]
        ax.scatter(values, np.arange(len(CONCEPT_NAMES)) + (j - 1.5) * 0.12, color=COLORS[model], label=LABELS[model], **dot)
    ax.set_yticks(range(len(CONCEPT_NAMES))); ax.set_yticklabels([GROUP_LABELS[g] for g in CONCEPT_NAMES]); ax.invert_yaxis()
    ax.set_xlabel("Concept accuracy % (test, trung bình 3 seeds)")
    ax.set_title("Hình 5. Độ chính xác concept theo từng nhóm")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(*ax.get_legend_handles_labels(), loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=4, fontsize=8)
    _save(fig, "fig5_concept_accuracy_by_group_test"); plt.close(fig)

    # Hình 6: dao động giữa seeds (test, đã lưu).
    order = ["ecbm_full", "ecbm_paper", "ecbm_concept_only", "hard_cbm", "soft_cbm"]
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.9), sharey=True)
    for ax, key, title, scale in zip(axes, ("f1_macro", "concept_accuracy", "decision_threshold"),
                                     ("Macro F1", "Concept accuracy %", "Ngưỡng quyết định"), (1, 100, 1)):
        for i, model in enumerate(order):
            values = [s[key] * scale for s in test["per_seed"][model]]
            ax.scatter(values, [i] * len(values), color=COLORS[model], **dot)
            ax.plot([np.mean(values)] * 2, [i - 0.25, i + 0.25], color=INK, lw=1.2, zorder=4)
        ax.set_title(title)
        ax.set_yticks(range(len(order))); ax.set_yticklabels([LABELS[m] for m in order]); ax.invert_yaxis()
    fig.suptitle("Hình 6. Dao động giữa 3 seeds trên test (chấm: từng seed, vạch: trung bình)", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(); _save(fig, "fig6_seed_variability_test"); plt.close(fig)

    # Hình 7: E_global với concept đúng so với concept dự đoán (validation).
    names = [("e_global_gt_concepts", "E_global + concept đúng (GT)"),
             ("e_global_predicted_concepts", "E_global + concept dự đoán"), ("e_class", "E_class (nhánh ảnh)")]
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    for i, (key, label) in enumerate(names):
        values = [val["e_global"][key][s]["roc_auc"] for s in SEEDS]
        ax.scatter(values, [i] * 3, color=COLORS["ecbm_full"], **dot)
        ax.plot([np.mean(values)] * 2, [i - 0.25, i + 0.25], color=INK, lw=1.2, zorder=4)
    ax.set_yticks(range(len(names))); ax.set_yticklabels([n[1] for n in names]); ax.invert_yaxis()
    ax.set_xlabel("ROC-AUC chẩn đoán (validation; chấm: từng seed, vạch: trung bình)")
    ax.set_title("Hình 7. E_global mất sức khi nhận concept dự đoán thay vì concept đúng")
    fig.tight_layout(); _save(fig, "fig7_e_global_gt_vs_predicted"); plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = shared.get_device("auto")
    val, points = validation_analysis(device)
    test = test_analysis()
    shared._save_json(os.path.join(OUTPUT_DIR, "numbers.json"),
                      {"note": "validation: tính mới từ checkpoint đã đóng băng; test: đọc lại kết quả đã lưu, không đọc lại ảnh test",
                       "validation": val, "test_saved": test})
    figures(val, points, test)
    print(f"Đã ghi số liệu và hình vào {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
