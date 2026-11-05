# TTA (test-time augmentation) theo results/derm7pt/tta/plan.json: trung bình dự đoán qua các bản lật của ảnh.
# Chỉ suy luận trên validation bằng checkpoint đã có; không train, không đọc test.


import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import src.dataset trước sklearn để tránh xung đột thư viện OpenMP trên một số nền tảng.
from src import dataset as D
from src.dataset import CONCEPT_NAMES, Derm7ptDataset, compute_train_observed_states
from src import runtime
from src.metrics import compute_concept_metrics, compute_metrics
from src.models import BlackBoxClassifier, get_soft_joint_cbm, load_blackbox_state_dict
from src.naming import backbone_from_config
from src.protocol import validate_concept_schema, validate_manifest
from src.selection import select_threshold
from src.transforms import get_transforms

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments import run_blackbox, run_ecbm, run_hard_cbm, run_soft_cbm

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "derm7pt", "tta")
ORIGINAL_MANIFEST = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
D4A_MANIFEST = os.path.join(PROJECT_ROOT, "data", "d4a", "manifest.csv")
MODELS = {
    "blackbox": ("efficientnet_b0/linear_e50_headlr0.001", ORIGINAL_MANIFEST),
    "hard_cbm": ("hard_cbm/efficientnet_b0_mlp128_e50_headlr0.001_d4a", D4A_MANIFEST),
    "soft_cbm": ("soft_cbm/efficientnet_b0_mlp128_e50_headlr0.001_d4a", D4A_MANIFEST),
    "ecbm": ("ecbm/efficientnet_b0_unified_exact_e50_d4a", D4A_MANIFEST),
}
SEEDS = (42, 123, 2026)
DIAGNOSIS = ("accuracy", "balanced_accuracy", "f1_macro", "roc_auc")

# Các bản ảnh trên tensor [B, C, H, W]: flip4 là phép lật đã dùng khi train; dihedral8 thêm chuyển vị.
FLIP4 = [lambda x: x, lambda x: x.flip(3), lambda x: x.flip(2), lambda x: x.flip(2).flip(3)]
DIHEDRAL8 = FLIP4 + [lambda x, f=f: f(x.transpose(2, 3)) for f in FLIP4]
VIEW_SETS = {"none": FLIP4[:1], "flip4": FLIP4, "dihedral8": DIHEDRAL8}


def load_model(kind, checkpoint, device):
    config = checkpoint["config"]
    if kind == "blackbox":
        model = BlackBoxClassifier(num_classes=2, pretrained=False, head=run_blackbox._head_preset(config),
                                   backbone=backbone_from_config(config))
        load_blackbox_state_dict(model, checkpoint["model_state_dict"])
    elif kind == "hard_cbm":
        model = run_hard_cbm._build_model("straight_through", CONCEPT_NAMES, D.CONCEPT_NUM_CLASSES, pretrained=False,
                                          diagnosis_head=run_hard_cbm._head_preset(config))
        model.load_state_dict(checkpoint["model_state_dict"])
    elif kind == "soft_cbm":
        model = get_soft_joint_cbm(CONCEPT_NAMES, D.CONCEPT_NUM_CLASSES, num_classes=2, pretrained=False,
                                   diagnosis_head=run_soft_cbm._head_preset(config))
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = run_ecbm._build_model(config, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


# Suy luận chính xác của ECBM từ năng lượng đã trung bình (cùng công thức với src.ecbm.exact_inference).
def ecbm_exact_from_energies(model, e_class, e_concepts, weights):
    w_class, w_concept, w_global = weights
    profiles = model.profiles
    e_global = model.global_class_energies(profiles)
    concept_sum = sum(energy[:, profiles[:, group]] for group, energy in enumerate(e_concepts))
    log_joint = -(w_class * e_class[:, None, :] + w_concept * concept_sum[:, :, None] + w_global * e_global[None])
    log_joint = log_joint - torch.logsumexp(log_joint.flatten(1), dim=1)[:, None, None]
    class_probabilities = torch.logsumexp(log_joint, dim=1).exp()
    profile_probabilities = torch.logsumexp(log_joint, dim=2).exp()
    concepts = [torch.zeros(len(e_class), count, device=e_class.device).index_add_(1, profiles[:, group],
                                                                                    profile_probabilities)
                for group, count in enumerate(model.state_counts)]
    return class_probabilities[:, 1], concepts


@torch.no_grad()
def predict(kind, model, images, views, config):
    if kind == "blackbox":
        return torch.stack([model(view(images)).softmax(-1)[:, 1] for view in views]).mean(0), None
    if kind == "ecbm":
        e_class, e_concepts = 0, None
        for view in views:
            features = model.extract_features(view(images))
            current = model.concept_energies(features)
            e_class = e_class + model.class_energies(features) / len(views)
            e_concepts = ([e / len(views) for e in current] if e_concepts is None
                          else [a + e / len(views) for a, e in zip(e_concepts, current)])
        return ecbm_exact_from_energies(model, e_class, e_concepts, config["inference_energy_weights"])
    # CBM: trung bình xác suất concept trước, rồi mới qua head chẩn đoán để giải thích khớp chẩn đoán.
    probabilities = None
    for view in views:
        features = model.extract_features(view(images))
        current = [model.concept_heads[name](features).softmax(-1) for name in CONCEPT_NAMES]
        probabilities = ([p / len(views) for p in current] if probabilities is None
                         else [a + p / len(views) for a, p in zip(probabilities, current)])
    if kind == "hard_cbm":
        bottleneck = torch.cat([torch.nn.functional.one_hot(p.argmax(-1), p.shape[-1]).float() for p in probabilities], -1)
    else:
        bottleneck = torch.cat(probabilities, -1)
    return model.diagnosis_head(bottleneck).softmax(-1)[:, 1], probabilities


def evaluate(labels, scores, concepts=None, truth=None, counts=None, observed=None):
    threshold, _ = select_threshold(labels, scores, "f1_macro")
    metrics = compute_metrics(labels, (scores >= threshold).astype(int), scores)
    row = {k: float(metrics[k]) for k in DIAGNOSIS}
    row["threshold"] = float(threshold)
    if concepts is not None:
        pred = {c: concepts[c].argmax(1) for c in CONCEPT_NAMES}
        cm = compute_concept_metrics(truth, pred, counts, train_observed_states=observed)
        row["concept_accuracy"] = cm["overall_concept_accuracy"]
        row["concept_f1_macro"] = cm["overall_f1_macro_train_observed"]
        row["per_concept_accuracy"] = {c: cm["per_concept"][c]["accuracy"] for c in CONCEPT_NAMES}
    return row


def run_one(kind, folder, manifest, seed, device):
    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "derm7pt", folder, f"seed{seed}.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    validate_manifest(checkpoint, manifest)
    schema = validate_concept_schema(checkpoint, manifest) if kind != "blackbox" else None
    config = checkpoint["config"]
    model = load_model(kind, checkpoint, device)
    dataset = Derm7ptDataset(manifest_path=manifest, project_root=PROJECT_ROOT, split="valid",
                             transform=get_transforms("valid", target_size=config["target_size"], augment=False))
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    collected = {name: {"scores": [], "concepts": {c: [] for c in CONCEPT_NAMES}} for name in VIEW_SETS}
    labels, truth, cases = [], {c: [] for c in CONCEPT_NAMES}, []
    for batch in loader:
        images = batch["image"].to(device)
        labels.extend(batch["label"].tolist())
        cases.extend(int(c) for c in batch["meta"]["case_num"])
        for g, c in enumerate(CONCEPT_NAMES):
            truth[c].extend(batch["concept_indices"][:, g].tolist())
        for name, views in VIEW_SETS.items():
            scores, concepts = predict(kind, model, images, views, config)
            collected[name]["scores"].append(scores.cpu().numpy())
            if concepts is not None:
                for c, p in zip(CONCEPT_NAMES, concepts):
                    collected[name]["concepts"][c].append(p.cpu().numpy())
    labels = np.array(labels)
    # Kiểm tra: nhánh 1 bản phải tái lập xác suất validation đã lưu.
    stored = {r["case_num"]: r["y_prob"] for r in
              runtime.load_json(os.path.join(PROJECT_ROOT, "results", "derm7pt", folder, f"seed{seed}.json"))["validation_predictions"]}
    single = np.concatenate(collected["none"]["scores"])
    gap = float(np.max(np.abs(single - np.array([stored[c] for c in cases]))))
    if gap > 1e-3:
        raise ValueError(f"{kind} seed{seed}: single-view scores differ from stored validation ({gap:.2e})")
    out = {"single_view_max_abs_gap": gap}
    for name in VIEW_SETS:
        scores = np.concatenate(collected[name]["scores"])
        if kind == "blackbox":
            out[name] = evaluate(labels, scores)
        else:
            counts = schema["concept_num_classes"]
            frame = pd.read_csv(manifest)
            observed = compute_train_observed_states(frame[frame["split"] == "train"], schema["label_mapping"])
            concepts = {c: np.concatenate(collected[name]["concepts"][c]) for c in CONCEPT_NAMES}
            out[name] = evaluate(labels, scores, concepts, {c: np.array(truth[c]) for c in CONCEPT_NAMES},
                                 counts, observed)
    return out


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)), "per_seed": values.tolist()}


def decide(runs, has_concepts):
    m = lambda arm, key: float(np.mean([r[arm][key] for r in runs]))
    wins = lambda key: sum(r["flip4"][key] > r["none"][key] for r in runs)
    auc_up = m("flip4", "roc_auc") >= m("none", "roc_auc") + 0.005 and wins("roc_auc") >= 2
    concept_up = (has_concepts and m("flip4", "concept_accuracy") >= m("none", "concept_accuracy") + 0.005
                  and wins("concept_accuracy") >= 2)
    guard_keys = ("accuracy", "balanced_accuracy", "f1_macro") + (("concept_f1_macro",) if has_concepts else ())
    guard = all(m("flip4", k) >= m("none", k) - 0.01 for k in guard_keys)
    return {"auc_up": bool(auc_up), "concept_up": bool(concept_up), "guard": bool(guard),
            "use_flip4_tta": bool((auc_up or concept_up) and guard)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = runtime.get_device(args.device)
    summary = {"plan": "plan.json", "test_read": False, "models": {}}
    for kind, (folder, manifest) in MODELS.items():
        runs = [run_one(kind, folder, manifest, seed, device) for seed in SEEDS]
        has_concepts = kind != "blackbox"
        keys = DIAGNOSIS + (("concept_accuracy", "concept_f1_macro") if has_concepts else ())
        block = {name: {k: mean_sd([r[name][k] for r in runs]) for k in keys} for name in VIEW_SETS}
        if has_concepts:
            for name in VIEW_SETS:
                block[name]["per_concept_accuracy"] = {c: mean_sd([r[name]["per_concept_accuracy"][c] for r in runs])
                                                       for c in CONCEPT_NAMES}
        block["single_view_max_abs_gap"] = max(r["single_view_max_abs_gap"] for r in runs)
        block["decision"] = decide(runs, has_concepts)
        summary["models"][kind] = block
        for name in VIEW_SETS:
            print(f"{kind:9s} {name:9s} " + "  ".join(f"{k}={block[name][k]['mean']:.3f}±{block[name][k]['sd']:.3f}"
                                                       for k in keys))
        print(f"{kind:9s} decision {json.dumps(block['decision'])}  gap={block['single_view_max_abs_gap']:.1e}")
    D.activate_concept_schema(D.DEFAULT_CONCEPT_NUM_CLASSES)
    runtime.save_json(os.path.join(OUT_DIR, "summary.json"), summary)


if __name__ == "__main__":
    main()
