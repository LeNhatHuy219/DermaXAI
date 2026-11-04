# Pilot feasibility của đề xuất đồng nhất huấn luyện và suy luận cho categorical ECBM (bản 2, mục 5–8).
# Chỉ đọc train/validation; không có stage nào nạp tập test.
# --stage gap: xuất độ chênh train–validation của các nhánh cục bộ ở ba checkpoint ECBM chính.
# --stage folds: chia 413 ca train thành 5 fold phân tầng theo chẩn đoán, ghi manifest cho từng fold.
# --stage base: train 5 fold model và 1 model trên đủ 413 ca, công thức unified, dừng cố định ở epoch 50.
# --stage cache: lưu local energies OOF cho train, trong mẫu cho train và cho validation (model đủ 413 ca).
# --stage heads: học lại global head cho A0-cal, A1, A2-IS, A2-OOF từ cùng khởi tạo A0;
#   A3-IS, A3-OOF, A3-cal (mixed masks) là thăm dò hậu kiểm theo criteria_a3.json.
# --stage evaluate: chấm validation từng arm: tự động, concept, đường can thiệp 128 tổ hợp.
# --stage check: kiểm tra số học và leakage của pipeline.
# --stage summary: gộp validation các seed và đối chiếu criteria.json đã khóa trước.


import argparse
import itertools
import json
import os
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import src.dataset trước sklearn để tránh xung đột thư viện OpenMP trên một số nền tảng.
from src.dataset import (CONCEPT_NAMES, CONCEPT_NUM_CLASSES, Derm7ptDataset, compute_concept_statistics,
                         get_dataloaders, compute_train_observed_states)
from src import runtime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from experiments import run_ecbm
from src.ecbm import exact_inference, perturb_concepts
from src.metrics import compute_concept_metrics, compute_metrics
from src.protocol import load_concept_schema
from src.selection import select_threshold
from src.transforms import get_transforms

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
PLAN_PATH = os.path.join(PROJECT_ROOT, "results", "derm7pt", "test_plan.json")
NUM_FOLDS = 5
STATE_COUNTS = [CONCEPT_NUM_CLASSES[name] for name in CONCEPT_NAMES]
PROFILES = torch.tensor(list(itertools.product(*[range(k) for k in STATE_COUNTS])), dtype=torch.long)
# Chỉ số profile theo thứ tự itertools.product: nhóm cuối thay đổi nhanh nhất.
STRIDES = torch.tensor([int(np.prod(STATE_COUNTS[g + 1:])) for g in range(len(STATE_COUNTS))], dtype=torch.long)
JOINT_WEIGHTS = (1.0, 1.0, 1.0)
HEAD = {"epochs": 20, "batch_size": 16, "lr": 1e-3, "weight_decay": 0.01}
ARMS = ("A0", "A0-cal", "A1", "A2-IS", "A2-OOF", "A3-IS", "A3-OOF", "A3-cal")
HEAD_ARMS = {"A1": "oof", "A2-IS": "in_sample", "A2-OOF": "oof", "A3-IS": "in_sample", "A3-OOF": "oof"}
CALIBRATION_ARMS = {"A0-cal": 0, "A3-cal": 4}  # số mask cố định mỗi ca; 0 = S luôn rỗng
CALIBRATION_MASK_SEED_OFFSET = 3
THRESHOLD_METRIC = "f1_macro"


def _results_dir(seed):
    return os.path.join(PROJECT_ROOT, "results", "derm7pt", "ecbm", "align_pilot", f"seed{seed}")


def _checkpoint_dir(seed):
    return os.path.join(PROJECT_ROOT, "checkpoints", "derm7pt", "ecbm", "align_pilot", f"seed{seed}")


def _fold_manifest(seed, fold):
    return os.path.join(_checkpoint_dir(seed), "manifests", f"fold{fold}.csv")


def _base_path(seed, fold=None):
    return os.path.join(_checkpoint_dir(seed), "full.pth" if fold is None else f"fold{fold}.pth")


def _config():
    return {"hidden_dim": run_ecbm.DEFAULT_HIDDEN_DIM, "dropout": 0.2}


def _load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = run_ecbm._build_model(_config(), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval(), checkpoint


# ---------- Độ chênh train–validation của checkpoint ECBM chính ----------

def _local_outputs(model, manifest_path, split, device, label_mapping):
    dataset = Derm7ptDataset(manifest_path=manifest_path, project_root=PROJECT_ROOT, split=split,
                             label_mapping=label_mapping,
                             transform=get_transforms("valid", target_size=224, augment=False))
    e_class, e_concepts, labels, concepts = [], [[] for _ in CONCEPT_NAMES], [], []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0):
            features = model.extract_features(batch["image"].to(device))
            e_class.append(model.class_energies(features).cpu())
            for g, energy in enumerate(model.concept_energies(features)):
                e_concepts[g].append(energy.cpu())
            labels.append(batch["label"])
            concepts.append(batch["concept_indices"])
    return {"e_class": torch.cat(e_class), "e_concepts": [torch.cat(e) for e in e_concepts],
            "labels": torch.cat(labels).long(), "concepts": torch.cat(concepts).long(),
            "case_num": dataset.df["case_num"].astype(int).tolist()}


def _local_summary(out):
    log_p = F.log_softmax(-out["e_class"], dim=1)
    pred = log_p.argmax(1).numpy()
    metrics = compute_metrics(out["labels"].numpy(), pred, log_p[:, 1].exp().numpy())
    concept_nll = [float(F.cross_entropy(-e, out["concepts"][:, g])) for g, e in enumerate(out["e_concepts"])]
    concept_acc = [float(((-e).argmax(1) == out["concepts"][:, g]).float().mean())
                   for g, e in enumerate(out["e_concepts"])]
    return {"e_class_nll": float(-log_p.gather(1, out["labels"][:, None]).mean()),
            "e_class_f1_macro_argmax": metrics["f1_macro"], "e_class_mean_max_probability": float(log_p.exp().max(1).values.mean()),
            "local_concept_accuracy": float(np.mean(concept_acc)), "local_concept_nll_sum": float(np.sum(concept_nll))}


def gap(device):
    device = runtime.get_device(device)
    plan = json.load(open(PLAN_PATH, encoding="utf-8"))["models"]["ecbm"]
    folder = plan["cohort"].replace("results/", "checkpoints/")
    mapping = load_concept_schema(MANIFEST_PATH)["label_mapping"]
    rows = {}
    for seed, expected in plan["checkpoints"].items():
        path = os.path.join(PROJECT_ROOT, folder, f"seed{seed}.pth")
        if runtime.file_sha256(path) != expected:
            raise ValueError(f"{path} khác hash trong test_plan.json")
        model, checkpoint = _load_model(path, device)
        rows[seed] = {"checkpoint": os.path.relpath(path, PROJECT_ROOT), "sha256": expected, "epoch": int(checkpoint["epoch"]),
                      **{split: _local_summary(_local_outputs(model, MANIFEST_PATH, split, device, mapping))
                         for split in ("train", "valid")}}
        print(f"seed {seed}: " + " || ".join(
            f"{split}: E_class NLL {rows[seed][split]['e_class_nll']:.3f} F1 {rows[seed][split]['e_class_f1_macro_argmax']:.3f}"
            f" | concept acc {100 * rows[seed][split]['local_concept_accuracy']:.1f}%" for split in ("train", "valid")))
    payload = {"definitions": {
        "preprocessing": "get_transforms('valid', 224, augment=False), model.eval()",
        "e_class_nll": "mean_i -log softmax(-E_class(x_i,·))[y_i], không trọng số lớp",
        "e_class_f1_macro_argmax": "Macro F1 của argmax softmax(-E_class), ngưỡng 0.5",
        "local_concept_accuracy": "trung bình 7 nhóm của accuracy argmax softmax(-E_concept_g)",
        "local_concept_nll_sum": "tổng 7 nhóm của mean CE(-E_concept_g), không trọng số trạng thái"},
        "checkpoints": rows}
    os.makedirs(os.path.dirname(_results_dir(0)), exist_ok=True)
    runtime.save_json(os.path.join(os.path.dirname(_results_dir(0)), "train_valid_gap.json"), payload)


# ---------- Fold, base model và cache ----------

def folds(seed):
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"is_inconsistent_profile": bool})
    train = manifest[manifest["split"] == "train"].reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=seed)
    assignment = np.empty(len(train), dtype=int)
    for fold, (_, held) in enumerate(splitter.split(train, train["diagnosis_binary"])):
        assignment[held] = fold
    os.makedirs(os.path.dirname(_fold_manifest(seed, 0)), exist_ok=True)
    shutil.copyfile(os.path.join(PROJECT_ROOT, "data", "label_mapping.json"),
                    os.path.join(os.path.dirname(_fold_manifest(seed, 0)), "label_mapping.json"))
    support = {}
    for fold in range(NUM_FOLDS):
        frame = manifest[manifest["split"].isin(["train", "valid"])].copy()
        is_train = frame["split"] == "train"
        fold_of = dict(zip(train["case_num"], assignment))
        frame.loc[is_train & (frame["case_num"].map(fold_of) == fold), "split"] = "heldout"
        frame.to_csv(_fold_manifest(seed, fold), index=False)
        fit = frame[frame["split"] == "train"]
        support[fold] = {name: {state: int(count) for state, count in fit[name].value_counts().items()}
                         for name in CONCEPT_NAMES}
    os.makedirs(_results_dir(seed), exist_ok=True)
    runtime.save_json(os.path.join(_results_dir(seed), "folds.json"), {
        "seed": seed, "num_folds": NUM_FOLDS, "stratify": "diagnosis_binary", "level": "case (không có patient ID)",
        "assignment": {int(c): int(f) for c, f in zip(train["case_num"], assignment)},
        "heldout_melanoma": {f: int(train.loc[assignment == f, "diagnosis_binary"].sum()) for f in range(NUM_FOLDS)},
        "fit_state_support": support})
    print("Fold sizes:", np.bincount(assignment).tolist())


# Train công thức unified với budget cố định, không đánh giá hay chọn checkpoint trên validation.
def _train_base(manifest_path, checkpoint_path, seed, device):
    runtime.set_seed(seed)
    settings = run_ecbm._resolve("unified")
    bundle = get_dataloaders(manifest_path=manifest_path, project_root=PROJECT_ROOT, batch_size=settings["batch_size"],
                             num_workers=0, target_size=224, augment_train=True,
                             augmentation_preset=run_ecbm.DEFAULT_AUGMENTATION_PRESET, include_test=False)
    class_weights = bundle["class_weights"].to(device)
    statistics = compute_concept_statistics(manifest_path, PROJECT_ROOT)
    state_weights = [torch.as_tensor(statistics[c]["weights"], dtype=torch.float32, device=device) for c in CONCEPT_NAMES]
    model = run_ecbm._build_model(_config(), pretrained=True).to(device)
    optimizer, scheduler = run_ecbm._build_optimizer(model, "unified", settings)
    for epoch in range(1, settings["epochs"] + 1):
        losses = run_ecbm.train_one_epoch(model, bundle["train"], optimizer, device, class_weights, state_weights)
        scheduler.step()
        if epoch % 10 == 0:
            print(f"  epoch {epoch:02d}: loss {losses['total']:.4f} (class {losses['class']:.4f}, "
                  f"concept {losses['concept']:.4f}, global {losses['global']:.4f})", flush=True)
    torch.save({"epoch": settings["epochs"], "model_state_dict": model.state_dict(), "seed": seed,
                "manifest_sha256": runtime.file_sha256(manifest_path),
                "train_case_nums": sorted(int(c) for c in bundle["datasets"]["train"].df["case_num"]),
                "class_weights": class_weights.cpu().tolist(),
                "selection": "fixed endpoint, không đọc validation"}, checkpoint_path)


def base(seed, device, overwrite):
    device = runtime.get_device(device)
    jobs = [(fold, _fold_manifest(seed, fold)) for fold in range(NUM_FOLDS)] + [(None, MANIFEST_PATH)]
    for fold, manifest_path in jobs:
        path = _base_path(seed, fold)
        if os.path.exists(path) and not overwrite:
            print(f"Bỏ qua {path} (đã có)")
            continue
        print(f"Train base {'full' if fold is None else f'fold {fold}'} (seed {seed})", flush=True)
        _train_base(manifest_path, path, seed, device)


def cache(seed, device):
    device = runtime.get_device(device)
    mapping = load_concept_schema(MANIFEST_PATH)["label_mapping"]
    assignment = json.load(open(os.path.join(_results_dir(seed), "folds.json")))["assignment"]
    full, full_checkpoint = _load_model(_base_path(seed), device)
    parts = []
    for fold in range(NUM_FOLDS):
        model, checkpoint = _load_model(_base_path(seed, fold), device)
        out = _local_outputs(model, _fold_manifest(seed, fold), "heldout", device, mapping)
        if set(out["case_num"]) & set(checkpoint["train_case_nums"]):
            raise ValueError(f"Fold {fold}: ca held-out nằm trong tập train của fold model")
        if any(assignment[str(c)] != fold for c in out["case_num"]):
            raise ValueError(f"Fold {fold}: case không khớp fold assignment")
        parts.append(out)
    order = torch.as_tensor(np.argsort(np.concatenate([p["case_num"] for p in parts])))
    oof = {"e_class": torch.cat([p["e_class"] for p in parts])[order],
           "e_concepts": [torch.cat([p["e_concepts"][g] for p in parts])[order] for g in range(len(CONCEPT_NAMES))],
           "labels": torch.cat([p["labels"] for p in parts])[order],
           "concepts": torch.cat([p["concepts"] for p in parts])[order],
           "case_num": np.concatenate([p["case_num"] for p in parts])[order.numpy()].tolist()}
    in_sample = _local_outputs(full, MANIFEST_PATH, "train", device, mapping)
    order = torch.as_tensor(np.argsort(in_sample["case_num"]))
    in_sample = {key: ([v[order] for v in value] if key == "e_concepts" else
                       (np.asarray(value)[order.numpy()].tolist() if key == "case_num" else value[order]))
                 for key, value in in_sample.items()}
    if oof["case_num"] != in_sample["case_num"] or len(oof["case_num"]) != 413:
        raise ValueError("OOF và in-sample không cùng 413 ca")
    if not (torch.equal(oof["labels"], in_sample["labels"]) and torch.equal(oof["concepts"], in_sample["concepts"])):
        raise ValueError("Nhãn hoặc concept khác nhau giữa hai cache")
    if set(full_checkpoint["train_case_nums"]) != set(oof["case_num"]):
        raise ValueError("Model đủ train không train đúng 413 ca")
    valid = _local_outputs(full, MANIFEST_PATH, "valid", device, mapping)
    path = os.path.join(_checkpoint_dir(seed), "cache.pt")
    torch.save({"oof": oof, "in_sample": in_sample, "valid": valid,
                "full_sha256": runtime.file_sha256(_base_path(seed)),
                "fold_sha256": [runtime.file_sha256(_base_path(seed, f)) for f in range(NUM_FOLDS)]}, path)
    summary = {name: _local_summary(data) for name, data in (("oof_train", oof), ("in_sample_train", in_sample),
                                                              ("full_valid", valid))}
    runtime.save_json(os.path.join(_results_dir(seed), "cache_summary.json"), summary)
    for name, s in summary.items():
        print(f"{name:16s}: E_class NLL {s['e_class_nll']:.3f} F1 {s['e_class_f1_macro_argmax']:.3f} "
              f"max-prob {s['e_class_mean_max_probability']:.3f} | concept acc {100 * s['local_concept_accuracy']:.1f}%")


# ---------- Joint likelihood trên local energies đã cache ----------

def profile_index(concepts):
    return (concepts * STRIDES).sum(-1)


# Logits chưa chuẩn hóa −E(x,c,y) cho mọi profile và nhãn: [B, P, 2]; bias là energy cộng thêm cho y = 1.
def joint_logits(e_class, e_concepts, g, weights, bias=0.0, allowed=None):
    w_y, w_c, w_g = weights
    concept_sum = sum(e[:, PROFILES[:, i]] for i, e in enumerate(e_concepts))
    energy = w_y * e_class[:, None, :] + w_c * concept_sum[:, :, None] + w_g * g[None]
    energy = energy + bias * torch.tensor([0.0, 1.0], dtype=energy.dtype)
    logits = -energy
    if allowed is not None:
        logits = logits.masked_fill(~allowed[:, :, None], float("-inf"))
    return logits


def allowed_profiles(concepts, mask):
    # mask [B, 7]: nhóm đã xác nhận được clamp về concept thật.
    return ((PROFILES[None] == concepts[:, None, :]) | ~mask[:, None, :]).all(-1)


# −log p(c_{S̄}*, y* | x, c_S*) theo cùng total energy dùng khi suy luận.
def conditional_nll(e_class, e_concepts, g, weights, concepts, labels, mask, bias=0.0):
    logits = joint_logits(e_class, e_concepts, g, weights, bias, allowed_profiles(concepts, mask))
    target = logits[torch.arange(len(labels)), profile_index(concepts), labels]
    return torch.logsumexp(logits.flatten(1), dim=1) - target


def marginals(e_class, e_concepts, g, weights, bias=0.0, allowed=None):
    logits = joint_logits(e_class, e_concepts, g, weights, bias, allowed)
    log_joint = logits - torch.logsumexp(logits.flatten(1), dim=1)[:, None, None]
    profile_probability = torch.logsumexp(log_joint, dim=2).exp()
    concept_probability = [torch.zeros(len(e_class), k).index_add_(1, PROFILES[:, g], profile_probability)
                           for g, k in enumerate(STATE_COUNTS)]
    return torch.logsumexp(log_joint, dim=1).exp()[:, 1].clamp(0.0, 1.0), concept_probability


# π(∅) = 1/2; còn lại chọn k đều trong 1..7 rồi một subset đều cỡ k (thứ hạng ngẫu nhiên < k).
def sample_masks(n, generator):
    groups = len(CONCEPT_NAMES)
    use = torch.rand(n, generator=generator) < 0.5
    budget = torch.randint(1, groups + 1, (n,), generator=generator)
    ranks = torch.rand(n, groups, generator=generator).argsort(1).argsort(1)
    return use[:, None] & (ranks < budget[:, None])


def _head_model(seed):
    model, _ = _load_model(_base_path(seed), "cpu")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = list(model.global_projection.parameters()) + list(model.global_output.parameters())
    for parameter in head:
        parameter.requires_grad_(True)
    return model, head


def _case_weights(labels):
    counts = torch.bincount(labels, minlength=2).double()
    return (len(labels) / (2 * counts))[labels].float()


# A1, A2-*, A3-*: cùng khởi tạo A0, cùng thứ tự minibatch, cùng budget; chỉ cập nhật global head.
def _train_head(seed, arm, data):
    model, head = _head_model(seed)
    optimizer = torch.optim.AdamW(head, lr=HEAD["lr"], weight_decay=HEAD["weight_decay"])
    order_generator = torch.Generator().manual_seed(seed)
    perturb_generator = torch.Generator().manual_seed(seed + 1)
    mask_generator = torch.Generator().manual_seed(seed + 2)
    weights = _case_weights(data["labels"])
    no_mask = torch.zeros(HEAD["batch_size"], len(CONCEPT_NAMES), dtype=torch.bool)
    history = []
    for epoch in range(HEAD["epochs"]):
        total, seen = 0.0, 0.0
        for idx in torch.randperm(len(data["labels"]), generator=order_generator).split(HEAD["batch_size"]):
            labels, concepts = data["labels"][idx], data["concepts"][idx]
            if arm == "A1":
                perturbed = perturb_concepts(concepts, STATE_COUNTS, *run_ecbm.PERTURBATION, generator=perturb_generator)
                losses = F.cross_entropy(-model.global_class_energies(perturbed), labels, reduction="none")
            else:
                g = model.global_class_energies(PROFILES)
                mask = sample_masks(len(idx), mask_generator) if arm.startswith("A3") else no_mask[:len(idx)]
                losses = conditional_nll(data["e_class"][idx], [e[idx] for e in data["e_concepts"]], g,
                                         JOINT_WEIGHTS, concepts, labels, mask)
            loss = (weights[idx] * losses).sum() / weights[idx].sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * float(weights[idx].sum())
            seen += float(weights[idx].sum())
        history.append(total / seen)
    return {name: value.detach().clone() for name, value in model.state_dict().items()
            if name.startswith(("global_projection", "global_output"))}, history


# A0-cal / A3-cal: chỉ học (w_y, w_c, w_G, label bias) bằng joint NLL trên OOF energies, G giữ nguyên A0.
# A0-cal: S luôn rỗng. A3-cal: lặp mỗi ca masks_per_case lần với mask cố định lấy theo π.
# Trọng số energy tham số hóa w = exp(θ) để không âm; bias không ràng buộc.
def _fit_calibration(seed, data, masks_per_case=0):
    model, _ = _head_model(seed)
    with torch.no_grad():
        g = model.global_class_energies(PROFILES)
    params = torch.zeros(4, requires_grad=True)
    rows = torch.arange(len(data["labels"])).repeat_interleave(max(masks_per_case, 1))
    if masks_per_case:
        masks = sample_masks(len(rows), torch.Generator().manual_seed(seed + CALIBRATION_MASK_SEED_OFFSET))
    else:
        masks = torch.zeros(len(rows), len(CONCEPT_NAMES), dtype=torch.bool)
    e_class, e_concepts = data["e_class"][rows], [e[rows] for e in data["e_concepts"]]
    labels, concepts = data["labels"][rows], data["concepts"][rows]
    weights = _case_weights(data["labels"])[rows]
    optimizer = torch.optim.LBFGS([params], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        losses = conditional_nll(e_class, e_concepts, g, tuple(params[:3].exp()), concepts, labels, masks, bias=params[3])
        loss = (weights * losses).sum() / weights.sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    fitted = params.detach()
    return [float(v) for v in fitted[:3].exp()] + [float(fitted[3])]


# Chỉ học các arm được yêu cầu; gộp vào heads.json đã có để không đổi các arm cũ.
def heads(seed, arms):
    store = torch.load(os.path.join(_checkpoint_dir(seed), "cache.pt"), weights_only=False)
    out_dir = os.path.join(_checkpoint_dir(seed), "heads")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(_results_dir(seed), "heads.json")
    report = (json.load(open(report_path)) if os.path.exists(report_path)
              else {"budget": HEAD, "joint_weights": list(JOINT_WEIGHTS), "arms": {}})
    for arm in arms:
        if arm in HEAD_ARMS:
            state, history = _train_head(seed, arm, store[HEAD_ARMS[arm]])
            torch.save(state, os.path.join(out_dir, f"{arm}.pt"))
            report["arms"][arm] = {"data": HEAD_ARMS[arm], "masks": "mixed" if arm.startswith("A3") else "none",
                                   "loss_history": history}
            print(f"{arm}: loss {history[0]:.4f} -> {history[-1]:.4f}")
        else:
            calibration = _fit_calibration(seed, store["oof"], CALIBRATION_ARMS[arm])
            report["arms"][arm] = {"data": "oof", "masks_per_case": CALIBRATION_ARMS[arm], "w_y_w_c_w_G_bias": calibration}
            print(f"{arm} (w_y, w_c, w_G, bias):", [round(v, 4) for v in calibration])
    runtime.save_json(report_path, report)


# ---------- Đánh giá trên validation ----------

def _arm_spec(seed, arm, report):
    model, _ = _head_model(seed)
    weights, bias = JOINT_WEIGHTS, 0.0
    if arm in HEAD_ARMS:
        model.load_state_dict(torch.load(os.path.join(_checkpoint_dir(seed), "heads", f"{arm}.pt")), strict=False)
    if arm in CALIBRATION_ARMS:
        *weights, bias = report["arms"][arm]["w_y_w_c_w_G_bias"]
    with torch.no_grad():
        return model.global_class_energies(PROFILES), tuple(weights), float(bias)


def _evaluate_arm(data, g, weights, bias, observed):
    labels, concepts = data["labels"].numpy(), data["concepts"]
    with torch.no_grad():
        scores, concept_probability = marginals(data["e_class"], data["e_concepts"], g, weights, bias)
    scores = scores.numpy()
    threshold, _ = select_threshold(labels, scores, THRESHOLD_METRIC)
    base_pred = (scores >= threshold).astype(int)
    automatic = compute_metrics(labels, base_pred, scores)
    concept_true = {n: concepts[:, i].numpy() for i, n in enumerate(CONCEPT_NAMES)}
    base_concepts = [p.argmax(1) for p in concept_probability]
    concept = compute_concept_metrics(concept_true, {n: p.numpy() for n, p in zip(CONCEPT_NAMES, base_concepts)},
                                      CONCEPT_NUM_CLASSES, train_observed_states=observed)
    curve, propagation = {m: [] for m in range(8)}, {m: [] for m in range(1, 7)}
    for m in range(8):
        for subset in itertools.combinations(range(len(CONCEPT_NAMES)), m):
            mask = torch.zeros(len(labels), len(CONCEPT_NAMES), dtype=torch.bool)
            mask[:, list(subset)] = True
            with torch.no_grad():
                s, probs = marginals(data["e_class"], data["e_concepts"], g, weights, bias,
                                     allowed_profiles(concepts, mask))
            pred = (s.numpy() >= threshold).astype(int)
            curve[m].append(compute_metrics(labels, pred, s.numpy())["f1_macro"])
            if m == 7:
                flips = int((pred != base_pred).sum())
            if 1 <= m <= 6:
                free = [g_ for g_ in range(len(CONCEPT_NAMES)) if g_ not in subset]
                gained = sum(int(((probs[g_].argmax(1) == concepts[:, g_]) & (base_concepts[g_] != concepts[:, g_])).sum())
                             for g_ in free)
                lost = sum(int(((probs[g_].argmax(1) != concepts[:, g_]) & (base_concepts[g_] == concepts[:, g_])).sum())
                           for g_ in free)
                propagation[m].append((gained - lost) / len(labels))
    f = [float(np.mean(curve[m])) for m in range(8)]
    return {"decision_threshold": float(threshold),
            "automatic": {k: automatic[k] for k in ("accuracy", "balanced_accuracy", "f1_macro", "roc_auc",
                                                    "sensitivity", "specificity")},
            "concept_accuracy": concept["overall_concept_accuracy"],
            "concept_f1_macro": concept["overall_f1_macro_all_defined"],
            "f_m": f, "auic_f1": float(sum((f[m] + f[m + 1]) / 2 for m in range(7)) / 7),
            "full_gt_f1": f[7], "flips_m7": flips,
            "propagation_net_per_case": {m: float(np.mean(v)) for m, v in propagation.items()}}


# Tín hiệu học nhãn còn lại của total objective: 1 − p(y*|·) khi tự động và khi clamp đủ 7 nhóm.
def _label_signal(data, g):
    full = torch.ones(len(data["labels"]), len(CONCEPT_NAMES), dtype=torch.bool)
    with torch.no_grad():
        auto, _ = marginals(data["e_class"], data["e_concepts"], g, JOINT_WEIGHTS)
        clamped, _ = marginals(data["e_class"], data["e_concepts"], g, JOINT_WEIGHTS,
                               allowed=allowed_profiles(data["concepts"], full))
        label_nll = conditional_nll(data["e_class"], data["e_concepts"], g, JOINT_WEIGHTS, data["concepts"],
                                    data["labels"], full)
        joint_nll = conditional_nll(data["e_class"], data["e_concepts"], g, JOINT_WEIGHTS, data["concepts"],
                                    data["labels"], torch.zeros_like(full))
    y = data["labels"].float()
    residual = lambda p: float((1 - torch.where(y == 1, p, 1 - p)).mean())
    return {"automatic_label_residual": residual(auto), "full_clamp_label_residual": residual(clamped),
            "joint_nll_S_empty": float(joint_nll.mean()), "label_nll_S_full": float(label_nll.mean())}


def evaluate(seed):
    store = torch.load(os.path.join(_checkpoint_dir(seed), "cache.pt"), weights_only=False)
    report = json.load(open(os.path.join(_results_dir(seed), "heads.json")))
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"is_inconsistent_profile": bool})
    observed = compute_train_observed_states(manifest[manifest["split"] == "train"],
                                                    load_concept_schema(MANIFEST_PATH)["label_mapping"])
    results, diagnostics = {}, {}
    for arm in ARMS:
        g, weights, bias = _arm_spec(seed, arm, report)
        results[arm] = {"energy_weights": list(weights), "label_bias": bias,
                        **_evaluate_arm(store["valid"], g, weights, bias, observed)}
        if arm == "A0":
            diagnostics = {name: _label_signal(store[name], g) for name in ("in_sample", "oof", "valid")}
    runtime.save_json(os.path.join(_results_dir(seed), "validation.json"),
                      {"seed": seed, "num_cases": len(store["valid"]["labels"]), "arms": results,
                       "label_signal_A0": diagnostics, "test_read": False})
    print(f"Validation ({len(store['valid']['labels'])} ca), seed {seed}")
    print(f"  {'Arm':<8} {'Macro F1':>9} {'ROC-AUC':>8} {'BAcc %':>7} {'cAcc %':>7} {'cF1':>7} "
          f"{'F1 m=7':>7} {'AUIC-F1':>8} {'flips7':>7}")
    for arm, r in results.items():
        print(f"  {arm:<8} {r['automatic']['f1_macro']:>9.4f} {r['automatic']['roc_auc']:>8.4f} "
              f"{100 * r['automatic']['balanced_accuracy']:>7.2f} {100 * r['concept_accuracy']:>7.2f} "
              f"{r['concept_f1_macro']:>7.4f} {r['full_gt_f1']:>7.4f} {r['auic_f1']:>8.4f} {r['flips_m7']:>7d}")
    print("Tín hiệu học nhãn với G của A0, trọng số (1,1,1):")
    for name, d in diagnostics.items():
        print(f"  {name:10s}: 1-p(y*) tự động {d['automatic_label_residual']:.4f} | clamp đủ {d['full_clamp_label_residual']:.4f}"
              f" | joint NLL {d['joint_nll_S_empty']:.3f}")


# ---------- Kiểm tra số học ----------

def check(seed, device):
    device = runtime.get_device(device)
    full, _ = _load_model(_base_path(seed), device)
    mapping = load_concept_schema(MANIFEST_PATH)["label_mapping"]
    dataset = Derm7ptDataset(manifest_path=MANIFEST_PATH, project_root=PROJECT_ROOT, split="valid", label_mapping=mapping,
                             transform=get_transforms("valid", target_size=224, augment=False))
    batch = next(iter(DataLoader(dataset, batch_size=8, shuffle=False)))
    with torch.no_grad():
        features = full.extract_features(batch["image"].to(device))
        reference = exact_inference(full, features, JOINT_WEIGHTS)
        e_class = full.class_energies(features).cpu()
        e_concepts = [e.cpu() for e in full.concept_energies(features)]
        g = full.global_class_energies(PROFILES.to(device)).cpu()
    scores, concept_probability = marginals(e_class, e_concepts, g, JOINT_WEIGHTS)
    print("1. S rỗng khớp exact_inference: max |Δ| nhãn",
          f"{(scores - reference['class_probabilities'][:, 1].cpu()).abs().max():.2e}, concept",
          f"{max((p - r.cpu()).abs().max() for p, r in zip(concept_probability, reference['concept_probabilities'])):.2e}")
    concepts, labels = batch["concept_indices"].long(), batch["label"].long()
    assert torch.equal(PROFILES[profile_index(concepts)], concepts), "profile_index sai"
    mask = torch.zeros(len(labels), len(CONCEPT_NAMES), dtype=torch.bool)
    mask[:, [0, 3, 6]] = True
    logits = joint_logits(e_class, e_concepts, g, JOINT_WEIGHTS, allowed=allowed_profiles(concepts, mask))
    probability = (logits - torch.logsumexp(logits.flatten(1), 1)[:, None, None]).exp()
    violating = ~allowed_profiles(concepts, mask)
    print(f"2. Chuẩn hóa: max |Σp − 1| {(probability.sum((1, 2)) - 1).abs().max():.2e}; "
          f"xác suất ngoài clamp {probability[violating].sum():.2e}")
    full_mask = torch.ones_like(mask)
    nll = conditional_nll(e_class, e_concepts, g, JOINT_WEIGHTS, concepts, labels, full_mask)
    closed = F.cross_entropy(-(e_class + g[profile_index(concepts)]), labels, reduction="none")
    print(f"3. Clamp đủ 7 nhóm = CE của E_class + E_global(c*): max |Δ| {(nll - closed).abs().max():.2e}")
    model, head = _head_model(seed)
    model.double()
    data = [e_class.double(), [e.double() for e in e_concepts]]
    parameter = model.global_output.weight
    loss = lambda: conditional_nll(*data, model.global_class_energies(PROFILES), JOINT_WEIGHTS, concepts, labels, mask).sum()
    value = loss()
    gradient, = torch.autograd.grad(value, parameter)
    eps = 1e-6
    with torch.no_grad():
        parameter[0, 0] += eps
        plus = float(loss())
        parameter[0, 0] -= 2 * eps
        minus = float(loss())
    print(f"4. Gradient: autograd {float(gradient[0, 0]):.6f} | sai phân {(plus - minus) / (2 * eps):.6f}")


# Gộp mean ± SD qua seeds và đối chiếu tiêu chí: arm − A1 về AUIC-F1 ≥ 0.01 ở ≥ 2/3 seed, kèm hai guardrail.
def summary(seeds):
    out_dir = os.path.dirname(_results_dir(0))
    runs = {s: json.load(open(os.path.join(_results_dir(s), "validation.json")))["arms"] for s in seeds}
    metrics = {"f1_macro": lambda r: r["automatic"]["f1_macro"], "roc_auc": lambda r: r["automatic"]["roc_auc"],
               "concept_accuracy": lambda r: r["concept_accuracy"], "concept_f1_macro": lambda r: r["concept_f1_macro"],
               "full_gt_f1": lambda r: r["full_gt_f1"], "auic_f1": lambda r: r["auic_f1"], "flips_m7": lambda r: r["flips_m7"]}
    table = {arm: {key: {"mean": float(np.mean([f(runs[s][arm]) for s in seeds])),
                         "sd": float(np.std([f(runs[s][arm]) for s in seeds], ddof=1))} for key, f in metrics.items()}
             for arm in ARMS if all(arm in runs[s] for s in seeds)}
    decisions = {}
    for arm in ("A0-cal", "A2-OOF", "A3-OOF", "A3-cal"):
        if any(arm not in runs[s] for s in seeds):
            continue
        delta = [runs[s][arm]["auic_f1"] - runs[s]["A1"]["auic_f1"] for s in seeds]
        guard = lambda key: all(metrics[key](runs[s][arm]) >= max(metrics[key](runs[s]["A0"]), metrics[key](runs[s]["A1"])) - 0.01
                                for s in seeds)
        decisions[arm] = {"auic_minus_A1": delta, "seeds_passing": int(sum(d >= 0.01 for d in delta)),
                          "guardrail_f1_macro": guard("f1_macro"), "guardrail_concept_f1_macro": guard("concept_f1_macro")}
        decisions[arm]["passed"] = (decisions[arm]["seeds_passing"] * 3 >= 2 * len(seeds)
                                    and decisions[arm]["guardrail_f1_macro"] and decisions[arm]["guardrail_concept_f1_macro"])
    reports = {s: json.load(open(os.path.join(_results_dir(s), "heads.json")))["arms"] for s in seeds}
    calibration = {arm: {s: reports[s][arm]["w_y_w_c_w_G_bias"] for s in seeds}
                   for arm in CALIBRATION_ARMS if all(arm in reports[s] for s in seeds)}
    if all("A3-OOF" in runs[s] for s in seeds):
        decisions["A3-OOF_minus_A2-OOF_auic"] = [runs[s]["A3-OOF"]["auic_f1"] - runs[s]["A2-OOF"]["auic_f1"] for s in seeds]
        print("A3-OOF − A2-OOF AUIC:", [round(x, 4) for x in decisions["A3-OOF_minus_A2-OOF_auic"]])
    runtime.save_json(os.path.join(out_dir, "summary.json"), {
        "seeds": list(seeds), "split": "validation", "test_read": False,
        "criteria_sha256": {name: runtime.file_sha256(os.path.join(out_dir, name)) for name in ("criteria.json", "criteria_a3.json")
                            if os.path.exists(os.path.join(out_dir, name))},
        "table": table, "decisions": decisions, "calibration_weights": calibration})
    for arm, d in decisions.items():
        if not isinstance(d, dict):
            continue
        print(f"{arm}: AUIC − A1 {[round(x, 4) for x in d['auic_minus_A1']]}, {d['seeds_passing']}/{len(seeds)} seed đạt, "
              f"guardrail F1 {d['guardrail_f1_macro']}, concept F1 {d['guardrail_concept_f1_macro']} -> "
              f"{'ĐẠT' if d['passed'] else 'KHÔNG ĐẠT'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pilot alignment cho categorical ECBM, chỉ train/validation")
    parser.add_argument("--stage", choices=["gap", "folds", "base", "cache", "heads", "evaluate", "check", "summary"],
                        required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arms", nargs="+", choices=[*HEAD_ARMS, *CALIBRATION_ARMS],
                        default=["A1", "A2-IS", "A2-OOF", "A0-cal"], help="Arm học ở stage heads")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "gap":
        gap(args.device)
    elif args.stage == "folds":
        folds(args.seed)
    elif args.stage == "base":
        base(args.seed, args.device, args.overwrite)
    elif args.stage == "cache":
        cache(args.seed, args.device)
    elif args.stage == "heads":
        heads(args.seed, args.arms)
    elif args.stage == "evaluate":
        evaluate(args.seed)
    elif args.stage == "summary":
        summary((42, 123, 2026))
    else:
        check(args.seed, args.device)


if __name__ == "__main__":
    main()
