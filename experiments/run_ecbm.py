# Runner ECBM: Huấn luyện và đánh giá Categorical ECBM trên Derm7pt.
# Hai công thức train: "unified" (quy trình chung với Black-box/Soft CBM/Hard CBM, mặc định) và "paper" (mã gốc).
# Suy luận trên cùng checkpoint: exact enumeration; gradient (Algorithm 1) với concept làm cứng trong E_global
# (Hard ECBM, như mã gốc); gradient_soft với concept dạng xác suất trong E_global (Soft ECBM).


import argparse
import os
import platform
import sys
import time
from typing import Any, Dict, List, Tuple

# Thêm thư mục gốc vào đường dẫn hệ thống
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import src.dataset trước sklearn để tránh xung đột thư viện OpenMP trên một số nền tảng.
from src.dataset import (CONCEPT_NAMES, CONCEPT_NUM_CLASSES, Derm7ptDataset, compute_concept_statistics,
                         get_dataloaders)

import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from torch.utils.data import DataLoader

from experiments import run_soft_cbm as shared
from src.ecbm import CategoricalECBM, ecbm_training_loss, exact_inference, gradient_inference
from src.metrics import compute_concept_metrics, compute_metrics, print_concept_metrics_table, print_metrics_table
from src.protocol import (load_concept_schema, manifest_fingerprint, prepare_training_outputs,
                          save_experiment_results, validate_concept_schema, validate_manifest)
from src.selection import (DEFAULT_CHECKPOINT_METRIC, DEFAULT_THRESHOLD_METRIC, add_selection_argument,
                           add_threshold_argument, metric_from_config, run_paths, selection_name,
                           selection_score, threshold_metric_from_config, threshold_selection_name)
from src.naming import DEFAULT_BACKBONE
from src.transforms import get_transforms

# Công thức train. "unified" dùng đúng ngân sách/optimizer/lịch LR/trọng số của Black-box, Soft CBM, Hard CBM;
# "paper" giữ công thức mã gốc (xmed-lab/ECBM, configs/cub.json) để kiểm tra cài đặt.
RECIPES = {
    "unified": {"protocol": "categorical_ecbm_unified_v1", "epochs": 50, "patience": None, "batch_size": 16,
                "optimizer": "AdamW", "lr": 1e-4, "head_lr": 1e-3, "weight_decay": 1e-2, "momentum": None,
                "scheduler": "CosineAnnealingLR(eta_min=1e-6)", "class_weighting": "balanced",
                "concept_weighting": "balanced", "selection_inference": "exact"},
    "paper": {"protocol": "categorical_ecbm_paper_v1", "epochs": 100, "patience": 15, "batch_size": 32,
              "optimizer": "SGD", "lr": 0.01, "head_lr": None, "weight_decay": 4e-5, "momentum": 0.9,
              "scheduler": "ReduceLROnPlateau(train loss, factor=0.1, patience=10)", "class_weighting": "none",
              "concept_weighting": "none", "selection_inference": "gradient"},
}
DEFAULT_RECIPE = "unified"
DEFAULT_HIDDEN_DIM = 32       # gốc 512 trên CUB
DEFAULT_SEED = 42
DEFAULT_AUGMENTATION_PRESET = "legacy_letterbox"
TRAIN_WEIGHTS = (3.0, 1.0, 1.0)        # λ_xy, λ_xc, λ_cy khi train (cấu trúc loss riêng của ECBM)
INFERENCE_WEIGHTS = (1.0, 1.0, 0.01)   # λ_xy, λ_xc, λ_cy khi suy luận
INFERENCE_STEPS = 80
INFERENCE_LR = 0.1
PERTURBATION = (0.2, 0.2)              # tỷ lệ ca và tỷ lệ nhóm concept bị xáo trộn cho E_global
# Soft và Hard ECBM train giống hệt nhau, chỉ khác cách E_global nhận concept khi suy luận gradient.
INFERENCE_METHODS = ("gradient", "exact", "gradient_soft")
INFERENCE_LABELS = {"gradient": "Hard ECBM (gradient)", "gradient_soft": "Soft ECBM (gradient)", "exact": "ECBM (exact)"}
ECBM_PROTOCOLS = {settings["protocol"]: name for name, settings in RECIPES.items()}


def _others(method: str, available=INFERENCE_METHODS) -> List[str]:
    return [m for m in available if m != method]


# Chạy các cách suy luận trên loader, trích feature ảnh một lần mỗi batch; không nhận nhãn hay concept thật.
def predict(model: CategoricalECBM, loader: DataLoader, device: torch.device,
            methods=INFERENCE_METHODS) -> Dict[str, Dict[str, Any]]:
    if any(method not in INFERENCE_METHODS for method in methods):
        raise ValueError(f"inference method must be one of {INFERENCE_METHODS}")
    model.eval()
    labels, concept_true = [], []
    scores = {m: [] for m in methods}
    concept_probabilities = {m: {c: [] for c in CONCEPT_NAMES} for m in methods}
    for batch in loader:
        with torch.no_grad():
            features = model.extract_features(batch["image"].to(device))
        for method in methods:
            if method in ("gradient", "gradient_soft"):
                output = gradient_inference(model, features, INFERENCE_WEIGHTS, INFERENCE_STEPS, INFERENCE_LR,
                                            hard_global=method == "gradient")
            else:
                output = exact_inference(model, features, INFERENCE_WEIGHTS)
            scores[method].extend(output["class_probabilities"][:, 1].cpu().tolist())
            for name, probability in zip(CONCEPT_NAMES, output["concept_probabilities"]):
                concept_probabilities[method][name].append(probability.cpu().numpy())
        labels.extend(batch["label"].tolist())
        concept_true.append(batch["concept_indices"])
    concept_true = torch.cat(concept_true).numpy()
    outputs = {}
    for method in methods:
        probabilities = {c: np.concatenate(concept_probabilities[method][c]) for c in CONCEPT_NAMES}
        outputs[method] = {
            "labels": np.asarray(labels, dtype=int),
            "scores": np.clip(np.asarray(scores[method], dtype=float), 0.0, 1.0),
            "concepts": {"true": {c: concept_true[:, i] for i, c in enumerate(CONCEPT_NAMES)},
                         "pred": {c: probabilities[c].argmax(1) for c in CONCEPT_NAMES},
                         "probabilities": probabilities},
        }
    return outputs


def _score(output, threshold, train_observed_states):
    diagnosis = compute_metrics(output["labels"], output["scores"] >= threshold, output["scores"])
    concepts = compute_concept_metrics(output["concepts"]["true"], output["concepts"]["pred"],
                                       CONCEPT_NUM_CLASSES, train_observed_states=train_observed_states)
    return diagnosis, concepts


# Kết quả một cách suy luận; ngưỡng chọn trên validation theo tiêu chí đã khai báo nếu chưa có, rồi đóng băng cho test.
def _inference_block(output, method, df, train_observed_states, split, threshold=None,
                     threshold_metric=DEFAULT_THRESHOLD_METRIC):
    at_05, concepts = _score(output, 0.5, train_observed_states)
    if threshold is None:
        threshold, _ = shared.select_validation_threshold(output["labels"], output["scores"], threshold_metric)
    metrics, _ = _score(output, threshold, train_observed_states)
    predictions = shared._extract_sample_predictions(df, (output["scores"] >= threshold).astype(int),
                                                     output["scores"], output["concepts"])
    return {"inference": method, "decision_threshold": float(threshold), f"{split}_metrics_at_0.5": at_05,
            f"{split}_metrics": metrics, f"{split}_concept_metrics": concepts, f"{split}_predictions": predictions}


def train_one_epoch(model, loader, optimizer, device, class_weights=None, state_weights=None) -> Dict[str, float]:
    model.train()
    totals, seen = {}, 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss, parts = ecbm_training_loss(model, batch["image"].to(device), batch["concept_indices"].to(device),
                                         batch["label"].to(device), TRAIN_WEIGHTS, PERTURBATION,
                                         class_weights=class_weights, concept_state_weights=state_weights)
        loss.backward()
        optimizer.step()
        count = len(batch["label"])
        for name, value in {"total": float(loss.detach()), **parts}.items():
            totals[name] = totals.get(name, 0.0) + value * count
        seen += count
    return {name: value / seen for name, value in totals.items()}


# Điền tham số chưa truyền bằng giá trị của công thức; trả về bộ siêu tham số đầy đủ của run.
def _resolve(recipe, epochs=None, batch_size=None, lr=None, head_lr=None, weight_decay=None, patience=None,
             selection_inference=None):
    if recipe not in RECIPES:
        raise ValueError("recipe must be unified or paper")
    base = RECIPES[recipe]
    settings = {"epochs": epochs or base["epochs"], "batch_size": batch_size or base["batch_size"],
                "lr": lr or base["lr"], "head_lr": head_lr if head_lr is not None else base["head_lr"],
                "weight_decay": base["weight_decay"] if weight_decay is None else weight_decay,
                "patience": patience or base["patience"],
                "selection_inference": selection_inference or base["selection_inference"]}
    if settings["selection_inference"] not in INFERENCE_METHODS:
        raise ValueError(f"selection inference must be one of {INFERENCE_METHODS}")
    return settings


# Tên cấu hình: công thức, cách suy luận chọn checkpoint, epochs; thêm hậu tố khi lệch giá trị của công thức.
def _configuration(recipe, settings):
    base = RECIPES[recipe]
    name = f"{recipe}_{settings['selection_inference']}_e{settings['epochs']}"
    for key, tag in (("batch_size", "b"), ("lr", "lr"), ("head_lr", "headlr"), ("weight_decay", "wd"),
                     ("patience", "pat")):
        if settings[key] != base[key]:
            name += f"_{tag}{settings[key]:g}" if settings[key] is not None else f"_no{tag}"
    return name


def _default_paths(seed=DEFAULT_SEED, recipe=DEFAULT_RECIPE, checkpoint_metric=DEFAULT_CHECKPOINT_METRIC,
                   threshold_metric=DEFAULT_THRESHOLD_METRIC, **overrides) -> Tuple[str, str]:
    settings = _resolve(recipe, **overrides)
    return run_paths(PROJECT_ROOT, checkpoint_metric, "ecbm", _configuration(recipe, settings), seed,
                     threshold_metric=threshold_metric, backbone=DEFAULT_BACKBONE)


def _recipe_from_config(config):
    return config.get("recipe", ECBM_PROTOCOLS.get(config.get("protocol")))


def _config_paths(config):
    return _default_paths(config["seed"], _recipe_from_config(config), metric_from_config(config),
                          threshold_metric_from_config(config), epochs=config["epochs"],
                          batch_size=config["batch_size"], lr=config["learning_rate"],
                          head_lr=config.get("head_learning_rate"), weight_decay=config["weight_decay"],
                          patience=config.get("patience"), selection_inference=config["selection_inference"])


def validate_ecbm_config(config):
    if not isinstance(config, dict) or config.get("protocol") not in ECBM_PROTOCOLS:
        raise ValueError("Checkpoint is not a categorical ECBM protocol")
    if config.get("selection_inference") not in INFERENCE_METHODS:
        raise ValueError("Unknown ECBM selection inference")
    metric_from_config(config)
    threshold_metric_from_config(config)


def _build_model(config, pretrained):
    return CategoricalECBM(CONCEPT_NAMES, CONCEPT_NUM_CLASSES, num_classes=2, pretrained=pretrained,
                           hidden_dim=config["hidden_dim"], dropout=config["dropout"])


# Optimizer/lịch LR theo công thức: unified tách backbone (lr) khỏi các mạng năng lượng và embedding (head_lr).
def _build_optimizer(model, recipe, settings):
    if recipe == "unified":
        backbone = list(model.features.parameters())
        backbone_ids = {id(parameter) for parameter in backbone}
        heads = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
        optimizer = torch.optim.AdamW([
            {"params": backbone, "lr": settings["lr"], "name": "backbone"},
            {"params": heads, "lr": settings["head_lr"], "name": "energies_and_embeddings"},
        ], weight_decay=settings["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"], eta_min=1e-6)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=settings["lr"], momentum=RECIPES["paper"]["momentum"],
                                    weight_decay=settings["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
    return optimizer, scheduler


# Train/validation, chọn best epoch theo F1@0.5 của cách suy luận đã khai báo, rồi chọn ngưỡng cho cả hai cách.
def run_ecbm_experiment(
    manifest_path: str = None,
    recipe: str = DEFAULT_RECIPE,
    epochs: int = None,
    batch_size: int = None,
    lr: float = None,
    head_lr: float = None,
    weight_decay: float = None,
    patience: int = None,
    seed: int = DEFAULT_SEED,
    device_name: str = "auto",
    checkpoint_path: str = None,
    results_path: str = None,
    save_results: bool = True,
    augmentation_preset: str = DEFAULT_AUGMENTATION_PRESET,
    overwrite: bool = False,
    num_workers: int = 0,
    checkpoint_metric: str = DEFAULT_CHECKPOINT_METRIC,
    selection_inference: str = None,
    threshold_metric: str = DEFAULT_THRESHOLD_METRIC,
) -> Dict[str, Any]:
    settings = _resolve(recipe, epochs, batch_size, lr, head_lr, weight_decay, patience, selection_inference)
    epochs, batch_size, selection_inference = settings["epochs"], settings["batch_size"], settings["selection_inference"]
    if epochs < 1 or batch_size < 1 or num_workers < 0 or (settings["patience"] is not None and settings["patience"] < 1):
        raise ValueError("epochs/patience/batch_size must be positive and num_workers non-negative")
    if not np.isfinite(settings["lr"]) or settings["lr"] <= 0 or settings["weight_decay"] < 0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    selection_name(checkpoint_metric)
    threshold_selection_name(threshold_metric)
    shared.set_seed(seed)
    device = shared.get_device(device_name)

    if manifest_path is None:
        manifest_path = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
    manifest_path = os.path.abspath(manifest_path)
    checkpoint_path, results_path = prepare_training_outputs(
        _default_paths(seed, recipe, checkpoint_metric, threshold_metric, epochs=epochs, batch_size=batch_size,
                       lr=settings["lr"], head_lr=settings["head_lr"], weight_decay=settings["weight_decay"],
                       patience=settings["patience"], selection_inference=selection_inference),
        checkpoint_path, results_path, overwrite, save_results)
    shared._prepare_output(checkpoint_path, overwrite)
    if save_results:
        shared._prepare_output(results_path, overwrite)

    base = RECIPES[recipe]
    print(f"Khởi động Categorical ECBM, công thức {recipe} (EfficientNet-B0 -> E_class, E_concept, E_global)")
    print(f"Thiết bị sử dụng: {device} | Random seed: {seed} | Chọn checkpoint bằng suy luận: {selection_inference}")
    print(f"Cấu hình: Epochs={epochs}, Patience={settings['patience']}, Batch Size={batch_size}, {base['optimizer']} "
          f"LR={settings['lr']}, Head LR={settings['head_lr']}, Weight Decay={settings['weight_decay']}, "
          f"λ train={TRAIN_WEIGHTS}, λ suy luận={INFERENCE_WEIGHTS}, ngưỡng theo {threshold_metric}")

    bundle = get_dataloaders(
        manifest_path=manifest_path, project_root=PROJECT_ROOT, batch_size=batch_size, num_workers=num_workers,
        target_size=224, augment_train=True, augmentation_preset=augmentation_preset, include_test=False)
    train_loader, valid_loader = bundle["train"], bundle["valid"]
    assert len(bundle["datasets"]["train"]) == 413, "Train split phải có đúng 413 mẫu"
    assert len(bundle["datasets"]["valid"]) == 203, "Valid split phải có đúng 203 mẫu"
    schema = load_concept_schema(manifest_path)
    train_observed_states = shared.compute_train_observed_states(
        bundle["datasets"]["train"].df, bundle["datasets"]["train"].label_mapping)

    # Trọng số lớp/trạng thái tính từ train giống Soft/Hard CBM; công thức gốc không dùng trọng số.
    class_weights = state_weights = statistics = None
    if recipe == "unified":
        class_weights = bundle["class_weights"].to(device)
        statistics = compute_concept_statistics(manifest_path, PROJECT_ROOT)
        state_weights = [torch.as_tensor(statistics[c]["weights"], dtype=torch.float32, device=device)
                         for c in CONCEPT_NAMES]

    config = {
        "model": "ecbm",
        "protocol": base["protocol"],
        "recipe": recipe,
        "reference": "Xu et al., ICLR 2024; xmed-lab/ECBM (configs/cub.json, cub_inference.json)",
        "backbone": "EfficientNet-B0",
        "pretrained_weights": "EfficientNet_B0_Weights.DEFAULT",
        "energies": "E_class(x,y), E_concept(x,c_g) per group, E_global(c,y); gated embeddings as upstream",
        "hidden_dim": DEFAULT_HIDDEN_DIM,
        "dropout": 0.2,
        "epochs": epochs,
        "patience": settings["patience"],
        "batch_size": batch_size,
        "optimizer": base["optimizer"],
        "learning_rate": settings["lr"],
        "head_learning_rate": settings["head_lr"],
        "momentum": base["momentum"],
        "weight_decay": settings["weight_decay"],
        "scheduler": base["scheduler"],
        "train_energy_weights": list(TRAIN_WEIGHTS),
        "concept_perturbation": list(PERTURBATION),
        "class_weighting": base["class_weighting"],
        "class_weights": None if class_weights is None else [float(w) for w in class_weights.cpu().tolist()],
        "concept_weighting": base["concept_weighting"],
        "concept_state_weights": None if statistics is None else {c: statistics[c]["weights"] for c in CONCEPT_NAMES},
        "inference_energy_weights": list(INFERENCE_WEIGHTS),
        "gradient_inference": {"optimizer": "Adam", "lr": INFERENCE_LR, "steps": INFERENCE_STEPS,
                               "initialization": "zero logits",
                               "global_concepts": {"gradient": "hard argmax (Hard ECBM)",
                                                   "gradient_soft": "relaxed probabilities (Soft ECBM)"},
                               "readout": "lowest-energy step per case"},
        "exact_inference": "enumerate all categorical concept profiles x labels",
        "selection_inference": selection_inference,
        "seed": seed,
        "device": str(device),
        "augmentation_preset": augmentation_preset,
        "target_size": 224,
        "num_workers": num_workers,
        "checkpoint_selection": selection_name(checkpoint_metric),
        "threshold_selection": threshold_selection_name(threshold_metric),
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "numpy_version": str(np.__version__),
        "pandas_version": str(pd.__version__),
        "scikit_learn_version": str(sklearn.__version__),
        "platform": platform.platform(),
    }
    model = _build_model(config, pretrained=True).to(device)
    optimizer, scheduler = _build_optimizer(model, recipe, settings)
    manifest_sha256 = shared._manifest_hash(manifest_path)

    best_score, best_epoch, stale = -1.0, -1, 0
    history: List[Dict[str, Any]] = []
    print("\n--- Bắt đầu huấn luyện Categorical ECBM ---")
    for epoch in range(1, epochs + 1):
        learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
        started = time.monotonic()
        losses = train_one_epoch(model, train_loader, optimizer, device, class_weights, state_weights)
        if recipe == "unified":
            scheduler.step()
        else:
            scheduler.step(losses["total"])
        outputs = predict(model, valid_loader, device)
        validation = {m: _score(outputs[m], 0.5, train_observed_states) for m in INFERENCE_METHODS}
        diagnosis = validation[selection_inference][0]
        score = selection_score(diagnosis, checkpoint_metric)
        is_best = score > best_score
        history.append({
            "epoch": epoch, "learning_rates": learning_rates, "train_loss": losses,
            **{f"validation_{m}_{k}_at_0.5": validation[m][0][k]
               for m in INFERENCE_METHODS for k in ("f1_macro", "balanced_accuracy", "roc_auc")},
            **{f"validation_{m}_concept_f1_macro_all_defined": validation[m][1]["overall_f1_macro_all_defined"]
               for m in INFERENCE_METHODS},
            "selected": is_best, "seconds": time.monotonic() - started,
        })
        if is_best:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(), "config": config,
                "selection_score": score, "val_metrics": diagnosis,
                "manifest_sha256": manifest_sha256, "manifest_fingerprint": manifest_fingerprint(manifest_path),
                "concept_schema": schema,
            }, checkpoint_path)
        else:
            stale += 1
        print(f"Epoch [{epoch:03d}/{epochs:03d}] | Train Loss: {losses['total']:.4f} (Class: {losses['class']:.4f}, "
              f"Concept: {losses['concept']:.4f}, Global: {losses['global']:.4f}) | "
              + " | ".join(f"{m}: F1 {validation[m][0]['f1_macro']:.4f} AUC {validation[m][0]['roc_auc']:.4f} "
                           f"cF1 {validation[m][1]['overall_f1_macro_all_defined']:.4f}" for m in INFERENCE_METHODS)
              + f" | {history[-1]['seconds']:.0f}s" + (" [BEST]" if is_best else ""))
        if settings["patience"] is not None and stale >= settings["patience"]:
            break

    print(f"\nHuấn luyện hoàn tất! Checkpoint tốt nhất tại Epoch {best_epoch:02d} "
          f"(Valid {checkpoint_metric}@0.5 theo {selection_inference}: {best_score:.4f})")

    # Chọn ngưỡng trên validation của checkpoint đã chọn cho cả hai cách suy luận, không đọc test.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    valid_df = bundle["datasets"]["valid"].df
    outputs = predict(model, valid_loader, device)
    blocks = {m: _inference_block(outputs[m], m, valid_df, train_observed_states, "validation",
                                  threshold_metric=threshold_metric) for m in INFERENCE_METHODS}
    checkpoint["decision_threshold"] = blocks[selection_inference]["decision_threshold"]
    checkpoint["decision_thresholds"] = {m: blocks[m]["decision_threshold"] for m in INFERENCE_METHODS}
    torch.save(checkpoint, checkpoint_path)

    primary = blocks[selection_inference]
    valid_table = print_metrics_table(primary["validation_metrics"], title=(
        f"Validation Set ({selection_inference}, Epoch {best_epoch}, threshold={primary['decision_threshold']:.6f})"))
    concept_table = print_concept_metrics_table(primary["validation_concept_metrics"])
    results = {
        "model": "ecbm",
        "mode": "train_validation",
        "best_epoch": int(best_epoch),
        "epochs_completed": len(history),
        "hyperparameters": config,
        "manifest_sha256": manifest_sha256,
        "manifest_fingerprint": manifest_fingerprint(manifest_path),
        "concept_schema": schema,
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_sha256": shared._manifest_hash(checkpoint_path),
        "summary": f"Categorical ECBM ({recipe}), best epoch {best_epoch}, selection inference {selection_inference}\n\n"
                   f"{valid_table}\n\n{concept_table}",
        **{key: value for key, value in primary.items() if key != "inference"},
        "other_inferences": {m: blocks[m] for m in _others(selection_inference)},
        "train_observed_states": train_observed_states,
        "history": history,
    }
    if save_results:
        shared._save_json(results_path, results)
        print(f"Kết quả chi tiết đã lưu tại: {results_path}")
    return results


# Nạp checkpoint và hai ngưỡng đã đóng băng, chấm official test cho cả hai cách suy luận.
def evaluate_ecbm_test(checkpoint_path: str, manifest_path: str = None, device_name: str = "auto",
                       results_path: str = None, save_results: bool = True, overwrite: bool = False) -> Dict[str, Any]:
    if manifest_path is None:
        manifest_path = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
    manifest_path = os.path.abspath(manifest_path)
    device = shared.get_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "decision_thresholds" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Checkpoint has no frozen validation thresholds/config; use a newly trained checkpoint")
    validate_manifest(checkpoint, manifest_path)
    schema = validate_concept_schema(checkpoint, manifest_path)
    config = checkpoint["config"]
    validate_ecbm_config(config)
    if save_results:
        results_path = results_path or _config_paths(config)[1]
        shared._prepare_output(results_path, overwrite)

    train_df = pd.read_csv(manifest_path, dtype={"is_inconsistent_profile": bool})
    train_df = train_df[train_df["split"] == "train"].reset_index(drop=True)
    train_observed_states = shared.compute_train_observed_states(train_df, schema["label_mapping"])
    test_dataset = Derm7ptDataset(manifest_path=manifest_path, project_root=PROJECT_ROOT, split="test",
                                  label_mapping=schema["label_mapping"],
                                  transform=get_transforms("valid", target_size=config["target_size"], augment=False))
    assert len(test_dataset) == 395, "Test split phải có đúng 395 mẫu"
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    model = _build_model(config, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    selection_inference = config["selection_inference"]
    # Chỉ chấm các cách suy luận đã có ngưỡng chốt trên validation; checkpoint cũ không có gradient_soft.
    methods = [m for m in INFERENCE_METHODS if m in checkpoint["decision_thresholds"]]
    outputs = predict(model, test_loader, device, methods)
    blocks = {m: _inference_block(outputs[m], m, test_dataset.df, train_observed_states, "test",
                                  threshold=checkpoint["decision_thresholds"][m]) for m in methods}
    primary = blocks[selection_inference]
    test_table = print_metrics_table(primary["test_metrics"], title=(
        f"Test Set ({selection_inference}, threshold={primary['decision_threshold']:.6f})"))
    concept_table = print_concept_metrics_table(primary["test_concept_metrics"])
    results = {
        "model": config.get("model", "ecbm"),
        "mode": "final_test",
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_sha256": shared._manifest_hash(checkpoint_path),
        "best_epoch": int(checkpoint["epoch"]),
        "hyperparameters": config,
        "manifest_sha256": checkpoint["manifest_sha256"],
        "manifest_fingerprint": manifest_fingerprint(manifest_path),
        "concept_schema": schema,
        "summary": f"{test_table}\n\n{concept_table}",
        **{key: value for key, value in primary.items() if key != "inference"},
        "test_other_inferences": {m: blocks[m] for m in _others(selection_inference, methods)},
    }
    if save_results:
        save_experiment_results(results_path, test=results, overwrite=overwrite)
        print(f"Kết quả test đã lưu tại: {results_path}")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Categorical ECBM: train → validation → frozen test, hoặc --skip_test")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--skip_test", action="store_true", help="Pilot: chỉ train/validation; dùng với --mode train")
    parser.add_argument("--recipe", choices=list(RECIPES), default=DEFAULT_RECIPE,
                        help="unified (mặc định): cùng công thức với Black-box/Soft CBM/Hard CBM; paper: công thức mã gốc")
    parser.add_argument("--epochs", type=int, help="Mặc định theo công thức: unified 50, paper 100")
    parser.add_argument("--patience", type=int, help="Dừng sớm; mặc định unified không dừng sớm, paper 15")
    parser.add_argument("--batch_size", type=int, help="Mặc định theo công thức: unified 16, paper 32")
    parser.add_argument("--lr", type=float, help="LR backbone (unified 1e-4) hoặc toàn model (paper 0.01)")
    parser.add_argument("--head_lr", type=float, help="LR mạng năng lượng và embedding (unified 1e-3)")
    parser.add_argument("--weight_decay", type=float, help="Mặc định theo công thức: unified 0.01, paper 4e-5")
    parser.add_argument("--selection_inference", choices=INFERENCE_METHODS,
                        help="Cách suy luận chọn checkpoint: exact (mặc định unified), gradient = Hard ECBM "
                             "(mặc định paper), gradient_soft = Soft ECBM")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị: auto, mps, cuda, cpu")
    parser.add_argument("--augmentation_preset", choices=["legacy_letterbox", "comparison"],
                        default=DEFAULT_AUGMENTATION_PRESET)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Mặc định 0 như các runner khác; worker trên macOS khởi động lại mỗi epoch")
    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    add_selection_argument(parser)
    add_threshold_argument(parser)
    args = parser.parse_args(argv)
    if args.skip_test and args.mode != "train":
        parser.error("--skip_test is only valid with --mode train")
    overrides = dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, head_lr=args.head_lr,
                     weight_decay=args.weight_decay, patience=args.patience,
                     selection_inference=args.selection_inference)
    defaults = _default_paths(args.seed, args.recipe, args.checkpoint_metric, args.threshold_metric, **overrides)

    if args.mode == "train":
        checkpoint_path, results_path = prepare_training_outputs(
            defaults, args.checkpoint_path, args.results_path, args.overwrite, not args.no_save)
        validation = run_ecbm_experiment(
            manifest_path=args.manifest_path, recipe=args.recipe, seed=args.seed, device_name=args.device,
            checkpoint_path=checkpoint_path, save_results=False, augmentation_preset=args.augmentation_preset,
            overwrite=args.overwrite, num_workers=args.num_workers, checkpoint_metric=args.checkpoint_metric,
            threshold_metric=args.threshold_metric, **overrides)
        if not args.no_save:
            save_experiment_results(results_path, validation=validation, overwrite=args.overwrite)
        if args.skip_test:
            return validation
        test = evaluate_ecbm_test(checkpoint_path=checkpoint_path, manifest_path=args.manifest_path,
                                  device_name=args.device, save_results=False, overwrite=args.overwrite)
        if not args.no_save:
            return save_experiment_results(results_path, validation=validation, test=test, overwrite=True)
        return {**validation, **test, "mode": "train_validation_test"}

    checkpoint_path = args.checkpoint_path or defaults[0]
    results_path = args.results_path
    if results_path is None:
        results_path = _config_paths(torch.load(checkpoint_path, map_location="cpu", weights_only=True)["config"])[1]
    if not args.no_save:
        shared._prepare_output(results_path, args.overwrite)
    test = evaluate_ecbm_test(checkpoint_path=checkpoint_path, manifest_path=args.manifest_path,
                              device_name=args.device, save_results=False, overwrite=args.overwrite)
    if not args.no_save:
        return save_experiment_results(results_path, test=test, overwrite=args.overwrite)
    return test


if __name__ == "__main__":
    main()
