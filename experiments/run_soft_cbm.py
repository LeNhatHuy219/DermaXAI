# Runner Soft CBM: Huấn luyện và đánh giá Soft Joint CBM kết hợp ảnh, concept và chẩn đoán.


import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Thêm thư mục gốc vào đường dẫn hệ thống
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import src.dataset trước sklearn để tránh xung đột thư viện OpenMP trên một số nền tảng.

from src.dataset import (CONCEPT_NAMES, CONCEPT_NUM_CLASSES, Derm7ptDataset,
                         get_dataloaders, compute_concept_statistics, compute_train_observed_states)
from src import runtime

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from src.metrics import (
    compute_concept_metrics,
    compute_metrics,
    print_concept_metrics_table,
    print_metrics_table,
)
from src.models import SoftJointCBM, get_soft_joint_cbm, load_soft_joint_cbm_state_dict
from src.cbm import make_diagnosis_head
from src.transforms import get_transforms, image_size_config, image_size_hw
from src.protocol import (manifest_fingerprint, load_concept_schema, prepare_training_outputs,
                          save_experiment_results,
                          validate_manifest, validate_concept_schema,
                          concept_schema_name, concept_schema_suffix)
from src.intervention import evaluate_soft_interventions
from src.naming import DEFAULT_BACKBONE, canonical_model
from src.selection import (DEFAULT_CHECKPOINT_METRIC, selection_name, selection_score,
                           metric_from_config, add_selection_argument,
                           configuration_name, run_paths, DEFAULT_THRESHOLD_METRIC,
                           add_threshold_argument, select_threshold, threshold_metric_from_config,
                           threshold_selection_name)

# Soft CBM (Soft Joint CBM): Huấn luyện đồng thời backbone, 7 concept heads và đầu chẩn đoán.

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_SEED = 42
DEFAULT_AUGMENTATION_PRESET = "legacy_letterbox"
DEFAULT_CONCEPT_LOSS_WEIGHT = 1.0
# Quy trình chung: head MLP128 với LR head 1e-3; Linear/LR chung vẫn chọn được qua CLI.
DEFAULT_DIAGNOSIS_HEAD = "mlp128"
DEFAULT_DIAGNOSIS_LR = 1e-3
SOFT_CBM_PROTOCOL = "soft_joint_state_weighted_v1"
DIAGNOSIS_HEADS = {"linear": "Linear(28, 2)", "mlp128": "MLP(28, 128, 2)"}


# Nhãn head theo số trạng thái của schema: 28 với schema gốc (trùng DIAGNOSIS_HEADS), 19 với D4a.
def _head_label(preset, total_states=28):
    labels = {"linear": f"Linear({int(total_states)}, 2)", "mlp128": f"MLP({int(total_states)}, 128, 2)"}
    return labels[preset]


def _config_total_states(config):
    weights = config.get("concept_state_weights") if isinstance(config, dict) else None
    return sum(len(v) for v in weights.values()) if isinstance(weights, dict) else 28


# Tính loss từng concept group bằng criterion tương ứng rồi lấy mean của bảy groups.
def _concept_loss(
    concept_logits: Dict[str, torch.Tensor],
    concept_indices: torch.Tensor,
    criterion: Dict[str, nn.Module],
) -> torch.Tensor:
    # Multi-Head Cross-Entropy: trung bình cộng CE qua 7 concept head
    losses = [
        criterion[c_name](concept_logits[c_name], concept_indices[:, i])
        for i, c_name in enumerate(CONCEPT_NAMES)
    ]
    return torch.stack(losses).mean()


# Cross-entropy concept có trọng số theo state; nhận weights đã tính từ dữ liệu train.
class StateWeightedCE(nn.Module):
    def __init__(self, weights):
        super().__init__()
        weights = torch.as_tensor(weights, dtype=torch.float32)
        if weights.ndim != 1 or not torch.isfinite(weights).all() or torch.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Expected finite non-negative state weights with positive total")
        self.register_buffer("weights", weights)

    def forward(self, logits, targets):
        loss = nn.functional.cross_entropy(logits, targets, weight=self.weights, reduction="none")
        denominator = self.weights[targets].sum()
        return loss.sum() / denominator.clamp_min(torch.finfo(logits.dtype).eps)


# Tạo criterion riêng cho mỗi concept group trên thiết bị đang dùng.
def build_concept_criteria(state_weights, device):
    if not isinstance(state_weights, dict) or set(state_weights) != set(CONCEPT_NAMES):
        raise ValueError("Soft CBM requires train-state weights for all seven concept groups")
    if any(len(state_weights[c]) != CONCEPT_NUM_CLASSES[c] for c in CONCEPT_NAMES):
        raise ValueError("Soft CBM state-weight dimensions differ from concept cardinalities")
    return {c: StateWeightedCE(state_weights[c]).to(device) for c in CONCEPT_NAMES}


# Đọc kiến trúc diagnosis head đã lưu; chấp nhận Linear hoặc MLP128.
def _head_preset(config):
    # Checkpoints Linear cũ có thể chưa khai báo kiến trúc head.
    total_states = _config_total_states(config)
    description = config.get("diagnosis_head", _head_label("linear", total_states))
    for preset in DIAGNOSIS_HEADS:
        if description == _head_label(preset, total_states):
            return preset
    raise ValueError("Unknown Soft CBM diagnosis head in checkpoint")


# Kiểm tra protocol soft joint, kiến trúc head và metadata weights trước khi nạp model.
def validate_soft_cbm_config(config):
    if not isinstance(config, dict):
        raise ValueError("Expected Soft CBM checkpoint configuration")
    if config.get("protocol") != SOFT_CBM_PROTOCOL or config.get("concept_weighting") != "balanced":
        raise ValueError("Checkpoint is not the finalized weighted Soft CBM protocol; train Soft CBM again")
    metric_from_config(config)
    head = _head_preset(config)
    if head == "mlp128" and (config.get("diagnosis_hidden_dim") != 128 or
            config.get("diagnosis_dropout") != 0.3 or config.get("diagnosis_normalization") != "LayerNorm"):
        raise ValueError("Invalid Soft CBM MLP128 diagnosis head specification")
    if "diagnosis_learning_rate" in config and (
            not np.isfinite(config["diagnosis_learning_rate"]) or config["diagnosis_learning_rate"] <= 0):
        raise ValueError("Invalid Soft CBM diagnosis learning rate")
    return build_concept_criteria(config.get("concept_state_weights"), torch.device("cpu"))


# Học joint bằng diagnosis loss + lambda * concept loss; gradient đi qua soft concepts.
def train_one_epoch(
    model: SoftJointCBM,
    loader: DataLoader,
    diag_criterion: nn.Module,
    concept_criterion: Dict[str, nn.Module],
    concept_loss_weight: float,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    # Huấn luyện 1 epoch, trả về (train_loss, train_diag_loss, train_concept_loss, train_diag_bacc)
    model.train()
    total_loss = 0.0
    total_diag_loss = 0.0
    total_concept_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        concept_indices = batch["concept_indices"].to(device)

        optimizer.zero_grad()
        diag_logits, concept_logits, _ = model(images)
        diag_loss = diag_criterion(diag_logits, labels)
        concept_loss = _concept_loss(concept_logits, concept_indices, concept_criterion)
        loss = diag_loss + concept_loss_weight * concept_loss
        loss.backward()
        optimizer.step()

        bs = len(labels)
        total_loss += loss.item() * bs
        total_diag_loss += diag_loss.item() * bs
        total_concept_loss += concept_loss.item() * bs

        preds = torch.argmax(diag_logits, dim=-1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    n = len(all_labels)
    diag_bacc = compute_metrics(all_labels, all_preds)["balanced_accuracy"]
    return total_loss / n, total_diag_loss / n, total_concept_loss / n, diag_bacc


# Eval/no_grad, thu diagnosis và concept metrics cùng probabilities/predictions từng ca.
@torch.no_grad()
def evaluate(
    model: SoftJointCBM,
    loader: DataLoader,
    diag_criterion: nn.Module,
    concept_criterion: Dict[str, nn.Module],
    concept_loss_weight: float,
    device: torch.device,
    threshold: float = 0.5,
    train_observed_states: Dict[str, List[int]] = None,
    concept_outputs: Dict[str, Any] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], float, float, float, np.ndarray, np.ndarray]:
    # Đánh giá đồng thời chẩn đoán (diagnosis) và 7 concept head
    model.eval()
    total_loss = 0.0
    total_diag_loss = 0.0
    total_concept_loss = 0.0
    all_diag_logits: List[torch.Tensor] = []
    all_labels: List[int] = []
    concept_true: Dict[str, List[int]] = {c: [] for c in CONCEPT_NAMES}
    concept_pred: Dict[str, List[int]] = {c: [] for c in CONCEPT_NAMES}
    concept_probabilities = {c: [] for c in CONCEPT_NAMES}

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        concept_indices = batch["concept_indices"].to(device)

        diag_logits, concept_logits, _ = model(images)
        diag_loss = diag_criterion(diag_logits, labels)
        concept_loss = _concept_loss(concept_logits, concept_indices, concept_criterion)
        loss = diag_loss + concept_loss_weight * concept_loss

        bs = len(labels)
        total_loss += loss.item() * bs
        total_diag_loss += diag_loss.item() * bs
        total_concept_loss += concept_loss.item() * bs

        all_diag_logits.append(diag_logits.cpu())
        all_labels.extend(labels.cpu().numpy())

        for i, c_name in enumerate(CONCEPT_NAMES):
            preds_c = torch.argmax(concept_logits[c_name], dim=-1).cpu().numpy()
            concept_pred[c_name].extend(preds_c.tolist())
            concept_true[c_name].extend(concept_indices[:, i].cpu().numpy().tolist())
            if concept_outputs is not None:
                concept_probabilities[c_name].append(torch.softmax(concept_logits[c_name], dim=-1).cpu().numpy())

    all_logits_tensor = torch.cat(all_diag_logits, dim=0)
    probs = torch.softmax(all_logits_tensor, dim=-1)[:, 1].numpy()
    preds = (probs >= threshold).astype(int)
    labels_np = np.array(all_labels, dtype=int)
    n = len(labels_np)

    diag_metrics = compute_metrics(labels_np, preds, probs)
    concept_metrics = compute_concept_metrics(
        {c: np.array(concept_true[c]) for c in CONCEPT_NAMES},
        {c: np.array(concept_pred[c]) for c in CONCEPT_NAMES},
        CONCEPT_NUM_CLASSES,
        train_observed_states=train_observed_states,
    )
    if concept_outputs is not None:
        concept_outputs.update({"true": {c: np.asarray(concept_true[c]) for c in CONCEPT_NAMES},
                                "pred": {c: np.asarray(concept_pred[c]) for c in CONCEPT_NAMES},
                                "probabilities": {c: np.concatenate(concept_probabilities[c]) for c in CONCEPT_NAMES}})
    return (
        diag_metrics,
        concept_metrics,
        total_loss / n,
        total_diag_loss / n,
        total_concept_loss / n,
        preds,
        probs,
    )


# Tạo checkpoint/JSON paths theo cấu hình, seed và tiêu chí chọn checkpoint.
def _default_paths(
    seed: int = DEFAULT_SEED,
    augmentation_preset: str = DEFAULT_AUGMENTATION_PRESET,
    concept_loss_weight: float = DEFAULT_CONCEPT_LOSS_WEIGHT,
    checkpoint_metric: str = DEFAULT_CHECKPOINT_METRIC,
    diagnosis_head: str = DEFAULT_DIAGNOSIS_HEAD,
    diagnosis_lr: float = DEFAULT_DIAGNOSIS_LR,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    threshold_metric: str = DEFAULT_THRESHOLD_METRIC,
    image_size=224,
    concept_schema=None,
) -> Tuple[str, str]:
    if diagnosis_head not in DIAGNOSIS_HEADS:
        raise ValueError("Unknown Soft CBM diagnosis head")
    configuration = configuration_name(diagnosis_head, epochs, diagnosis_lr,
        augmentation_preset=augmentation_preset, concept_loss_weight=concept_loss_weight,
        lr=lr, weight_decay=weight_decay, image_size=image_size) + concept_schema_suffix(concept_schema)
    return run_paths(PROJECT_ROOT, checkpoint_metric, "soft_cbm", configuration, seed,
                     threshold_metric=threshold_metric, backbone=DEFAULT_BACKBONE)


# Suy ra đúng checkpoint/JSON từ config đã chốt, thay vì dùng default mới để đoán.
def _config_paths(config):
    return _default_paths(
        config["seed"], config["augmentation_preset"], config["concept_loss_weight"],
        metric_from_config(config), diagnosis_head=_head_preset(config),
        diagnosis_lr=config.get("diagnosis_learning_rate"),
        epochs=config.get("epochs", DEFAULT_EPOCHS), lr=config.get("learning_rate", DEFAULT_LR),
        weight_decay=config.get("weight_decay", DEFAULT_WEIGHT_DECAY),
        threshold_metric=threshold_metric_from_config(config),
        image_size=config.get("target_size", 224),
        concept_schema=config.get("concept_schema_name"),
    )


# Tạo AdamW chung LR hoặc tách diagnosis head thành parameter group có LR riêng.
def _build_optimizer(model, lr, weight_decay, diagnosis_lr=None):
    # Giữ nguyên optimizer của baseline khi không yêu cầu LR head riêng.
    if diagnosis_lr is None:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if not np.isfinite(diagnosis_lr) or diagnosis_lr <= 0:
        raise ValueError("diagnosis learning rate must be positive and finite")
    diagnosis_parameters = list(model.diagnosis_head.parameters())
    diagnosis_ids = {id(parameter) for parameter in diagnosis_parameters}
    concept_parameters = [parameter for parameter in model.parameters()
                          if id(parameter) not in diagnosis_ids]
    return torch.optim.AdamW([
        {"params": concept_parameters, "lr": lr, "name": "concept_predictor"},
        {"params": diagnosis_parameters, "lr": diagnosis_lr, "name": "diagnosis_head"},
    ], weight_decay=weight_decay)


# Train/validation, chọn best epoch rồi chọn ngưỡng, lưu cả config/history/predictions.
def run_soft_cbm_experiment(
    manifest_path: str = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    concept_loss_weight: float = DEFAULT_CONCEPT_LOSS_WEIGHT,
    seed: int = DEFAULT_SEED,
    device_name: str = "auto",
    checkpoint_path: str = None,
    results_path: str = None,
    save_results: bool = True,
    augmentation_preset: str = DEFAULT_AUGMENTATION_PRESET,
    overwrite: bool = False,
    num_workers: int = 0,
    checkpoint_metric: str = DEFAULT_CHECKPOINT_METRIC,
    diagnosis_head: str = DEFAULT_DIAGNOSIS_HEAD,
    diagnosis_lr: float = DEFAULT_DIAGNOSIS_LR,
    threshold_metric: str = DEFAULT_THRESHOLD_METRIC,
    image_size=224,
) -> Dict[str, Any]:
    image_hw = image_size_hw(image_size)
    if epochs < 1 or batch_size < 1 or num_workers < 0:
        raise ValueError("epochs/batch_size must be positive and num_workers non-negative")
    selection_name(checkpoint_metric)
    threshold_selection_name(threshold_metric)
    if not np.isfinite(concept_loss_weight) or concept_loss_weight < 0:
        raise ValueError("concept_loss_weight must be finite and non-negative")
    if not np.isfinite(lr) or lr <= 0 or not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    if diagnosis_head not in DIAGNOSIS_HEADS:
        raise ValueError("Unknown Soft CBM diagnosis head")
    if diagnosis_lr is not None and (not np.isfinite(diagnosis_lr) or diagnosis_lr <= 0):
        raise ValueError("diagnosis learning rate must be positive and finite")
    runtime.set_seed(seed)
    device = runtime.get_device(device_name)

    if manifest_path is None:
        manifest_path = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
    manifest_path = os.path.abspath(manifest_path)
    schema_name = concept_schema_name(manifest_path, PROJECT_ROOT)
    checkpoint_path, results_path = prepare_training_outputs(
        _default_paths(seed, augmentation_preset, concept_loss_weight, checkpoint_metric,
                       diagnosis_head, diagnosis_lr, epochs, lr, weight_decay, threshold_metric, image_hw,
                       schema_name),
        checkpoint_path, results_path, overwrite, save_results,
    )

    runtime.prepare_output(checkpoint_path, overwrite)
    if save_results:
        runtime.prepare_output(results_path, overwrite)

    print(f"Khởi động Soft CBM (EfficientNet-B0 -> 7 concept heads -> g {diagnosis_head})")
    print(f"Thiết bị sử dụng: {device} | Random seed: {seed}")
    print(
        f"Cấu hình: Epochs={epochs}, Batch Size={batch_size}, LR={lr}, "
        f"Diagnosis LR={diagnosis_lr if diagnosis_lr is not None else lr}, Weight Decay={weight_decay}, "
        f"Augmentation={augmentation_preset}, Concept Loss Weight (lambda)={concept_loss_weight}"
    )

    bundle = get_dataloaders(
        manifest_path=manifest_path,
        project_root=PROJECT_ROOT,
        batch_size=batch_size,
        num_workers=num_workers,
        target_size=image_hw,
        augment_train=True,
        augmentation_preset=augmentation_preset,
        include_test=False,
    )

    train_loader = bundle["train"]
    valid_loader = bundle["valid"]
    class_weights = bundle["class_weights"].to(device)

    assert len(bundle["datasets"]["train"]) == 413, "Train split phải có đúng 413 mẫu"
    assert len(bundle["datasets"]["valid"]) == 203, "Valid split phải có đúng 203 mẫu"

    label_mapping = bundle["datasets"]["train"].label_mapping
    schema = load_concept_schema(manifest_path)
    train_observed_states = compute_train_observed_states(bundle["datasets"]["train"].df, label_mapping)
    statistics = compute_concept_statistics(manifest_path, PROJECT_ROOT)
    state_weights = {c: statistics[c]["weights"] for c in CONCEPT_NAMES}

    model = get_soft_joint_cbm(CONCEPT_NAMES, CONCEPT_NUM_CLASSES, num_classes=2, pretrained=True,
                              diagnosis_head=diagnosis_head)
    model.to(device)

    diag_criterion = nn.CrossEntropyLoss(weight=class_weights)
    concept_criterion = build_concept_criteria(state_weights, device)
    optimizer = _build_optimizer(model, lr, weight_decay, diagnosis_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_score = -1.0
    best_epoch = -1
    history: List[Dict[str, float]] = []
    config = {
        "model": "soft_cbm",
        "protocol": SOFT_CBM_PROTOCOL,
        "backbone": "EfficientNet-B0",
        "pretrained_weights": "EfficientNet_B0_Weights.DEFAULT",
        "diagnosis_head": _head_label(diagnosis_head, schema["total_states"]),
        "concept_heads": "Linear(1280, Ki) per concept",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "concept_loss_weight": concept_loss_weight,
        "seed": seed,
        "device": str(device),
        "diagnosis_loss": "Weighted_CrossEntropy",
        "concept_loss": "Multi-Head_CrossEntropy_balanced_mean",
        "concept_weighting": "balanced",
        "concept_state_weights": state_weights,
        "concept_train_statistics": statistics,
        "dropout": 0.2,
        "class_weights": [float(x) for x in class_weights.cpu().tolist()],
        "augmentation_preset": augmentation_preset,
        "target_size": image_size_config(image_hw),
        "num_workers": num_workers,
        "scheduler": "CosineAnnealingLR(eta_min=1e-6)",
        "checkpoint_selection": selection_name(checkpoint_metric),
        "threshold_selection": threshold_selection_name(threshold_metric),
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "numpy_version": str(np.__version__),
        "pandas_version": str(pd.__version__),
        "scikit_learn_version": str(sklearn.__version__),
        "platform": platform.platform(),
    }
    if diagnosis_lr is not None:
        config["diagnosis_learning_rate"] = diagnosis_lr
        config["optimizer_parameter_groups"] = "concept_predictor_and_diagnosis_head"
    if diagnosis_head == "mlp128":
        config.update(diagnosis_hidden_dim=128, diagnosis_dropout=0.3,
                      diagnosis_normalization="LayerNorm")
    if schema_name:
        config["concept_schema_name"] = schema_name
    manifest_sha256 = runtime.file_sha256(manifest_path)

    print("\n--- Bắt đầu huấn luyện Soft CBM (Soft Joint CBM) ---")
    for epoch in range(1, epochs + 1):
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        train_loss, train_diag_loss, train_concept_loss, train_bacc = train_one_epoch(
            model, train_loader, diag_criterion, concept_criterion, concept_loss_weight, optimizer, device
        )
        val_metrics, val_concept_metrics, val_loss, val_diag_loss, val_concept_loss, _, _ = evaluate(
            model, valid_loader, diag_criterion, concept_criterion, concept_loss_weight, device,
            train_observed_states=train_observed_states,
        )
        scheduler.step()

        val_bacc = val_metrics["balanced_accuracy"]
        val_f1_macro = val_metrics["f1_macro"]
        val_f1_mel = val_metrics["f1_melanoma"]
        val_auc = val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else 0.0
        val_concept_f1 = val_concept_metrics["overall_f1_macro_all_defined"]
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_diagnosis_loss": train_diag_loss,
            "train_concept_loss": train_concept_loss,
            "train_diagnosis_balanced_accuracy": train_bacc,
            "validation_loss": val_loss,
            "validation_diagnosis_loss": val_diag_loss,
            "validation_concept_loss": val_concept_loss,
            "validation_balanced_accuracy_at_0.5": val_bacc,
            "validation_f1_macro_at_0.5": val_f1_macro,
            "validation_f1_melanoma_at_0.5": val_f1_mel,
            "validation_roc_auc": val_auc,
            "validation_pr_auc": val_metrics["pr_auc"],
            "validation_concept_f1_macro_all_defined": val_concept_f1,
            "learning_rates": learning_rates,
        })

        # Diagnosis Macro F1 chọn checkpoint; concept F1 và BAcc là metrics riêng.
        score = selection_score(val_metrics, checkpoint_metric)
        is_best = score > best_score
        best_marker = " [BEST]" if is_best else ""

        if is_best:
            best_score = score
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_balanced_acc": val_bacc,
                "val_f1_macro": val_f1_macro,
                "selection_score": score,
                "val_metrics": val_metrics,
                "val_concept_metrics": val_concept_metrics,
                "config": config,
                "manifest_sha256": manifest_sha256,
                "manifest_fingerprint": manifest_fingerprint(manifest_path),
                "concept_schema": schema,
            }, checkpoint_path)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} (Diag: {train_diag_loss:.4f}, Concept: {train_concept_loss:.4f}, BAcc: {train_bacc * 100:.2f}%) | "
            f"Valid Loss: {val_loss:.4f} | Valid BAcc: {val_bacc * 100:.2f}% | Valid F1-Mel: {val_f1_mel:.4f} | "
            f"Valid Macro F1: {val_f1_macro:.4f} | "
            f"Valid AUC: {val_auc:.4f} | Valid Concept F1: {val_concept_f1:.4f}{best_marker}"
        )

    print(f"\nHuấn luyện hoàn tất! Checkpoint tốt nhất tại Epoch {best_epoch:02d} (Valid {checkpoint_metric}@0.5: {best_score:.4f})")
    print(f"Checkpoint đã lưu tại: {checkpoint_path}")

    # Chọn threshold trên validation của checkpoint đã chọn, không đọc test.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    load_soft_joint_cbm_state_dict(model, checkpoint["model_state_dict"])
    val_concept_outputs = {}
    (
        val_metrics_at_05, val_concept_metrics_at_05, val_loss_best, _, _, _, val_probs_best,
    ) = evaluate(
        model, valid_loader, diag_criterion, concept_criterion, concept_loss_weight, device,
        train_observed_states=train_observed_states,
        concept_outputs=val_concept_outputs,
    )
    val_true = bundle["datasets"]["valid"].df["diagnosis_binary"].to_numpy(dtype=int)
    threshold, _ = select_threshold(val_true, val_probs_best, threshold_metric)
    val_preds_best = (val_probs_best >= threshold).astype(int)
    val_metrics_best = compute_metrics(val_true, val_preds_best, val_probs_best)
    checkpoint["decision_threshold"] = threshold
    checkpoint["validation_metrics_at_threshold"] = val_metrics_best
    torch.save(checkpoint, checkpoint_path)

    valid_table = print_metrics_table(
        val_metrics_best, title=f"Validation Set (Epoch {best_epoch}, threshold={threshold:.6f})"
    )
    concept_table = print_concept_metrics_table(val_concept_metrics_at_05)
    val_sample_preds = runtime.sample_prediction_records(
        bundle["datasets"]["valid"].df, val_preds_best, val_probs_best, val_concept_outputs
    )

    summary_text = (
        f"Soft CBM (EfficientNet-B0 -> 7 concept heads -> g {config['diagnosis_head']})\n"
        f"Best Epoch: {best_epoch}/{epochs}\n"
        f"Training samples: 413, Valid: 203\n"
        f"Threshold selected on validation: {threshold:.6f}\n\n"
        f"{valid_table}\n\n{concept_table}"
    )

    results = {
        "model": "soft_cbm",
        "mode": "train_validation",
        "best_epoch": int(best_epoch),
        "hyperparameters": config,
        "manifest_sha256": manifest_sha256,
        "manifest_fingerprint": manifest_fingerprint(manifest_path),
        "concept_schema": schema,
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_sha256": runtime.file_sha256(checkpoint_path),
        "decision_threshold": threshold,
        "summary": summary_text,
        "validation_metrics_at_0.5": val_metrics_at_05,
        "validation_metrics": val_metrics_best,
        "validation_concept_metrics": val_concept_metrics_at_05,
        "validation_loss": val_loss_best,
        "validation_predictions": val_sample_preds,
        "train_observed_states": train_observed_states,
        "history": history,
    }

    if save_results:
        runtime.save_json(results_path, results)
        print(f"Kết quả chi tiết đã lưu tại: {results_path}")

    return results


# Nạp frozen soft CBM/head/ngưỡng, chấm official test và giữ nguyên quyết định validation.
def evaluate_soft_cbm_test(
    checkpoint_path: str,
    manifest_path: str = None,
    device_name: str = "auto",
    results_path: str = None,
    save_results: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    if manifest_path is None:
        manifest_path = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
    manifest_path = os.path.abspath(manifest_path)
    device = runtime.get_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "decision_threshold" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Checkpoint has no frozen validation threshold/config; use a newly trained checkpoint")
    validate_manifest(checkpoint, manifest_path)
    schema = validate_concept_schema(checkpoint, manifest_path)

    config = checkpoint["config"]
    validate_soft_cbm_config(config)
    threshold = float(checkpoint["decision_threshold"])
    if not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold in checkpoint")
    if results_path is None:
        _, results_path = _config_paths(config)
    if save_results:
        runtime.prepare_output(results_path, overwrite)

    train_df = pd.read_csv(manifest_path, dtype={"is_inconsistent_profile": bool})
    train_df = train_df[train_df["split"] == "train"].reset_index(drop=True)
    label_mapping = schema["label_mapping"]
    train_observed_states = compute_train_observed_states(train_df, label_mapping)

    test_dataset = Derm7ptDataset(
        manifest_path=manifest_path,
        project_root=PROJECT_ROOT,
        split="test",
        label_mapping=label_mapping,
        transform=get_transforms("valid", target_size=config["target_size"], augment=False),
    )
    assert len(test_dataset) == 395, "Test split phải có đúng 395 mẫu"
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    model = get_soft_joint_cbm(schema["concept_names"], schema["concept_num_classes"], num_classes=2,
                              pretrained=False, dropout=config.get("dropout", 0.2),
                              diagnosis_head=_head_preset(config)).to(device)
    load_soft_joint_cbm_state_dict(model, checkpoint["model_state_dict"])
    weights = torch.tensor(config["class_weights"], dtype=torch.float32, device=device)
    diag_criterion = nn.CrossEntropyLoss(weight=weights)
    concept_criterion = build_concept_criteria(config.get("concept_state_weights"), device)
    test_concept_outputs = {}

    test_metrics, test_concept_metrics, test_loss, _, _, test_preds, test_probs = evaluate(
        model, test_loader, diag_criterion, concept_criterion, config["concept_loss_weight"], device,
        threshold=threshold, train_observed_states=train_observed_states,
        concept_outputs=test_concept_outputs,
    )
    test_table = print_metrics_table(test_metrics, title=f"Test Set (threshold={threshold:.6f})")
    concept_table = print_concept_metrics_table(test_concept_metrics)

    results = {
        "model": "soft_cbm",
        "mode": "final_test",
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_sha256": runtime.file_sha256(checkpoint_path),
        "best_epoch": int(checkpoint["epoch"]),
        "hyperparameters": config,
        "manifest_sha256": checkpoint["manifest_sha256"],
        "manifest_fingerprint": manifest_fingerprint(manifest_path),
        "concept_schema": schema,
        "decision_threshold": threshold,
        "summary": f"{test_table}\n\n{concept_table}",
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "test_concept_metrics": test_concept_metrics,
        "test_predictions": runtime.sample_prediction_records(test_dataset.df, test_preds, test_probs,
                                                       test_concept_outputs or None),
    }
    if save_results:
        save_experiment_results(results_path, test=results, overwrite=overwrite)
        print(f"Kết quả test đã lưu tại: {results_path}")
    return results


# Ghép đúng checkpoint/export, thay soft groups bằng GT one-hot qua frozen head, lưu kết quả.
def run_soft_cbm_intervention(checkpoint_path, predictions_path, manifest_path=None,
                        results_path=None, overwrite=False):
    manifest_path = os.path.abspath(manifest_path or str(Path(PROJECT_ROOT) / "data/manifest.csv"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_manifest(checkpoint, manifest_path)
    schema = validate_concept_schema(checkpoint, manifest_path)
    validate_soft_cbm_config(checkpoint["config"])
    export = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    validate_manifest(export, manifest_path)
    validate_concept_schema(export, manifest_path)
    if export.get("mode") not in {"final_test", "train_validation_test"} or canonical_model(export.get("model")) != "soft_cbm":
        raise ValueError("Expected frozen Soft CBM test predictions")
    if export.get("hyperparameters") != checkpoint["config"]:
        raise ValueError("Prediction training config differs from checkpoint")
    if export.get("checkpoint_sha256") != runtime.file_sha256(checkpoint_path):
        raise ValueError("Prediction export does not belong to this checkpoint; re-export Soft CBM test")
    threshold = float(checkpoint["decision_threshold"])
    if export.get("decision_threshold") != threshold:
        raise ValueError("Prediction threshold differs from frozen checkpoint")
    test = pd.read_csv(manifest_path)
    test = test[test.split == "test"].set_index("case_num")
    rows = export["test_predictions"]
    ids = [int(row["case_num"]) for row in rows]
    if len(ids) != len(test) or set(ids) != set(test.index) or len(set(ids)) != len(ids):
        raise ValueError("Prediction case IDs differ from the official test split")
    probabilities, true_indices, labels = [], [], []
    for row in rows:
        case = test.loc[row["case_num"]]
        if row["y_true"] != int(case.diagnosis_binary):
            raise ValueError("Prediction labels differ from manifest")
        targets = [schema["label_mapping"][c][case[c]] for c in schema["concept_names"]]
        if "concept_probabilities" not in row or "concept_true" not in row:
            raise ValueError("Export has no per-case concepts; re-export test with the updated Soft CBM runner")
        if targets != [row["concept_true"][c] for c in schema["concept_names"]]:
            raise ValueError("Prediction concept targets differ from manifest")
        if any(len(row["concept_probabilities"][c]) != schema["concept_num_classes"][c]
               for c in schema["concept_names"]):
            raise ValueError("Export concept probability block dimensions differ from schema")
        probabilities.append([p for c in schema["concept_names"] for p in row["concept_probabilities"][c]])
        true_indices.append(targets)
        labels.append(row["y_true"])
    head = make_diagnosis_head(schema["total_states"], 2, _head_preset(checkpoint["config"]))
    state = checkpoint["model_state_dict"]
    head.load_state_dict({name.removeprefix("diagnosis_head."): value
                          for name, value in state.items() if name.startswith("diagnosis_head.")})
    result = evaluate_soft_interventions(head, probabilities, true_indices, labels, schema, threshold, ids)
    if not np.allclose(result["baseline_y_prob"], [row["y_prob"] for row in rows], atol=1e-6, rtol=1e-5):
        raise ValueError("Export concept probabilities do not reproduce baseline diagnosis")
    if result["baseline_y_pred"] != [row["y_pred"] for row in rows]:
        raise ValueError("Export diagnosis predictions do not use the frozen threshold")
    result.update({"model": "soft_cbm", "mode": "intervention_analysis",
                   "hyperparameters": checkpoint["config"],
                   "concept_schema": schema, "manifest_fingerprint": manifest_fingerprint(manifest_path),
                   "checkpoint_sha256": runtime.file_sha256(checkpoint_path),
                   "checkpoint_epoch": int(checkpoint["epoch"]),
                   "evaluation_environment": {"device": "cpu", "platform": platform.platform(),
                                              "torch_version": str(torch.__version__),
                                              "numpy_version": np.__version__,
                                              "scikit_learn_version": sklearn.__version__},
                   "checkpoint_path": os.path.abspath(checkpoint_path),
                   "predictions_path": os.path.abspath(predictions_path)})
    results_path = results_path or str(Path(predictions_path).parent / "intervention" / Path(predictions_path).name)
    if Path(results_path).resolve() in {Path(checkpoint_path).resolve(), Path(predictions_path).resolve(), Path(manifest_path).resolve()}:
        raise ValueError("Intervention output must not overwrite its inputs")
    runtime.prepare_output(results_path, overwrite)
    runtime.save_json(results_path, result)
    print(f"Saved {result['num_subsets']} intervention subsets to {results_path}")
    return result


# Đọc mode train/test/intervention; --skip_test dừng sau validation, không chọn gì trên test.
def main(argv=None):
    parser = argparse.ArgumentParser(description="Soft CBM (weighted Soft Joint CBM): train → validation → frozen test, or --skip_test pilot; intervention")
    parser.add_argument("--mode", choices=["train", "test", "intervention"], default="train", help="train: train rồi tự động test; test: đánh giá checkpoint; intervention: can thiệp concepts")
    parser.add_argument("--skip_test", action="store_true", help="Pilot: chỉ train/validation; dùng với --mode train")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--diagnosis_lr", type=float, default=DEFAULT_DIAGNOSIS_LR,
                        help=f"LR riêng cho diagnosis head (mặc định: {DEFAULT_DIAGNOSIS_LR})")
    parser.add_argument("--diagnosis_head", choices=list(DIAGNOSIS_HEADS), default=DEFAULT_DIAGNOSIS_HEAD,
                        help="mlp128 (mặc định): Linear/LayerNorm/ReLU/Dropout/Linear; linear: head tuyến tính")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument(
        "--concept_loss_weight", type=float, default=DEFAULT_CONCEPT_LOSS_WEIGHT,
        help="Trọng số lambda cho concept loss trong L = L_diagnosis + lambda * L_concept",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị: auto, mps, cuda, cpu")
    parser.add_argument(
        "--augmentation_preset",
        choices=["legacy_letterbox", "comparison"],
        default=DEFAULT_AUGMENTATION_PRESET,
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_size", default="224", help="N (vuông) hoặc CAOxRỘNG, ví dụ 512x768")
    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--results_path", type=str, default=None, help="File JSON chứa validation/test; kết quả can thiệp khi --mode intervention")
    parser.add_argument("--predictions_path", type=str, default=None, help="JSON kết quả Soft CBM chứa test_predictions để phân tích intervention")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    add_selection_argument(parser)
    add_threshold_argument(parser)
    args = parser.parse_args(argv)
    if args.skip_test and args.mode != "train":
        parser.error("--skip_test is only valid with --mode train")
    defaults = _default_paths(
        args.seed, args.augmentation_preset, args.concept_loss_weight, args.checkpoint_metric,
        args.diagnosis_head, args.diagnosis_lr, args.epochs, args.lr, args.weight_decay,
        args.threshold_metric, args.image_size,
        concept_schema_name(args.manifest_path or os.path.join(PROJECT_ROOT, "data", "manifest.csv"), PROJECT_ROOT),
    )

    if args.mode == "train":
        checkpoint_path, results_path = prepare_training_outputs(
            defaults,
            args.checkpoint_path, args.results_path,
            args.overwrite, not args.no_save,
        )
        validation = run_soft_cbm_experiment(
            manifest_path=args.manifest_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            concept_loss_weight=args.concept_loss_weight,
            seed=args.seed,
            device_name=args.device,
            checkpoint_path=checkpoint_path,
            save_results=False,
            augmentation_preset=args.augmentation_preset,
            overwrite=args.overwrite,
            num_workers=args.num_workers,
            checkpoint_metric=args.checkpoint_metric,
            diagnosis_head=args.diagnosis_head,
            diagnosis_lr=args.diagnosis_lr,
            threshold_metric=args.threshold_metric,
            image_size=args.image_size,
        )
        if not args.no_save:
            save_experiment_results(results_path, validation=validation, overwrite=args.overwrite)
        if args.skip_test:
            return validation
        test = evaluate_soft_cbm_test(
            checkpoint_path=checkpoint_path, manifest_path=args.manifest_path,
            device_name=args.device,
            save_results=False, overwrite=args.overwrite,
        )
        if not args.no_save:
            return save_experiment_results(results_path, validation=validation, test=test, overwrite=True)
        return {**validation, **test, "mode": "train_validation_test"}
    elif args.mode == "test":
        checkpoint_path = args.checkpoint_path or defaults[0]
        results_path = args.results_path
        if results_path is None:
            config = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["config"]
            results_path = _config_paths(config)[1]
        if not args.no_save:
            runtime.prepare_output(results_path, args.overwrite)
        test = evaluate_soft_cbm_test(
            checkpoint_path=checkpoint_path,
            manifest_path=args.manifest_path,
            device_name=args.device,
            save_results=False,
            overwrite=args.overwrite,
        )
        if not args.no_save:
            return save_experiment_results(results_path, test=test, overwrite=args.overwrite)
        return test
    else:
        if args.no_save:
            parser.error("--no_save is not supported for intervention mode")
        checkpoint_path = args.checkpoint_path or defaults[0]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        validate_soft_cbm_config(config)
        predictions_path = args.predictions_path or _config_paths(config)[1]
        return run_soft_cbm_intervention(checkpoint_path, predictions_path, args.manifest_path,
                                   args.results_path, args.overwrite)


if __name__ == "__main__":
    main()
