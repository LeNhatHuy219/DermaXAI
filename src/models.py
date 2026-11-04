from typing import Dict, List, Mapping

import torch
import torch.nn as nn
import torchvision.models as models

# Các CBM được định nghĩa trong cbm.py; giữ import qua src.models cho code đang dùng.
from src.cbm import SoftJointCBM, make_diagnosis_head

from src.naming import BACKBONES, DEFAULT_BACKBONE

# Head của Black-box nhận 1280 chiều đặc trưng ảnh; mô tả được lưu vào config checkpoint.
BLACKBOX_HEADS = {"linear": "Linear(1280, 2)", "mlp128": "MLP(1280, 128, 2)"}


# Dựng phần trích xuất đặc trưng (features + pooling) của backbone; thêm backbone mới bằng một nhánh tại đây.
def build_backbone(name: str = DEFAULT_BACKBONE, pretrained: bool = True):
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone {name}; supported: {sorted(BACKBONES)}")
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    network = models.efficientnet_b0(weights=weights)
    return nn.Sequential(network.features, network.avgpool), network.classifier[1].in_features  # 1280 chiều


class BlackBoxClassifier(nn.Module):
    # Mô hình phân loại trực tiếp từ ảnh sang nhãn chẩn đoán (black-box); tên model là id backbone
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
        head: str = "linear",
        backbone: str = DEFAULT_BACKBONE,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone

        # Backbone với trọng số ImageNet; giữ phần trích xuất đặc trưng (features + pooling)
        self.features, in_features = build_backbone(backbone, pretrained)

        # Lớp phân loại chẩn đoán nhị phân (0: Non-Melanoma, 1: Melanoma)
        # Linear là baseline; mlp128 dùng cùng head MLP với Soft CBM/Hard CBM để so sánh cùng cấu hình.
        if head not in BLACKBOX_HEADS:
            raise ValueError("Unknown Black-box classifier head; expected linear or mlp128")
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes) if head == "linear"
            else make_diagnosis_head(in_features, num_classes, head),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # Trích xuất vector đặc trưng bậc cao (1280 chiều) từ ảnh soi da
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dự đoán trực tiếp logits chẩn đoán từ ảnh đầu vào
        feat = self.extract_features(x)
        logits = self.classifier(feat)
        return logits


def load_blackbox_state_dict(model: BlackBoxClassifier, state_dict: Mapping[str, torch.Tensor]) -> None:
    # Nạp trọng số Black-box hiện tại hoặc kế thừa; loại bỏ tiền tố backbone cũ nếu có.
    if any(key.startswith("backbone.") for key in state_dict):
        state_dict = {key: value for key, value in state_dict.items() if not key.startswith("backbone.")}
    model.load_state_dict(state_dict, strict=True)


def load_soft_joint_cbm_state_dict(model: SoftJointCBM, state_dict: Mapping[str, torch.Tensor]) -> None:
    # Nạp trọng số checkpoint Soft CBM (giữ đối xứng với load_blackbox_state_dict của Black-box).
    model.load_state_dict(state_dict, strict=True)


def get_soft_joint_cbm(
    concept_names: List[str],
    concept_num_classes: Dict[str, int],
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.2,
    diagnosis_head: str = "linear",
) -> SoftJointCBM:
    # Khởi tạo mô hình Soft CBM (Soft Joint CBM)
    return SoftJointCBM(
        concept_names=concept_names,
        concept_num_classes=concept_num_classes,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        diagnosis_head=diagnosis_head,
    )
