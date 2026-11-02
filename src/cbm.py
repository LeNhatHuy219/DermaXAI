# Định nghĩa chung các Concept Bottleneck Models: Soft CBM và Hard CBM (straight-through).
# Cả ba dùng EfficientNet-B0; mặc định head Linear, khác bottleneck/gradient.
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def make_diagnosis_head(input_dim: int, num_classes: int = 2,
                        architecture: str = "linear") -> nn.Module:
    # Head chỉ nhận concept: Linear là baseline; MLP128 thử tương tác phi tuyến.
    # MLP theo cấu trúc Nápoles: Linear -> LayerNorm -> ReLU -> dropout -> Linear.
    if architecture == "linear":
        return nn.Linear(input_dim, num_classes)
    if architecture == "mlp128":
        return nn.Sequential(nn.Linear(input_dim, 128), nn.LayerNorm(128),
                             nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes))
    raise ValueError("Unknown diagnosis head; expected linear or mlp128")


# Soft CBM: ảnh -> xác suất concept mềm -> logits chẩn đoán; huấn luyện joint.
class SoftJointCBM(nn.Module):
    def __init__(
        self,
        concept_names: List[str],
        concept_num_classes: Dict[str, int],
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
        diagnosis_head: str = "linear",
    ):
        super().__init__()
        self.concept_names = list(concept_names)
        self.concept_num_classes = dict(concept_num_classes)
        self.total_concept_states = sum(self.concept_num_classes[name] for name in self.concept_names)
        self.num_classes = num_classes

        # Backbone ImageNet dùng chung; Derm7pt có 7 concept groups, tổng 28 states.
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features  # 1280 chiều

        # Bỏ classifier gốc, giữ features và global average pooling để lấy vector 1280D.
        self.features = nn.Sequential(
            backbone.features,
            backbone.avgpool,
        )
        self.dropout = nn.Dropout(p=dropout)

        # Mỗi khái niệm lâm sàng có một concept head Linear riêng: 1280 -> Ki
        self.concept_heads = nn.ModuleDict({
            name: nn.Linear(in_features, self.concept_num_classes[name])
            for name in self.concept_names
        })

        # Diagnosis head chỉ nhận bottleneck, không nhận trực tiếp đặc trưng ảnh.
        # Soft CBM chọn Linear/MLP trên xác suất mềm; Hard CBM dùng cùng head trên one-hot.
        self.diagnosis_head = make_diagnosis_head(self.total_concept_states, num_classes, diagnosis_head)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        # Chuyển output pooling (batch, 1280, 1, 1) thành (batch, 1280).
        feat = torch.flatten(feat, 1)
        return feat

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        feat = self.dropout(self.extract_features(x))

        # Logits thô từng concept head (dùng cho Multi-Head Cross-Entropy)
        concept_logits: Dict[str, torch.Tensor] = {
            name: self.concept_heads[name](feat) for name in self.concept_names
        }

        # Softmax riêng mỗi group để tổng xác suất của group bằng 1.
        # Nối các groups thành bottleneck (batch, 28) trong cấu hình Derm7pt.
        concept_vector = torch.cat(
            [torch.softmax(concept_logits[name], dim=-1) for name in self.concept_names],
            dim=-1,
        )

        # Softmax khả vi: diagnosis loss truyền gradient về concept heads/backbone.
        diag_logits = self.diagnosis_head(concept_vector)
        # Trả thêm logits concept cho concept loss và vector cho đánh giá/intervention.
        return diag_logits, concept_logits, concept_vector


def hard_categorical(logits: torch.Tensor, straight_through: bool = False) -> torch.Tensor:
    # Chọn state xác suất cao nhất của một group và mã hóa thành one-hot.
    probabilities = logits.softmax(dim=-1)
    hard = F.one_hot(probabilities.argmax(dim=-1), logits.shape[-1]).to(logits.dtype)

    # Straight-through: forward dùng one-hot argmax, backward truyền gradient qua softmax.
    return hard + (probabilities - probabilities.detach()) if straight_through else hard


# Hard CBM: CBM bottleneck one-hot huấn luyện joint bằng straight-through estimator.
class HardJointCBM(SoftJointCBM):
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        features = self.dropout(self.extract_features(x))
        logits = {name: self.concept_heads[name](features) for name in self.concept_names}

        # Ghép one-hot 7 nhóm concepts thành vector 28 chiều đưa vào diagnosis head.
        vector = torch.cat([hard_categorical(logits[name], self.training)
                            for name in self.concept_names], dim=-1)
        return self.diagnosis_head(vector), logits, vector


def get_hard_joint_cbm(concept_names: List[str], concept_num_classes: Dict[str, int],
                       num_classes: int = 2, pretrained: bool = True,
                       dropout: float = 0.2, diagnosis_head: str = "linear") -> HardJointCBM:
    # Factory Hard CBM giữ nguyên giao diện khởi tạo để runners/tests dùng chung.
    return HardJointCBM(concept_names, concept_num_classes, num_classes, pretrained, dropout, diagnosis_head)
