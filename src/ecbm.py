# ECBM: Categorical ECBM (Xu et al., ICLR 2024), cài lại theo mã gốc xmed-lab/ECBM cho 7 nhóm concept Derm7pt.
# Ba năng lượng: E_class(x, y) ảnh–nhãn, E_concept(x, c) ảnh–concept, E_global(c, y) concept–nhãn.
# Năng lượng thấp nghĩa là tương thích cao; mỗi nhóm concept là một biến categorical có K_g trạng thái.
import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# Cổng nhân embedding của mã gốc: relu(h + h * normalize(e)).
def _gate(hidden: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
    return F.relu(hidden * (1.0 + F.normalize(embedding, dim=-1, eps=1e-12)))


class CategoricalECBM(nn.Module):
    def __init__(
        self,
        concept_names: Sequence[str],
        concept_num_classes: Dict[str, int],
        num_classes: int = 2,
        pretrained: bool = True,
        hidden_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.concept_names = list(concept_names)
        self.state_counts = [int(concept_num_classes[name]) for name in self.concept_names]
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features  # 1280 chiều
        self.features = nn.Sequential(backbone.features, backbone.avgpool)
        self.dropout = nn.Dropout(p=dropout)

        # Embedding nhãn và từng trạng thái concept, khởi tạo N(0, 1) như mã gốc.
        # Một embedding cho mỗi trạng thái, dùng chung cho E_concept và E_global.
        self.class_embedding = nn.Parameter(torch.randn(num_classes, hidden_dim))
        self.concept_embeddings = nn.ParameterList(
            [nn.Parameter(torch.randn(count, hidden_dim)) for count in self.state_counts])

        # Mã gốc: xy_fc1/classifier_xy, xc_fc1 dùng chung/classifier_xc từng nhóm, concept_proj/classifier_cy.
        self.class_projection = nn.Linear(in_features, hidden_dim)
        self.class_output = nn.Linear(hidden_dim, 1)
        self.concept_projection = nn.Linear(in_features, hidden_dim)
        self.concept_outputs = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in self.state_counts])
        self.global_projection = nn.Linear(len(self.state_counts) * hidden_dim, hidden_dim)
        self.global_output = nn.Linear(hidden_dim, 1)

        # Mọi concept profile (3·3·5·4·3·2·8 = 8.640 với Derm7pt) cho exact enumeration; không lưu vào state_dict.
        profiles = torch.tensor(list(itertools.product(*[range(count) for count in self.state_counts])), dtype=torch.long)
        self.register_buffer("profiles", profiles, persistent=False)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.features(images), 1)

    # E_class(x, y) cho mọi nhãn: [B, C].
    def class_energies(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(self.class_projection(features))
        return self.class_output(_gate(hidden[:, None, :], self.class_embedding[None])).squeeze(-1)

    # E_concept(x, c_g) cho mọi trạng thái của từng nhóm: list [B, K_g].
    def concept_energies(self, features: torch.Tensor) -> List[torch.Tensor]:
        hidden = self.dropout(self.concept_projection(features))
        return [output(_gate(hidden[:, None, :], embedding[None])).squeeze(-1)
                for output, embedding in zip(self.concept_outputs, self.concept_embeddings)]

    # E_global(c, y) từ vector concept đã ghép [..., 7d] và vector nhãn [..., d].
    def _global_energy(self, concept_vector: torch.Tensor, class_vector: torch.Tensor) -> torch.Tensor:
        return self.global_output(_gate(self.global_projection(concept_vector), class_vector)).squeeze(-1)

    def concept_vector(self, concepts: torch.Tensor) -> torch.Tensor:
        # Trạng thái rời rạc [N, 7] -> embedding chưa chuẩn hóa đã ghép [N, 7d] (c_embed_cy của mã gốc).
        return torch.cat([embedding[concepts[:, group]] for group, embedding in enumerate(self.concept_embeddings)], dim=-1)

    # E_global(c, y) cho mọi nhãn với profile cố định: [N, C].
    def global_class_energies(self, concepts: torch.Tensor) -> torch.Tensor:
        vector = self.concept_vector(concepts)
        return self._global_energy(vector[:, None, :], self.class_embedding[None])

    # Năng lượng nới lỏng của Algorithm 1: concept/nhãn là xác suất, nhân với embedding.
    def relaxed_energy(self, features, concept_probabilities, class_probabilities, weights, hard_global=True):
        class_vector = class_probabilities @ self.class_embedding
        class_hidden = self.dropout(self.class_projection(features))
        concept_hidden = self.dropout(self.concept_projection(features))
        mixed = [probability @ embedding for probability, embedding in zip(concept_probabilities, self.concept_embeddings)]
        e_class = self.class_output(_gate(class_hidden, class_vector)).squeeze(-1)
        e_concept = sum(output(_gate(concept_hidden, vector)).squeeze(-1)
                        for output, vector in zip(self.concept_outputs, mixed))
        if hard_global:
            # Mã gốc làm cứng concept trước E_global ở mỗi bước, nên E_global không có gradient về concept.
            global_vector = self.concept_vector(torch.stack([p.argmax(-1) for p in concept_probabilities], dim=-1))
        else:
            global_vector = torch.cat(mixed, dim=-1)
        e_global = self._global_energy(global_vector, class_vector)
        w_class, w_concept, w_global = weights
        return w_class * e_class + w_concept * e_concept + w_global * e_global


# Bản categorical của cy_augment: chọn ceil(B·p_sample) ca và ceil(G·p_concept) nhóm dùng chung,
# đổi mỗi ô được chọn sang một trạng thái hợp lệ khác; nhóm nhị phân đúng bằng phép lật bit.
def perturb_concepts(concepts: torch.Tensor, state_counts: Sequence[int], sample_fraction: float = 0.2,
                     concept_fraction: float = 0.2, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    perturbed = concepts.clone()
    if sample_fraction == 0 or concept_fraction == 0:
        return perturbed
    rows = torch.randperm(len(concepts), generator=generator)[:math.ceil(len(concepts) * sample_fraction)]
    groups = torch.randperm(len(state_counts), generator=generator)[:math.ceil(len(state_counts) * concept_fraction)]
    for group in groups.tolist():
        offsets = torch.randint(1, state_counts[group], (len(rows),), generator=generator)
        rows_device, offsets = rows.to(concepts.device), offsets.to(concepts.device)
        perturbed[rows_device, group] = (perturbed[rows_device, group] + offsets) % state_counts[group]
    return perturbed


# Loss của mã gốc: λ_xy·CE(−E_class) + λ_xc·Σ_g CE(−E_concept_g) + λ_cy·CE(−E_global(c̃, ·)),
# với c̃ là concept thật đã xáo trộn khi train. Mã gốc không có trọng số; quy trình chung truyền thêm
# trọng số lớp cho hai loss chẩn đoán và trọng số trạng thái cho từng nhóm concept, giống Soft/Hard CBM.
def ecbm_training_loss(model: CategoricalECBM, images, concepts, labels, weights=(3.0, 1.0, 1.0),
                       perturbation=(0.2, 0.2), class_weights: Optional[torch.Tensor] = None,
                       concept_state_weights: Optional[Sequence[torch.Tensor]] = None
                       ) -> Tuple[torch.Tensor, Dict[str, float]]:
    features = model.extract_features(images)
    state_weights = concept_state_weights or [None] * len(model.state_counts)
    class_loss = F.cross_entropy(-model.class_energies(features), labels, weight=class_weights)
    concept_loss = sum(F.cross_entropy(-energy, concepts[:, group], weight=state_weights[group])
                       for group, energy in enumerate(model.concept_energies(features)))
    global_concepts = perturb_concepts(concepts, model.state_counts, *perturbation) if model.training else concepts
    global_loss = F.cross_entropy(-model.global_class_energies(global_concepts), labels, weight=class_weights)
    w_class, w_concept, w_global = weights
    total = w_class * class_loss + w_concept * concept_loss + w_global * global_loss
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite ECBM training loss")
    return total, {"class": float(class_loss.detach()), "concept": float(concept_loss.detach()),
                   "global": float(global_loss.detach())}


# Suy luận gradient (Algorithm 1): Adam trên logits concept/nhãn khởi tạo 0, không nhận nhãn hay concept thật.
# Thay dừng sớm theo mean năng lượng cả batch của mã gốc bằng số bước cố định, giữ điểm năng lượng thấp nhất từng ca.
def gradient_inference(model: CategoricalECBM, features: torch.Tensor, weights=(1.0, 1.0, 0.01), steps: int = 80,
                       lr: float = 0.1, hard_global: bool = True) -> Dict[str, torch.Tensor]:
    features = features.detach()
    with torch.enable_grad():
        concept_logits = [torch.zeros(len(features), count, device=features.device, requires_grad=True)
                          for count in model.state_counts]
        class_logits = torch.zeros(len(features), model.num_classes, device=features.device, requires_grad=True)
        variables = concept_logits + [class_logits]
        optimizer = torch.optim.Adam(variables, lr=lr)
        best_energy = torch.full((len(features),), float("inf"), device=features.device)
        best_concepts = [torch.zeros_like(logits) for logits in concept_logits]
        best_classes = torch.zeros_like(class_logits)
        for step in range(steps + 1):
            concepts = [logits.softmax(-1) for logits in concept_logits]
            classes = class_logits.softmax(-1)
            energy = model.relaxed_energy(features, concepts, classes, weights, hard_global)
            if not torch.isfinite(energy).all():
                raise FloatingPointError("Non-finite ECBM inference energy")
            with torch.no_grad():
                improved = energy < best_energy
                best_energy = torch.where(improved, energy, best_energy)
                best_concepts = [torch.where(improved[:, None], value, previous)
                                 for value, previous in zip(concepts, best_concepts)]
                best_classes = torch.where(improved[:, None], classes, best_classes)
            if step == steps:
                break
            optimizer.zero_grad(set_to_none=True)
            for variable, gradient in zip(variables, torch.autograd.grad(energy.sum(), variables)):
                variable.grad = gradient
            optimizer.step()
    return {"class_probabilities": best_classes.detach(),
            "concept_probabilities": [value.detach() for value in best_concepts], "energy": best_energy.detach()}


# Exact enumeration: p(y, c | x) ∝ exp(−[w·E_class + w·Σ_g E_concept + w·E_global]) trên mọi profile rời rạc.
# Trả về p(y | x) và xác suất biên từng nhóm concept; clamp (nhóm -> trạng thái [B]) giới hạn profile khi can thiệp.
@torch.no_grad()
def exact_inference(model: CategoricalECBM, features: torch.Tensor, weights=(1.0, 1.0, 0.01),
                    clamp: Optional[Dict[int, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
    w_class, w_concept, w_global = weights
    profiles = model.profiles
    e_class = model.class_energies(features)
    e_concepts = model.concept_energies(features)
    e_global = model.global_class_energies(profiles)  # [P, C], không phụ thuộc ảnh
    concept_sum = sum(energy[:, profiles[:, group]] for group, energy in enumerate(e_concepts))  # [B, P]
    log_joint = -(w_class * e_class[:, None, :] + w_concept * concept_sum[:, :, None] + w_global * e_global[None])
    if clamp:
        allowed = torch.ones(len(features), len(profiles), dtype=torch.bool, device=features.device)
        for group, states in clamp.items():
            allowed &= profiles[None, :, group] == states[:, None]
        log_joint = log_joint.masked_fill(~allowed[:, :, None], float("-inf"))
    log_joint = log_joint - torch.logsumexp(log_joint.flatten(1), dim=1)[:, None, None]
    class_probabilities = torch.logsumexp(log_joint, dim=1).exp()
    profile_probabilities = torch.logsumexp(log_joint, dim=2).exp()  # [B, P]
    concept_probabilities = [
        torch.zeros(len(features), count, device=features.device).index_add_(1, profiles[:, group], profile_probabilities)
        for group, count in enumerate(model.state_counts)]
    return {"class_probabilities": class_probabilities, "concept_probabilities": concept_probabilities}
