import contextlib
import io
import itertools
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.run_ecbm import (RECIPES, _build_optimizer, _config_paths, _default_paths, _resolve,
                                 evaluate_ecbm_test, validate_ecbm_config)
from src.selection import select_threshold
from src.dataset import CONCEPT_NAMES, CONCEPT_NUM_CLASSES
from src.protocol import load_concept_schema, manifest_fingerprint
from src.ecbm import (CategoricalECBM, ecbm_training_loss, exact_inference, gradient_inference,
                      perturb_concepts)


def _tiny_model(counts=(2, 3)):
    torch.manual_seed(0)
    names = [f"g{i}" for i in range(len(counts))]
    return CategoricalECBM(names, dict(zip(names, counts)), pretrained=False, hidden_dim=4).eval()


class ECBMProtocolTests(unittest.TestCase):
    def test_exact_inference_matches_brute_force_with_and_without_clamp(self):
        # 1. Exact enumeration khớp phép duyệt tay mọi (profile, nhãn); clamp chỉ giữ profile hợp lệ
        model, weights = _tiny_model(), (1.0, 0.7, 0.3)
        features = torch.randn(3, 1280)
        with torch.no_grad():
            e_class, e_concepts = model.class_energies(features), model.concept_energies(features)
            joint = torch.zeros(3, 2, 3, 2)
            for a, b in itertools.product(range(2), range(3)):
                global_energy = model.global_class_energies(torch.tensor([[a, b]]))[0]
                concept_energy = (e_concepts[0][:, a] + e_concepts[1][:, b])[:, None]
                joint[:, a, b, :] = -(weights[0] * e_class + weights[1] * concept_energy + weights[2] * global_energy)
        for clamp in (None, {1: torch.tensor([2, 0, 1])}):
            target = joint.clone()
            if clamp:
                for row, state in enumerate(clamp[1].tolist()):
                    target[row, :, [s for s in range(3) if s != state], :] = float("-inf")
            probability = target.flatten(1).softmax(1).view(3, 2, 3, 2)
            output = exact_inference(model, features, weights, clamp=clamp)
            torch.testing.assert_close(output["class_probabilities"], probability.sum((1, 2)))
            torch.testing.assert_close(output["concept_probabilities"][0], probability.sum((2, 3)))
            torch.testing.assert_close(output["concept_probabilities"][1], probability.sum((1, 3)))

    def test_hard_global_has_no_gradient_to_concepts(self):
        # 2. Như mã gốc: E_global nhận concept argmax nên không truyền gradient về concept; bản soft thì có
        model = _tiny_model()
        features = torch.randn(2, 1280)
        logits = [torch.randn(2, count, requires_grad=True) for count in model.state_counts]
        classes = torch.full((2, 2), 0.5)
        for hard, expect_gradient in ((True, False), (False, True)):
            energy = model.relaxed_energy(features, [l.softmax(-1) for l in logits], classes, (0.0, 0.0, 1.0), hard)
            gradients = torch.autograd.grad(energy.sum(), logits, allow_unused=True)
            has_gradient = any(g is not None and g.abs().sum() > 0 for g in gradients)
            self.assertEqual(has_gradient, expect_gradient)

    def test_gradient_inference_returns_distributions_and_lowers_energy(self):
        # 3. Suy luận gradient trả về phân phối hợp lệ và năng lượng tốt nhất không cao hơn điểm khởi tạo
        model = _tiny_model()
        features = torch.randn(4, 1280)
        start = model.relaxed_energy(features, [torch.full((4, k), 1.0 / k) for k in model.state_counts],
                                     torch.full((4, 2), 0.5), (1.0, 1.0, 0.01)).detach()
        output = gradient_inference(model, features, steps=10)
        torch.testing.assert_close(output["class_probabilities"].sum(-1), torch.ones(4))
        for probability in output["concept_probabilities"]:
            torch.testing.assert_close(probability.sum(-1), torch.ones(4))
        self.assertTrue(torch.all(output["energy"] <= start + 1e-6))

    def test_concept_perturbation_matches_upstream_counts(self):
        # 4. Xáo trộn ceil(20%) ca và ceil(20%) nhóm dùng chung; ô bị đổi nhận trạng thái hợp lệ khác
        counts = [CONCEPT_NUM_CLASSES[name] for name in CONCEPT_NAMES]
        concepts = torch.stack([torch.randint(0, k, (32,)) for k in counts], dim=1)
        original = concepts.clone()
        perturbed = perturb_concepts(concepts, counts, 0.2, 0.2, generator=torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(concepts, original))
        changed = perturbed != concepts
        self.assertEqual(int(changed.any(1).sum()), 7)
        self.assertEqual(int(changed.any(0).sum()), 2)
        self.assertTrue(all(int(perturbed[:, g].max()) < k for g, k in enumerate(counts)))

    def test_training_loss_is_finite_and_reaches_all_energies(self):
        # 5. Loss gốc (3/1/1) hữu hạn và gradient tới cả ba năng lượng lẫn embedding
        model = CategoricalECBM(CONCEPT_NAMES, CONCEPT_NUM_CLASSES, pretrained=False).train()
        self.assertEqual(len(model.profiles), 8640)
        self.assertNotIn("profiles", model.state_dict())
        concepts = torch.stack([torch.randint(0, CONCEPT_NUM_CLASSES[n], (4,)) for n in CONCEPT_NAMES], dim=1)
        loss, parts = ecbm_training_loss(model, torch.randn(4, 3, 64, 64), concepts, torch.tensor([0, 1, 0, 1]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(parts), {"class", "concept", "global"})
        for parameter in (model.class_embedding, model.concept_embeddings[0], model.class_projection.weight,
                          model.concept_projection.weight, model.global_projection.weight):
            self.assertIsNotNone(parameter.grad)

    def test_default_paths_are_distinct_and_recoverable(self):
        # 6. Mặc định là công thức chung (50 epochs, exact); đường dẫn khác theo seed/công thức/suy luận/ngưỡng
        self.assertTrue(_default_paths(42)[1].endswith(os.path.join("derm7pt", "ecbm", "efficientnet_b0_unified_exact_e50", "seed42.json")))
        self.assertNotEqual(_default_paths(42), _default_paths(123))
        self.assertNotEqual(_default_paths(42), _default_paths(42, "paper"))
        self.assertNotEqual(_default_paths(42), _default_paths(42, selection_inference="gradient"))
        self.assertIn("_thrbacc", _default_paths(42, threshold_metric="balanced_accuracy")[1])
        settings = _resolve("unified")
        config = {"protocol": RECIPES["unified"]["protocol"], "recipe": "unified", "seed": 42,
                  "selection_inference": "exact", "epochs": 50, "batch_size": 16, "learning_rate": 1e-4,
                  "head_learning_rate": 1e-3, "weight_decay": 1e-2, "patience": None,
                  "checkpoint_selection": "validation_f1_macro_at_0.5",
                  "threshold_selection": "maximize_validation_f1_macro_tie_nearest_0.5"}
        self.assertEqual(settings["epochs"], 50)
        validate_ecbm_config(config)
        self.assertEqual(_config_paths(config), _default_paths(42))
        with self.assertRaises(ValueError):
            validate_ecbm_config({**config, "protocol": "other"})

    def test_paper_stage_one_config_is_still_readable(self):
        # 6b. Run bản gốc (bước 1) không ghi recipe/head LR và chọn ngưỡng theo BAcc; vẫn suy ra đúng cấu hình
        config = {"protocol": RECIPES["paper"]["protocol"], "seed": 42, "selection_inference": "gradient",
                  "epochs": 100, "batch_size": 32, "learning_rate": 0.01, "weight_decay": 4e-5, "patience": 15,
                  "checkpoint_selection": "validation_f1_macro_at_0.5",
                  "threshold_selection": "maximize_validation_balanced_accuracy_tie_nearest_0.5"}
        validate_ecbm_config(config)
        self.assertTrue(_config_paths(config)[1].endswith(os.path.join("efficientnet_b0_paper_gradient_e100_thrbacc", "seed42.json")))

    def test_unified_recipe_matches_cbm_optimizer_and_schedule(self):
        # 6c. Công thức chung: AdamW, backbone 1e-4, mạng năng lượng/embedding 1e-3, cosine; đủ mọi tham số
        model = CategoricalECBM(CONCEPT_NAMES, CONCEPT_NUM_CLASSES, pretrained=False)
        optimizer, scheduler = _build_optimizer(model, "unified", _resolve("unified"))
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual([g["lr"] for g in optimizer.param_groups], [1e-4, 1e-3])
        self.assertEqual({id(p) for p in optimizer.param_groups[0]["params"]}, {id(p) for p in model.features.parameters()})
        grouped = [id(p) for g in optimizer.param_groups for p in g["params"]]
        self.assertEqual(sorted(grouped), sorted(id(p) for p in model.parameters()))
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        self.assertEqual(scheduler.T_max, 50)
        optimizer, _ = _build_optimizer(model, "paper", _resolve("paper"))
        self.assertIsInstance(optimizer, torch.optim.SGD)

    def test_class_and_state_weights_change_training_loss(self):
        # 6d. Trọng số lớp/trạng thái của công thức chung thực sự đi vào loss chẩn đoán và concept
        model = CategoricalECBM(CONCEPT_NAMES, CONCEPT_NUM_CLASSES, pretrained=False).eval()
        torch.manual_seed(1)
        images = torch.randn(4, 3, 64, 64)
        concepts = torch.stack([torch.randint(0, CONCEPT_NUM_CLASSES[n], (4,)) for n in CONCEPT_NAMES], dim=1)
        labels = torch.tensor([0, 0, 0, 1])
        _, plain = ecbm_training_loss(model, images, concepts, labels)
        weights = [torch.rand(CONCEPT_NUM_CLASSES[n]) + 0.1 for n in CONCEPT_NAMES]
        _, weighted = ecbm_training_loss(model, images, concepts, labels, class_weights=torch.tensor([0.6, 2.3]),
                                         concept_state_weights=weights)
        for part in ("class", "concept", "global"):
            self.assertNotAlmostEqual(plain[part], weighted[part], places=6)

    def test_f1_and_bacc_threshold_rules_can_choose_different_thresholds(self):
        # 6e. Tiêu chí ngưỡng có tác dụng: Macro F1 chọn 0.9 (ít dự đoán dương hơn), BAcc chọn 0.5
        y = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        p = np.array([.1, .2, .3, .4, .6, .7, .5, .9])
        self.assertAlmostEqual(select_threshold(y, p)[0], 0.9)
        self.assertAlmostEqual(select_threshold(y, p, "balanced_accuracy")[0], 0.5)

    def test_test_mode_uses_only_frozen_thresholds_for_each_inference(self):
        # 7. Chấm test dùng đúng ngưỡng đã chốt trên validation cho từng cách suy luận; checkpoint cũ không có ngưỡng
        # gradient_soft thì không chấm soft (không chọn ngưỡng trên test). Dataset/suy luận được giả lập.
        manifest = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
        config = {"protocol": RECIPES["paper"]["protocol"], "seed": 7, "selection_inference": "gradient", "epochs": 100,
                  "checkpoint_selection": "validation_f1_macro_at_0.5", "hidden_dim": 4, "dropout": 0.2,
                  "target_size": 224, "batch_size": 32}
        model = CategoricalECBM(CONCEPT_NAMES, CONCEPT_NUM_CLASSES, pretrained=False, hidden_dim=4)
        checkpoint = {"epoch": 3, "model_state_dict": model.state_dict(), "config": config,
                      "decision_threshold": 0.3, "decision_thresholds": {"gradient": 0.3, "exact": 0.6},
                      "manifest_fingerprint": manifest_fingerprint(manifest), "manifest_sha256": "unused",
                      "concept_schema": load_concept_schema(manifest)}
        labels = np.array([0] * 294 + [1] * 101)
        scores = np.linspace(0.0, 1.0, 395)
        concepts = {"true": {c: np.zeros(395, dtype=int) for c in CONCEPT_NAMES},
                    "pred": {c: np.zeros(395, dtype=int) for c in CONCEPT_NAMES},
                    "probabilities": {c: np.eye(CONCEPT_NUM_CLASSES[c])[np.zeros(395, dtype=int)] for c in CONCEPT_NAMES}}
        requested = []

        def fake_predict(model, loader, device, methods):
            requested.append(list(methods))
            return {m: {"labels": labels, "scores": scores, "concepts": concepts} for m in methods}

        class FakeTestDataset:
            def __init__(self, *args, **kwargs):
                self.df = pd.DataFrame({"case_num": range(1, 396), "source_index": range(395), "split": ["test"] * 395,
                                        "diagnosis": ["sample"] * 395, "is_inconsistent_profile": [False] * 395,
                                        "diagnosis_binary": labels})

            def __len__(self):
                return 395

            def __getitem__(self, index):
                raise AssertionError("The mocked evaluator should not read images")

        results = []
        for thresholds in ({"gradient": 0.3, "exact": 0.6}, {"gradient": 0.3, "exact": 0.6, "gradient_soft": 0.8}):
            with tempfile.TemporaryDirectory(prefix="ecbm_test_mode_") as temp_dir:
                path = os.path.join(temp_dir, "model.pth")
                torch.save({**checkpoint, "decision_thresholds": thresholds}, path)
                with patch("experiments.run_ecbm.Derm7ptDataset", FakeTestDataset), \
                        patch("experiments.run_ecbm.predict", side_effect=fake_predict), \
                        contextlib.redirect_stdout(io.StringIO()):
                    results.append(evaluate_ecbm_test(path, manifest, "cpu", save_results=False))
        old, new = results
        self.assertEqual(requested, [["gradient", "exact"], ["gradient", "exact", "gradient_soft"]])
        self.assertEqual(old["decision_threshold"], 0.3)
        self.assertEqual(sum(r["y_pred"] for r in old["test_predictions"]), int((scores >= 0.3).sum()))
        self.assertEqual(set(old["test_other_inferences"]), {"exact"})
        self.assertEqual(set(new["test_other_inferences"]), {"exact", "gradient_soft"})
        for method, threshold in (("exact", 0.6), ("gradient_soft", 0.8)):
            block = new["test_other_inferences"][method]
            self.assertEqual(block["decision_threshold"], threshold)
            self.assertEqual(sum(r["y_pred"] for r in block["test_predictions"]), int((scores >= threshold).sum()))

    def test_soft_and_hard_ecbm_share_training_but_differ_at_inference(self):
        # 8. Soft ECBM (concept xác suất trong E_global) và Hard ECBM (argmax) dùng cùng mô hình, khác kết quả suy luận
        model = _tiny_model()
        features = torch.randn(6, 1280)
        hard = gradient_inference(model, features, weights=(1.0, 1.0, 1.0), steps=15, hard_global=True)
        soft = gradient_inference(model, features, weights=(1.0, 1.0, 1.0), steps=15, hard_global=False)
        self.assertFalse(torch.allclose(hard["class_probabilities"], soft["class_probabilities"]))


if __name__ == "__main__":
    unittest.main()
