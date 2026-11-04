import os
import sys
import tempfile
import unittest
import contextlib
import io
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.run_blackbox import (_build_optimizer, _default_paths, _head_preset, evaluate_blackbox_test)
from src.dataset import get_dataloaders
from src.selection import select_threshold
from src.runtime import file_sha256
from src.metrics import compute_metrics
from src.models import BlackBoxClassifier, load_blackbox_state_dict
from src.transforms import get_transforms


class BlackboxProtocolTests(unittest.TestCase):
    def test_efficientnet_has_one_feature_tree_and_forward_shape(self):
        # 1. Kiểm tra kiến trúc EfficientNet-B0: có duy nhất 1 cây trích xuất đặc trưng (1280 chiều) và đầu ra (batch, 2)
        model = BlackBoxClassifier(pretrained=False).eval()
        state_keys = model.state_dict().keys()
        self.assertFalse(any(key.startswith("backbone.") for key in state_keys))
        with torch.no_grad():
            image = torch.zeros(1, 3, 224, 224)
            self.assertEqual(tuple(model(image).shape), (1, 2))
            self.assertEqual(tuple(model.extract_features(image).shape), (1, 1280))

    def test_legacy_duplicate_keys_can_be_loaded(self):
        # 2. Kiểm tra khả năng tương thích ngược: nạp được trọng số từ các checkpoint cũ bị trùng tên prefix
        model = BlackBoxClassifier(pretrained=False)
        state = model.state_dict()
        legacy = dict(state)
        feature_key = next(key for key in state if key.startswith("features."))
        legacy["backbone.features.0.0.weight"] = state[feature_key].clone()
        load_blackbox_state_dict(model, legacy)

    def test_comparison_augmentation_and_eval_preprocessing(self):
        # 3. Kiểm tra các bước tăng cường ảnh train và tiền xử lý LetterboxResize cho valid/test
        train_transform = get_transforms("train", augmentation_preset="comparison")
        names = [type(step).__name__ for step in train_transform.transforms]
        self.assertEqual(
            names[:5],
            ["RandomResizedCrop", "RandomHorizontalFlip", "RandomVerticalFlip", "ColorJitter", "RandomRotation"],
        )
        image = Image.new("RGB", (300, 250), (120, 90, 50))
        self.assertEqual(tuple(train_transform(image).shape), (3, 224, 224))
        valid_transform = get_transforms("valid", augmentation_preset="comparison")
        self.assertEqual(type(valid_transform.transforms[0]).__name__, "LetterboxResize")
        self.assertTrue(torch.equal(valid_transform(image), valid_transform(image)))

    def test_pr_auc_is_average_precision(self):
        # 4. Kiểm tra công thức PR-AUC khớp chính xác với Average Precision Score của scikit-learn
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.6, 0.4, 0.8])
        metrics = compute_metrics(y_true, (y_prob >= 0.5).astype(int), y_prob)
        self.assertAlmostEqual(metrics["pr_auc"], average_precision_score(y_true, y_prob))

    def test_threshold_is_selected_from_validation(self):
        # 5. Kiểm tra thuật toán quét tìm ngưỡng quyết định tối ưu tau* trên tập validation
        # Mặc định Macro F1: 0.45 và 0.8 cùng đạt 0.733, hòa chọn ngưỡng gần 0.5; BAcc giữ cho protocol cũ.
        threshold, f1 = select_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.55, 0.45, 0.8]))
        self.assertAlmostEqual(threshold, 0.45)
        self.assertAlmostEqual(f1, 0.7333333333333334)
        threshold, bacc = select_threshold(
            np.array([0, 0, 1, 1]), np.array([0.1, 0.55, 0.45, 0.8]), "balanced_accuracy")
        self.assertAlmostEqual(threshold, 0.45)
        self.assertAlmostEqual(bacc, 0.75)

    def test_train_dataloaders_exclude_test(self):
        # 6. Kiểm tra giao thức bảo vệ dữ liệu: DataLoader huấn luyện tuyệt đối không nạp tập Test
        bundle = get_dataloaders(
            manifest_path=os.path.join(PROJECT_ROOT, "data", "manifest.csv"),
            project_root=PROJECT_ROOT,
            batch_size=16,
            num_workers=0,
            augmentation_preset="comparison",
            include_test=False,
        )
        self.assertEqual(len(bundle["datasets"]["train"]), 413)
        self.assertEqual(len(bundle["datasets"]["valid"]), 203)
        self.assertNotIn("test", bundle)
        self.assertNotIn("test", bundle["datasets"])

    def test_default_paths_are_seed_specific(self):
        # 7. Kiểm tra đường dẫn lưu checkpoint và kết quả phải gắn liền với seed ngẫu nhiên
        self.assertNotEqual(_default_paths(42, "comparison"), _default_paths(123, "comparison"))
        self.assertNotEqual(_default_paths(42, "comparison"), _default_paths(42, "legacy_letterbox"))

    def test_explicit_test_mode_uses_frozen_threshold(self):
        # 8. Kiểm tra giao thức đánh giá Test của Black-box: bắt buộc dùng ngưỡng tau* đã đóng băng từ validation
        manifest = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
        model = BlackBoxClassifier(pretrained=False)
        config = {
            "seed": 7, "augmentation_preset": "comparison", "target_size": 224,
            "batch_size": 16, "class_weights": [0.64, 2.29],
        }
        checkpoint = {
            "epoch": 1, "model_state_dict": model.state_dict(), "config": config,
            "decision_threshold": 0.42, "manifest_sha256": file_sha256(manifest),
        }

        class FakeTestDataset:
            def __init__(self, *args, **kwargs):
                self.df = pd.DataFrame({
                    "case_num": range(1, 396), "source_index": range(395),
                    "split": ["test"] * 395, "diagnosis": ["sample"] * 395,
                    "is_inconsistent_profile": [False] * 395,
                    "diagnosis_binary": [0] * 294 + [1] * 101,
                })

            def __len__(self):
                return 395

            def __getitem__(self, index):
                raise AssertionError("The mocked evaluator should not read images")

        probs = np.array([0.1] * 294 + [0.9] * 101)
        preds = (probs >= 0.42).astype(int)
        metrics = compute_metrics(np.array([0] * 294 + [1] * 101), preds, probs)
        with tempfile.TemporaryDirectory(prefix="blackbox_test_mode_") as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "model.pth")
            result_path = os.path.join(temp_dir, "test.json")
            torch.save(checkpoint, checkpoint_path)
            with patch("experiments.run_blackbox.Derm7ptDataset", FakeTestDataset), patch(
                "experiments.run_blackbox.evaluate", return_value=(metrics, 0.1, preds, probs)
            ) as mocked_evaluate:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = evaluate_blackbox_test(checkpoint_path, manifest, "cpu", result_path)
            self.assertEqual(result["decision_threshold"], 0.42)
            self.assertEqual(result["mode"], "final_test")
            self.assertEqual(len(result["test_predictions"]), 395)
            self.assertEqual(mocked_evaluate.call_args.kwargs["threshold"], 0.42)

    def test_mlp128_head_and_separate_head_lr(self):
        # 9. Head MLP128 giống Soft CBM/Hard CBM; head_lr tách đúng hai nhóm tham số, mặc định giữ một nhóm
        linear = BlackBoxClassifier(pretrained=False)
        self.assertEqual({key for key in linear.state_dict() if key.startswith("classifier.")},
                         {"classifier.1.weight", "classifier.1.bias"})
        self.assertEqual(len(_build_optimizer(linear, 1e-4, 1e-2).param_groups), 1)
        model = BlackBoxClassifier(pretrained=False, head="mlp128").eval()
        self.assertIsInstance(model.classifier[1][1], torch.nn.LayerNorm)
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.zeros(2, 3, 224, 224)).shape), (2, 2))
        groups = _build_optimizer(model, 1e-4, 1e-2, head_lr=1e-3).param_groups
        self.assertEqual([group["lr"] for group in groups], [1e-4, 1e-3])
        grouped = [id(p) for group in groups for p in group["params"]]
        self.assertEqual(sorted(grouped), sorted(id(p) for p in model.parameters()))
        with self.assertRaises(ValueError):
            BlackBoxClassifier(pretrained=False, head="mlp64")

    def test_non_default_head_config_has_distinct_paths(self):
        # 10. Model black-box mang tên backbone; mặc định quy trình chung (Linear, 50 epochs, head LR 1e-3)
        baseline_checkpoint, baseline_results = _default_paths(42)
        self.assertTrue(baseline_checkpoint.endswith(os.path.join("derm7pt", "efficientnet_b0", "linear_e50_headlr0.001", "seed42.pth")))
        self.assertTrue(baseline_results.endswith(os.path.join("derm7pt", "efficientnet_b0", "linear_e50_headlr0.001", "seed42.json")))
        self.assertIn("_thrbacc", _default_paths(42, threshold_metric="balanced_accuracy")[1])
        checkpoint, results = _default_paths(42, epochs=20, classifier_head="mlp128", head_lr=1e-3)
        self.assertTrue(checkpoint.endswith(os.path.join("efficientnet_b0", "mlp128_e20_headlr0.001", "seed42.pth")))
        self.assertTrue(results.endswith(os.path.join("efficientnet_b0", "mlp128_e20_headlr0.001", "seed42.json")))
        self.assertNotEqual(_default_paths(42, epochs=20)[0], baseline_checkpoint)

    def test_backbone_names_the_model_and_legacy_configs_map_to_efficientnet_b0(self):
        # 10b. Tên model black-box là id backbone; config cũ (không ghi hoặc ghi "EfficientNet-B0") đọc ra efficientnet_b0
        from src.naming import backbone_from_config, canonical_model
        self.assertEqual(backbone_from_config({}), "efficientnet_b0")
        self.assertEqual(backbone_from_config({"backbone": "EfficientNet-B0"}), "efficientnet_b0")
        self.assertEqual(canonical_model("M1_BlackBox_EfficientNetB0"), "efficientnet_b0")
        with self.assertRaises(ValueError):
            backbone_from_config({"backbone": "ResNet-50"})
        with self.assertRaises(ValueError):
            BlackBoxClassifier(pretrained=False, backbone="resnet50")
        with self.assertRaises(ValueError):
            _default_paths(42, backbone="resnet50")
        self.assertEqual(BlackBoxClassifier(pretrained=False).backbone_name, "efficientnet_b0")

    def test_head_preset_is_read_from_checkpoint_config(self):
        # 11. Checkpoint cũ không ghi head là Linear; test mode dựng lại đúng MLP128 từ config
        self.assertEqual(_head_preset({}), "linear")
        self.assertEqual(_head_preset({"classifier_head": "MLP(1280, 128, 2)"}), "mlp128")
        with self.assertRaises(ValueError):
            _head_preset({"classifier_head": "MLP(1280, 64, 2)"})


if __name__ == "__main__":
    unittest.main()
