# Kiểm thử pilot Soft CBM MLP, gradient liên hợp và tính tương thích test/can thiệp đóng băng.
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.dataset import CONCEPT_NAMES, CONCEPT_NUM_CLASSES, compute_diagnosis_weights
from src import runtime
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from experiments import run_soft_cbm, run_soft_cbm_interventions
from src.models import get_soft_joint_cbm
from src.protocol import load_concept_schema

ROOT = Path(__file__).resolve().parents[1]


class TinyBackbone(nn.Module):
    # Chỉ thay thế phần trích xuất đặc trưng ảnh; dùng factory/forward Soft CBM thực tế.
    def __init__(self, **kwargs):
        super().__init__()
        self.features = nn.Linear(3, 8)
        self.avgpool = nn.Identity()
        self.classifier = nn.Sequential(nn.Dropout(.2), nn.Linear(8, 2))


class TinyDataset(Dataset):
    def __init__(self, manifest_path, split, **kwargs):
        df = pd.read_csv(manifest_path)
        self.df = df[df.split == split].reset_index(drop=True)
        self.label_mapping = load_concept_schema(manifest_path)["label_mapping"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return {"image": torch.tensor([index/1000, 1., -1.]),
                "label": torch.tensor(int(row.diagnosis_binary)),
                "concept_indices": torch.tensor([self.label_mapping[c][row[c]] for c in CONCEPT_NAMES])}


class SoftCBMUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="soft_cbm_upgrade_test_")
        self.root = Path(self.temp.name)
        data = self.root / "data"
        data.mkdir()
        self.manifest = data / "manifest.csv"
        shutil.copy2(ROOT / "data/manifest.csv", self.manifest)
        shutil.copy2(ROOT / "data/label_mapping.json", data / "label_mapping.json")
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.old_threads)
        self.temp.cleanup()

    @contextlib.contextmanager
    def pipeline(self):
        train, valid = TinyDataset(self.manifest, "train"), TinyDataset(self.manifest, "valid")
        bundle = {"train": DataLoader(train, batch_size=128), "valid": DataLoader(valid, batch_size=128),
                  "datasets": {"train": train, "valid": valid},
                  "class_weights": compute_diagnosis_weights(str(self.manifest), str(ROOT))}
        with patch.object(run_soft_cbm, "PROJECT_ROOT", str(self.root)), \
             patch.object(run_soft_cbm, "get_dataloaders", return_value=bundle) as loaders, \
             patch("src.cbm.models.efficientnet_b0", side_effect=TinyBackbone), \
             patch.object(run_soft_cbm, "Derm7ptDataset", side_effect=TinyDataset) as test_dataset, \
             contextlib.redirect_stdout(io.StringIO()):
            yield loaders, test_dataset

    def args(self):
        return ["--manifest_path", str(self.manifest), "--epochs", "2", "--batch_size", "128",
                "--device", "cpu", "--num_workers", "0", "--diagnosis_head", "mlp128",
                "--diagnosis_lr", "0.001", "--checkpoint_metric", "f1_macro"]

    def test_mlp_factory_keeps_soft_bottleneck_and_joint_diagnosis_gradient(self):
        torch.manual_seed(17)
        with patch("src.cbm.models.efficientnet_b0", side_effect=TinyBackbone):
            model = get_soft_joint_cbm(CONCEPT_NAMES, CONCEPT_NUM_CLASSES,
                                       pretrained=False, diagnosis_head="mlp128")
        self.assertIsInstance(model.diagnosis_head, nn.Sequential)
        logits, concepts, vector = model(torch.randn(4, 3))
        self.assertEqual(tuple(vector.shape), (4, 28))
        self.assertTrue(torch.any((vector > 0) & (vector < 1)))
        for name in CONCEPT_NAMES:
            self.assertTrue(torch.allclose(concepts[name].softmax(-1).sum(-1), torch.ones(4)))
        nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1])).backward()
        for group in [model.features, model.concept_heads, model.diagnosis_head]:
            self.assertGreater(sum(p.grad.abs().sum().item() for p in group.parameters()
                                   if p.grad is not None), 0.)

    def test_separate_learning_rates_cover_parameters_once_and_update_both_groups(self):
        with patch("src.cbm.models.efficientnet_b0", side_effect=TinyBackbone):
            model = get_soft_joint_cbm(CONCEPT_NAMES, CONCEPT_NUM_CLASSES,
                                       pretrained=False, diagnosis_head="mlp128")
        optimizer = run_soft_cbm._build_optimizer(model, 1e-4, .01, 1e-3)
        groups = optimizer.param_groups
        ids = [id(p) for g in groups for p in g["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        self.assertEqual({id(p) for p in groups[1]["params"]},
                         {id(p) for p in model.diagnosis_head.parameters()})
        self.assertEqual([g["lr"] for g in groups], [1e-4, 1e-3])
        before = [[p.detach().clone() for p in group["params"]] for group in groups]
        logits, _, _ = model(torch.randn(4, 3))
        nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1])).backward()
        optimizer.step()
        for old, group in zip(before, groups):
            self.assertTrue(any(not torch.equal(a, p) for a, p in zip(old, group["params"])))

    def test_new_configuration_paths_are_distinct_and_default_baseline_is_preserved(self):
        # Mặc định là quy trình chung: MLP128, head LR 1e-3, 50 epochs; mỗi thay đổi có đường dẫn riêng.
        baseline = run_soft_cbm._default_paths()
        self.assertTrue(baseline[0].endswith(os.path.join("derm7pt", "soft_cbm", "efficientnet_b0_mlp128_e50_headlr0.001", "seed42.pth")))
        candidates = [run_soft_cbm._default_paths(**kwargs) for kwargs in [
            {"diagnosis_head": "linear"}, {"diagnosis_lr": None}, {"epochs": 20},
            {"lr": .0005}, {"weight_decay": .001}, {"concept_loss_weight": .5},
            {"threshold_metric": "balanced_accuracy"}]]
        self.assertEqual(len(set([baseline, *candidates])), 8)
        new = run_soft_cbm._default_paths(diagnosis_head="mlp128", diagnosis_lr=.001, epochs=20)
        self.assertTrue(new[0].endswith(os.path.join("soft_cbm", "efficientnet_b0_mlp128_e20_headlr0.001", "seed42.pth")))

    def test_pilot_then_frozen_mlp_test_and_subprocess_intervention(self):
        with self.pipeline() as (loaders, test_dataset):
            with patch.object(run_soft_cbm, "evaluate_soft_cbm_test", side_effect=AssertionError("Pilot read test")):
                pilot = run_soft_cbm.main(self.args() + ["--skip_test"])
            self.assertFalse(loaders.call_args.kwargs["include_test"])
            test_dataset.assert_not_called()
            self.assertEqual(pilot["mode"], "train_validation")
            self.assertNotIn("test_metrics", pilot)
            self.assertEqual(len(pilot["validation_predictions"]), 203)
            checkpoint, result_path = run_soft_cbm._config_paths(pilot["hyperparameters"])
            frozen_hash = runtime.file_sha256(checkpoint)
            self.assertEqual(pilot["checkpoint_sha256"], frozen_hash)
            config = pilot["hyperparameters"]
            self.assertEqual(config["diagnosis_head"], "MLP(28, 128, 2)")
            self.assertEqual(config["diagnosis_learning_rate"], .001)
            self.assertEqual(pilot["history"][0]["learning_rates"], [1e-4, .001])
            self.assertLess(pilot["history"][1]["learning_rates"][0], 1e-4)
            before = json.loads(Path(result_path).read_text())
            # With an explicit checkpoint, test infers architecture and output path from its config.
            final = run_soft_cbm.main(["--mode", "test", "--checkpoint_path", checkpoint,
                                 "--manifest_path", str(self.manifest), "--device", "cpu", "--overwrite"])
            self.assertEqual(final["mode"], "train_validation_test")
            self.assertEqual(final["decision_threshold"], before["decision_threshold"])
            self.assertEqual(final["best_epoch"], before["best_epoch"])
            self.assertEqual(runtime.file_sha256(checkpoint), frozen_hash)
            self.assertEqual(len(final["test_predictions"]), 395)
            for key in ["validation_predictions", "validation_metrics", "validation_metrics_at_0.5",
                        "validation_concept_metrics", "history"]:
                self.assertEqual(final[key], before[key])
            for row in final["test_predictions"]:
                self.assertEqual(row["y_pred"], int(row["y_prob"] >= final["decision_threshold"]))
        intervention_path = self.root / "intervention.json"
        completed = subprocess.run(
            [sys.executable, "-B", str(ROOT / "experiments/run_soft_cbm.py"), "--mode", "intervention",
             "--checkpoint_path", checkpoint, "--predictions_path", result_path,
             "--manifest_path", str(self.manifest), "--results_path", str(intervention_path)],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"})
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(intervention_path.read_text())
        self.assertEqual(result["num_subsets"], 128)
        self.assertEqual(result["decision_threshold"], pilot["decision_threshold"])
        np.testing.assert_allclose(result["baseline_y_prob"],
                                   [r["y_prob"] for r in final["test_predictions"]], atol=1e-6)

    def test_no_save_pilot_still_saves_checkpoint_without_evaluating_test(self):
        with self.pipeline() as (_, test_dataset):
            with patch.object(run_soft_cbm, "evaluate_soft_cbm_test", side_effect=AssertionError("Pilot read test")):
                result = run_soft_cbm.main(self.args() + ["--skip_test", "--no_save"])
            checkpoint, result_path = run_soft_cbm._config_paths(result["hyperparameters"])
            self.assertTrue(Path(checkpoint).exists())
            self.assertFalse(Path(result_path).exists())
            test_dataset.assert_not_called()

    def test_three_seed_mlp_interventions_find_the_new_cohort_and_separate_output_folder(self):
        with self.pipeline():
            for seed in [42, 123, 2026]:
                run_soft_cbm.main(self.args() + ["--seed", str(seed)])
            with patch.object(run_soft_cbm_interventions, "ROOT", self.root), \
                 patch.object(run_soft_cbm_interventions, "export_report") as report:
                summary = run_soft_cbm_interventions.main([
                    "--manifest_path", str(self.manifest), "--diagnosis_head", "mlp128",
                    "--diagnosis_lr", ".001", "--epochs", "2"])
            self.assertEqual(summary["seeds"], [42, 123, 2026])
            output = report.call_args.args[2]
            self.assertEqual(output, self.root.resolve() / "results/derm7pt/soft_cbm/efficientnet_b0_mlp128_e2_headlr0.001/intervention")
            self.assertFalse((output.parent.parent / "linear_e10/intervention").exists())
            for path in summary["source_paths"]:
                result = json.loads(Path(path).read_text())
                self.assertEqual(result["num_subsets"], 128)
                self.assertEqual(result["hyperparameters"]["diagnosis_head"], "MLP(28, 128, 2)")

    def test_invalid_skip_test_modes_fail_before_reading_a_checkpoint(self):
        for mode in ["test", "intervention"]:
            with self.subTest(mode=mode), contextlib.redirect_stderr(io.StringIO()):
                with patch.object(run_soft_cbm.torch, "load", side_effect=AssertionError("Read checkpoint")):
                    with self.assertRaises(SystemExit):
                        run_soft_cbm.main(["--mode", mode, "--skip_test"])

    def test_bad_head_metadata_and_learning_rate_are_rejected(self):
        config = {"protocol": run_soft_cbm.SOFT_CBM_PROTOCOL, "concept_weighting": "balanced",
                  "concept_state_weights": {c: [1.] * k for c, k in CONCEPT_NUM_CLASSES.items()},
                  "diagnosis_head": "MLP(28, 128, 2)", "diagnosis_hidden_dim": 128,
                  "diagnosis_dropout": .3, "diagnosis_normalization": "LayerNorm",
                  "diagnosis_learning_rate": .001}
        run_soft_cbm.validate_soft_cbm_config(config)
        for change in [{"diagnosis_head": "unknown"}, {"diagnosis_hidden_dim": 64},
                       {"diagnosis_dropout": .2}, {"diagnosis_learning_rate": float("nan")},
                       {"diagnosis_learning_rate": 0.}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                run_soft_cbm.validate_soft_cbm_config({**copy.deepcopy(config), **change})


if __name__ == "__main__":
    unittest.main()
