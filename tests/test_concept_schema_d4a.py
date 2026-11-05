# Kiểm tra schema concept D4a (data/d4a): số trạng thái, one-hot, đường dẫn kết quả riêng,
# và schema gốc giữ nguyên hash để checkpoint cũ vẫn hợp lệ.
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image
import torch

from src import dataset as D
from src.cbm import HardJointCBM, SoftJointCBM
from src.ecbm import CategoricalECBM
from src.protocol import concept_schema_name, load_concept_schema
from experiments import run_ecbm, run_hard_cbm, run_soft_cbm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "manifest.csv"
D4A_MANIFEST = ROOT / "data" / "d4a" / "manifest.csv"
D4A_COUNTS = {"pigment_network": 3, "streaks": 3, "pigmentation": 3, "regression_structures": 2,
              "dots_and_globules": 3, "blue_whitish_veil": 2, "vascular_structures": 3}


class ConceptSchemaD4aTest(unittest.TestCase):
    def tearDown(self):
        # Trả schema gốc cho các test khác chạy cùng tiến trình.
        load_concept_schema(DEFAULT_MANIFEST)

    def test_default_schema_hash_is_unchanged(self):
        saved = json.loads((ROOT / "results/derm7pt/hard_cbm/efficientnet_b0_mlp128_e50_headlr0.001/seed42.json")
                           .read_text(encoding="utf-8"))["concept_schema"]
        current = load_concept_schema(DEFAULT_MANIFEST)
        self.assertEqual(current["sha256"], saved["sha256"])
        self.assertEqual(current["total_states"], 28)
        self.assertEqual(D.TOTAL_CONCEPT_STATES, 28)

    def test_d4a_counts_and_activation(self):
        schema = load_concept_schema(D4A_MANIFEST)
        self.assertEqual(schema["concept_num_classes"], D4A_COUNTS)
        self.assertEqual(schema["total_states"], 19)
        self.assertEqual(dict(D.CONCEPT_NUM_CLASSES), D4A_COUNTS)
        self.assertEqual(D.TOTAL_CONCEPT_STATES, 19)
        load_concept_schema(DEFAULT_MANIFEST)
        self.assertEqual(D.TOTAL_CONCEPT_STATES, 28)
        self.assertEqual(dict(D.CONCEPT_NUM_CLASSES), D.DEFAULT_CONCEPT_NUM_CLASSES)

    def test_d4a_manifest_only_changes_merged_columns(self):
        original, merged = pd.read_csv(DEFAULT_MANIFEST), pd.read_csv(D4A_MANIFEST)
        changed = {"pigmentation", "regression_structures", "vascular_structures",
                   "concept_profile", "is_inconsistent_profile"}
        self.assertEqual(list(original.columns), list(merged.columns))
        for column in original.columns:
            if column not in changed:
                self.assertTrue(original[column].equals(merged[column]), column)
        # Trạng thái chỉ có ở test (pigmentation localized regular) được gộp vào "regular".
        rows = original["pigmentation"] == "localized regular"
        self.assertTrue((merged.loc[rows, "pigmentation"] == "regular").all())
        self.assertEqual(set(merged["regression_structures"]), {"absent", "present"})

    def test_dataset_onehot_uses_its_own_mapping(self):
        load_concept_schema(DEFAULT_MANIFEST)  # dataset không phụ thuộc schema đang kích hoạt
        ds = D.Derm7ptDataset(str(D4A_MANIFEST), str(ROOT), split="valid", transform=lambda image: torch.zeros(1))
        self.assertEqual(ds.total_concept_states, 19)
        self.assertEqual(ds.concept_offsets["vascular_structures"], 16)
        # Không đọc ảnh thật: thay Image.open bằng ảnh giả.
        with patch.object(D.Image, "open", return_value=Image.new("RGB", (4, 4))):
            for index in range(5):
                item = ds[index]
                onehot, indices = item["concept_onehot"], item["concept_indices"]
                self.assertEqual(tuple(onehot.shape), (19,))
                self.assertEqual(float(onehot.sum()), 7.0)
                for group, name in enumerate(D.CONCEPT_NAMES):
                    self.assertEqual(float(onehot[ds.concept_offsets[name] + int(indices[group])]), 1.0)

    def test_models_follow_schema(self):
        load_concept_schema(D4A_MANIFEST)
        soft = SoftJointCBM(D.CONCEPT_NAMES, D.CONCEPT_NUM_CLASSES, pretrained=False)
        hard = HardJointCBM(D.CONCEPT_NAMES, D.CONCEPT_NUM_CLASSES, pretrained=False)
        self.assertEqual(soft.total_concept_states, 19)
        self.assertEqual(hard.total_concept_states, 19)
        ecbm = CategoricalECBM(D.CONCEPT_NAMES, D.CONCEPT_NUM_CLASSES, pretrained=False, hidden_dim=4)
        self.assertEqual(len(ecbm.profiles), 972)

    def test_result_paths_are_separate(self):
        self.assertIsNone(concept_schema_name(DEFAULT_MANIFEST, ROOT))
        self.assertEqual(concept_schema_name(D4A_MANIFEST, ROOT), "d4a")
        base_hard = run_hard_cbm._default_paths(42)
        d4a_hard = run_hard_cbm._default_paths(42, concept_schema="d4a")
        self.assertEqual(Path(d4a_hard[1]).parent.name, Path(base_hard[1]).parent.name + "_d4a")
        base_soft, d4a_soft = run_soft_cbm._default_paths(42), run_soft_cbm._default_paths(42, concept_schema="d4a")
        self.assertEqual(Path(d4a_soft[1]).parent.name, Path(base_soft[1]).parent.name + "_d4a")
        base_ecbm, d4a_ecbm = run_ecbm._default_paths(42), run_ecbm._default_paths(42, concept_schema="d4a")
        self.assertEqual(Path(d4a_ecbm[1]).parent.name, Path(base_ecbm[1]).parent.name + "_d4a")

    def test_head_label_follows_state_count(self):
        self.assertEqual(run_hard_cbm._head_label("mlp128"), "MLP(28, 128, 2)")
        self.assertEqual(run_hard_cbm._head_label("mlp128", 19), "MLP(19, 128, 2)")
        self.assertEqual(run_soft_cbm._head_label("linear", 19), "Linear(19, 2)")
        config = {"diagnosis_head": "MLP(19, 128, 2)",
                  "concept_state_weights": {c: [1.0] * k for c, k in D4A_COUNTS.items()}}
        self.assertEqual(run_hard_cbm._head_preset(config), "mlp128")
        self.assertEqual(run_soft_cbm._head_preset(config), "mlp128")


if __name__ == "__main__":
    unittest.main()
