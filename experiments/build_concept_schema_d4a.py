# Tạo schema concept D4a (gộp trạng thái hiếm) ở data/d4a/: manifest.csv + label_mapping.json.
# Quy tắc gộp lấy nguyên văn từ results/derm7pt/concept_merge_check/plan.json (đã chốt trước khi kiểm tra).
# Chỉ đổi nhãn chữ ở 3 cột concept, rồi tính lại concept_profile và is_inconsistent_profile như manifest gốc.


import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import CONCEPT_NAMES

import pandas as pd

PLAN = os.path.join(PROJECT_ROOT, "results", "derm7pt", "concept_merge_check", "plan.json")
PROFILE_SEPARATOR = " | "


def merged_schema(original_mapping, merge):
    # Trạng thái mới xếp theo thứ tự chữ cái như label_mapping gốc ("absent" đứng đầu).
    mapping = {}
    for concept in CONCEPT_NAMES:
        target = merge.get(concept, {name: name for name in original_mapping[concept]})
        if set(target) != set(original_mapping[concept]):
            raise ValueError(f"Merge rule for {concept} must cover every original state")
        mapping[concept] = {name: index for index, name in enumerate(sorted(set(target.values())))}
    return mapping


def build(output_dir):
    plan = json.load(open(PLAN, encoding="utf-8"))
    merge = {k: v for k, v in plan["merge_D4a"].items() if isinstance(v, dict)}
    original_mapping = json.load(open(os.path.join(PROJECT_ROOT, "data", "label_mapping.json"), encoding="utf-8"))
    manifest = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "manifest.csv"), dtype={"is_inconsistent_profile": bool})

    # Kiểm tra cách tính hai cột dẫn xuất của manifest gốc trước khi tính lại cho D4a.
    profile = manifest[CONCEPT_NAMES].astype(str).agg(PROFILE_SEPARATOR.join, axis=1)
    inconsistent = manifest.groupby(profile)["diagnosis_binary"].transform("nunique") > 1
    if not (profile.equals(manifest["concept_profile"]) and inconsistent.equals(manifest["is_inconsistent_profile"])):
        raise ValueError("Original manifest profile columns do not follow the expected rule")

    d4a = manifest.copy()
    for concept, rule in merge.items():
        d4a[concept] = d4a[concept].map(rule)
        if d4a[concept].isna().any():
            raise ValueError(f"Unmapped state in {concept}")
    d4a["concept_profile"] = d4a[CONCEPT_NAMES].astype(str).agg(PROFILE_SEPARATOR.join, axis=1)
    d4a["is_inconsistent_profile"] = d4a.groupby("concept_profile")["diagnosis_binary"].transform("nunique") > 1

    mapping = merged_schema(original_mapping, merge)
    os.makedirs(output_dir, exist_ok=True)
    d4a.to_csv(os.path.join(output_dir, "manifest.csv"), index=False)
    with open(os.path.join(output_dir, "label_mapping.json"), "w", encoding="utf-8") as handle:
        json.dump(mapping, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    counts = {c: len(mapping[c]) for c in CONCEPT_NAMES}
    print(json.dumps({"concept_num_classes": counts, "total_states": sum(counts.values()),
                      "inconsistent_fraction_all_splits": float(d4a["is_inconsistent_profile"].mean())}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=os.path.join(PROJECT_ROOT, "data", "d4a"))
    build(parser.parse_args().output_dir)
