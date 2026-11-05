# Đối chứng NGOÀI kế hoạch cho results/derm7pt/resolution_fusion: lợi ích của kết hợp 224 + 512x768 có hơn
# việc lấy trung bình hai mô hình bất kỳ cùng độ phân giải không? Chỉ dùng dự đoán validation đã lưu, không đọc test.


import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_resolution_fusion import CONFIGS, OUT_DIR, PROJECT_ROOT, SEEDS, fuse_pair, mean_sd
from src import runtime

KEYS = ("f1_macro", "roc_auc", "concept_accuracy")


def run(pairs):
    out = {}
    for model, seed_pairs in pairs.items():
        runs = [fuse_pair(model, a, b)["fusion"] for a, b in seed_pairs]
        out[model] = {k: mean_sd([r[k] for r in runs]) for k in KEYS if k in runs[0]}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t4_224_dir", required=True,
                        help="Thư mục output Kaggle v3 (train/<model>_seed<seed>/result.json)")
    args = parser.parse_args()
    res = lambda m, suf, s: os.path.join(PROJECT_ROOT, "results", "derm7pt", CONFIGS[m] + suf, f"seed{s}.json")
    t4 = lambda m, s: os.path.join(args.t4_224_dir, "train", f"{m}_seed{s}", "result.json")
    nxt = dict(zip(SEEDS, SEEDS[1:] + SEEDS[:1]))
    groups = {
        "224_mps+512 (cặp chính)": {m: [(res(m, "", s), res(m, "_r512x768", s)) for s in SEEDS] for m in CONFIGS},
        "224_t4+512 (cặp phụ)": {m: [(t4(m, s), res(m, "_r512x768", s)) for s in SEEDS] for m in CONFIGS},
        "224_mps+224_t4 (cùng seed)": {m: [(res(m, "", s), t4(m, s)) for s in SEEDS] for m in CONFIGS},
        "224_mps+224_mps (khác seed)": {m: [(res(m, "", s), res(m, "", nxt[s])) for s in SEEDS] for m in CONFIGS},
        "512+512 (khác seed)": {m: [(res(m, "_r512x768", s), res(m, "_r512x768", nxt[s])) for s in SEEDS]
                               for m in CONFIGS},
    }
    summary = {"note": "Phân tích NGOÀI kế hoạch (plan.json không có); chỉ để diễn giải, không dùng để chọn.",
               "test_read": False, "pairs": "khác seed: (42,123), (123,2026), (2026,42)",
               "groups": {g: run(p) for g, p in groups.items()}}
    runtime.save_json(os.path.join(OUT_DIR, "control_same_resolution.json"), summary)
    for m in CONFIGS:
        for g, block in summary["groups"].items():
            print(f"{m:9s} {g:30s} " + "  ".join(f"{k}={v['mean']:.3f}±{v['sd']:.3f}" for k, v in block[m].items()))
        print()


if __name__ == "__main__":
    main()
