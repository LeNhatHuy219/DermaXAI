# Tiện ích dùng chung cho các runner: seed, thiết bị, ghi kết quả, hash file và tổng hợp qua seeds.
import hashlib
import json
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch

from src.dataset import CONCEPT_NAMES


# Gán seed cho Python/NumPy/PyTorch để kiểm soát nguồn ngẫu nhiên của run.
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# Chọn thiết bị theo CLI; auto ưu tiên CUDA, rồi MPS, rồi CPU.
def get_device(device_arg: str = "auto") -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# SHA-256 nội dung file để ghép đúng manifest/checkpoint với kết quả.
def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Tạo thư mục cha và từ chối ghi đè file đã có nếu chưa bật overwrite.
def prepare_output(path: str, overwrite: bool) -> None:
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"Artifact already exists: {path}. Choose another path or pass --overwrite.")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# Mean và sample SD qua seeds, kèm giá trị từng seed.
def mean_sd(values) -> Dict[str, Any]:
    values = [float(v) for v in values]
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)), "values": values}


# Ghép case IDs/GT/xác suất/dự đoán thành records để lưu và kiểm tra từng ca; concept là tùy chọn.
def sample_prediction_records(df, y_pred: np.ndarray, y_prob: np.ndarray,
                              concept_outputs: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    records = []
    for index, ((_, row), pred, prob) in enumerate(zip(df.iterrows(), y_pred, y_prob)):
        records.append({
            "case_num": int(row["case_num"]),
            "source_index": int(row["source_index"]),
            "split": str(row["split"]),
            "diagnosis": str(row["diagnosis"]),
            "is_inconsistent": bool(row["is_inconsistent_profile"]),
            "y_true": int(row["diagnosis_binary"]),
            "y_pred": int(pred),
            "y_prob": float(prob),
        })
        if concept_outputs is not None:
            records[-1].update({
                "concept_true": {c: int(concept_outputs["true"][c][index]) for c in CONCEPT_NAMES},
                "concept_pred": {c: int(concept_outputs["pred"][c][index]) for c in CONCEPT_NAMES},
                "concept_probabilities": {c: concept_outputs["probabilities"][c][index].tolist()
                                          for c in CONCEPT_NAMES},
            })
    return records
