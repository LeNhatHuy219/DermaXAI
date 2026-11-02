# Tên model dùng trong code/thư mục kết quả và tên hiển thị trong báo cáo.
# Artifact cũ ghi tên M1–M5; LEGACY_MODEL_IDS chỉ dùng để đọc lại chúng, run mới ghi tên mới.

DATASET = "derm7pt"

# Backbone ảnh đã hỗ trợ: id dùng trong code/thư mục -> tên hiển thị. Model black-box lấy chính id backbone làm tên;
# CBM/ECBM ghi id backbone ở đầu thư mục cấu hình. Thêm backbone mới bằng cách bổ sung vào đây và src/models.py.
BACKBONES = {"efficientnet_b0": "EfficientNet-B0"}
DEFAULT_BACKBONE = "efficientnet_b0"

MODEL_DISPLAY_NAMES = {
    **BACKBONES,
    "oracle_lr": "Oracle LR (concept thật -> chẩn đoán)",
    "soft_cbm": "Soft CBM",
    "hard_cbm": "Hard CBM",
    "ecbm": "ECBM",
}

LEGACY_MODEL_IDS = {
    "M1_BlackBox_EfficientNetB0": "efficientnet_b0",
    "M2_Oracle_LogisticRegression": "oracle_lr",
    "M3_SoftJointCBM": "soft_cbm",
    "M4_HardJointCBM": "hard_cbm",
    "M5_CategoricalECBM": "ecbm",
}


def canonical_model(name):
    # Đưa tên cũ (M1–M5) và tên mới về cùng một khóa để so khớp artifact.
    return LEGACY_MODEL_IDS.get(name, name)


def backbone_from_config(config):
    # Checkpoint ghi tên hiển thị ("EfficientNet-B0"); checkpoint cũ không ghi gì đều dùng EfficientNet-B0.
    name = config.get("backbone", BACKBONES[DEFAULT_BACKBONE])
    for key, display in BACKBONES.items():
        if name in (key, display):
            return key
    raise ValueError(f"Unknown backbone in checkpoint: {name}")
