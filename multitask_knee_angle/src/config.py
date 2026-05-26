"""Configuration module for multi-task knee angle model.

All default hyperparameters and data settings are defined here.
"""

import torch


class Config:
    """Central configuration for the multi-task learning project."""

    # ── Data ───────────────────────────────────────────────────────────
    feature_cols = [
        "LEFT_TA", "LEFT_MG", "LEFT_SOL",
        "LEFT_BF", "LEFT_ST", "LEFT_VL", "LEFT_RF",
    ]
    num_channels = len(feature_cols)          # 7

    cls_label_col = "MODE"
    reg_label_col = "LEFT_KNEE"

    num_classes = 8
    valid_mode_labels = list(range(8))        # 0–7

    # ── Sliding window ─────────────────────────────────────────────────
    window_size: int = 128
    stride: int = 64

    # ── Train / val / test split (chronological) ──────────────────────
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # ── Standardisation ───────────────────────────────────────────────
    scale_reg_target: bool = True

    # ── Model ─────────────────────────────────────────────────────────
    conv_channels: list = [32, 64]             # output channels per conv layer
    conv_kernel: int = 3
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 1
    lstm_bidirectional: bool = True
    fc_hidden_size: int = 64
    dropout: float = 0.3

    # ── Loss ──────────────────────────────────────────────────────────
    lambda_reg: float = 1.0
    reg_loss_type: str = "mse"                 # "mse" or "smooth_l1"
    use_class_weights: bool = True

    # ── Optimiser ─────────────────────────────────────────────────────
    lr: float = 1e-3
    weight_decay: float = 0.0

    # ── Scheduler ─────────────────────────────────────────────────────
    scheduler_factor: float = 0.5
    scheduler_patience: int = 15

    # ── Early stopping ────────────────────────────────────────────────
    early_stop_patience: int = 30
    monitor_metric: str = "val_macro_f1"       # "val_macro_f1" or "val_total_loss"

    # ── Training ──────────────────────────────────────────────────────
    batch_size: int = 64
    epochs: int = 100

    # ── Device ────────────────────────────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Paths (set at runtime) ────────────────────────────────────────
    csv_path: str = ""
    csv_dir: str = "/home/lenovo/project_hzq/claude_project/KNEE_joint_activity/Muti_task_stduy_knee-activaty/data/Processed"
    output_dir: str = "outputs"


def get_default_config(**overrides) -> Config:
    """Return a Config instance with optional overrides applied."""
    cfg = Config()
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            raise KeyError(f"Unknown config key: {k}")
    return cfg
