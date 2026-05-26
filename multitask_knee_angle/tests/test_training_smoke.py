"""Smoke test: run 1 epoch of training to verify end-to-end flow."""

import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import KneeMultiTaskDataset, FeatureScaler, RegTargetScaler
from src.model import CNNLSTMMultiTask
from src.logger_utils import setup_logger


def make_fake_windows(n: int = 100, C: int = 7, W: int = 128):
    """Generate synthetic windowed data."""
    rng = np.random.RandomState(0)
    X = rng.randn(n, C, W).astype(np.float32)
    y_cls = rng.randint(0, 8, size=n).astype(np.int64)
    y_reg = rng.randn(n, 1).astype(np.float32)
    return X, y_cls, y_reg


def test_training_smoke():
    """Run a single training epoch on fake data and verify no errors."""
    X, y_cls, y_reg = make_fake_windows(100)

    scaler = FeatureScaler()
    X = scaler.fit_transform(X)

    reg_scaler = RegTargetScaler()
    y_reg = reg_scaler.fit_transform(y_reg)

    ds = KneeMultiTaskDataset(X, y_cls, y_reg)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    model = CNNLSTMMultiTask(num_channels=7, window_size=128, num_classes=8)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for batch in loader:
        x = batch["x"]
        cls_target = batch["cls_label"]
        reg_target = batch["reg_label"]

        cls_logits, knee_pred = model(x)

        loss_cls = criterion_cls(cls_logits, cls_target)
        loss_reg = criterion_reg(knee_pred, reg_target)
        loss = loss_cls + 1.0 * loss_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss)
        break  # one batch is enough for smoke

    # Verify loss is computable and finite
    assert loss.item() > 0


def test_checkpoint_save_load():
    """Verify checkpoint can be saved and loaded."""
    import tempfile

    model = CNNLSTMMultiTask(num_channels=7, window_size=128, num_classes=8)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "num_classes": 8,
        "num_channels": 7,
        "window_size": 128,
        "train_feature_mean": [0.0] * 7,
        "train_feature_std": [1.0] * 7,
        "reg_target_mean": 0.0,
        "reg_target_scale": 1.0,
        "feature_cols": ["LEFT_TA", "LEFT_MG", "LEFT_SOL", "LEFT_BF",
                         "LEFT_ST", "LEFT_VL", "LEFT_RF"],
        "best_epoch": 1,
        "best_metric": 0.85,
    }

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        torch.save(ckpt, f.name)
        ckpt_path = f.name

    try:
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model2 = CNNLSTMMultiTask(num_channels=7, window_size=128, num_classes=8)
        model2.load_state_dict(loaded["model_state_dict"])

        # Verify outputs match
        model.eval()
        model2.eval()
        x = torch.randn(4, 7, 128)
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)
        assert torch.equal(out1[0], out2[0])
        assert torch.equal(out1[1], out2[1])
    finally:
        os.unlink(ckpt_path)


def test_log_file_generated():
    """Verify logger creates a log file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = setup_logger(tmpdir, "test")
        logger.info("Test log message")

        log_file = os.path.join(tmpdir, "train.log")
        assert os.path.isfile(log_file)

        with open(log_file, "r") as f:
            content = f.read()
        assert "Test log message" in content
