"""Tests for dataset module."""

import numpy as np
import torch
import pytest

from src.dataset import (
    build_windows, split_data_chronologically,
    KneeMultiTaskDataset, FeatureScaler, RegTargetScaler,
)


def make_fake_data(T: int = 500):
    """Create fake time-series data."""
    rng = np.random.RandomState(42)
    features = rng.randn(T, 7).astype(np.float32)
    modes = rng.randint(0, 8, size=T).astype(np.int64)
    knee = rng.randn(T).astype(np.float32) * 10.0 + 50.0
    return features, modes, knee


class TestBuildWindows:
    def test_output_shapes(self):
        feat, modes, knee = make_fake_data(500)
        X, y_cls, y_reg = build_windows(feat, modes, knee, window_size=128, stride=64)

        assert X.shape[0] == y_cls.shape[0] == y_reg.shape[0]
        assert X.shape[1:] == (7, 128)          # [N, channels, window]
        assert y_reg.shape[1] == 1

    def test_window_count(self):
        feat, modes, knee = make_fake_data(500)
        X, y_cls, y_reg = build_windows(feat, modes, knee, window_size=128, stride=64)

        expected = (500 - 128) // 64 + 1
        assert len(X) == expected

    def test_cls_label_is_majority(self):
        feat = np.zeros((10, 7), dtype=np.float32)
        modes = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int64)
        knee = np.zeros(10, dtype=np.float32)
        _, y_cls, _ = build_windows(feat, modes, knee, window_size=3, stride=1)

        # window 0: modes[0:3] = [0,0,0] -> majority 0
        assert y_cls[0] == 0
        # window 1: modes[1:4] = [0,0,1] -> majority 0
        assert y_cls[1] == 0

    def test_reg_label_is_last_timestep(self):
        feat = np.zeros((10, 7), dtype=np.float32)
        modes = np.zeros(10, dtype=np.int64)
        knee = np.arange(10, dtype=np.float32)
        _, _, y_reg = build_windows(feat, modes, knee, window_size=3, stride=1)

        # window 0: end-1 = 2
        assert y_reg[0][0] == 2.0
        # window 1: end-1 = 3
        assert y_reg[1][0] == 3.0

    def test_raises_when_data_too_short(self):
        feat = np.zeros((50, 7), dtype=np.float32)
        modes = np.zeros(50, dtype=np.int64)
        knee = np.zeros(50, dtype=np.float32)
        with pytest.raises(ValueError):
            build_windows(feat, modes, knee, window_size=128, stride=64)


class TestSplitDataChronologically:
    def test_split_ratios(self):
        feat, modes, knee = make_fake_data(1000)
        (tr, _, _), (va, _, _), (te, _, _) = split_data_chronologically(
            feat, modes, knee, train_ratio=0.70, val_ratio=0.15)

        assert len(tr) == 700
        assert len(va) == 150
        assert len(te) == 150

    def test_no_overlap(self):
        feat, modes, knee = make_fake_data(100)
        # Use values that uniquely identify each sample
        feat = np.arange(100, dtype=np.float32).reshape(-1, 1).repeat(7, axis=1)
        modes = np.arange(100, dtype=np.int64)
        knee = np.arange(100, dtype=np.float32)

        (tr, tr_m, _), (va, va_m, _), (te, te_m, _) = split_data_chronologically(
            feat, modes, knee, train_ratio=0.70, val_ratio=0.15)

        total = len(tr) + len(va) + len(te)
        assert total == 100


class TestKneeMultiTaskDataset:
    def test_len(self):
        X = np.random.randn(10, 7, 128).astype(np.float32)
        y_cls = np.zeros(10, dtype=np.int64)
        y_reg = np.zeros((10, 1), dtype=np.float32)
        ds = KneeMultiTaskDataset(X, y_cls, y_reg)
        assert len(ds) == 10

    def test_getitem(self):
        X = np.random.randn(5, 7, 128).astype(np.float32)
        y_cls = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        y_reg = np.random.randn(5, 1).astype(np.float32)
        ds = KneeMultiTaskDataset(X, y_cls, y_reg)

        item = ds[0]
        assert isinstance(item, dict)
        assert item["x"].shape == (7, 128)
        assert item["x"].dtype == torch.float32
        assert item["cls_label"].dtype == torch.int64
        assert item["reg_label"].dtype == torch.float32
        assert item["reg_label"].shape == (1,)


class TestFeatureScaler:
    def test_fit_transform_shape(self):
        X = np.random.randn(100, 7, 128).astype(np.float32)
        scaler = FeatureScaler()
        X_scaled = scaler.fit_transform(X)
        assert X_scaled.shape == X.shape

    def test_zero_mean_unit_variance(self):
        X = np.random.randn(200, 7, 64).astype(np.float32) * 2.0 + 5.0
        scaler = FeatureScaler()
        X_scaled = scaler.fit_transform(X)

        # Reshape to [N*W, C] for per-channel check
        N, C, W = X_scaled.shape
        X_flat = X_scaled.transpose(0, 2, 1).reshape(-1, C)
        means = X_flat.mean(axis=0)
        stds = X_flat.std(axis=0)
        assert np.allclose(means, 0, atol=1e-5)
        assert np.allclose(stds, 1, atol=1e-5)


class TestRegTargetScaler:
    def test_noop_when_not_fitted(self):
        scaler = RegTargetScaler()
        assert not scaler.is_active

    def test_inverse_roundtrip(self):
        y = np.random.randn(100, 1).astype(np.float32) * 10 + 50
        scaler = RegTargetScaler()
        y_scaled = scaler.fit_transform(y)
        y_back = scaler.inverse_transform(y_scaled)
        assert np.allclose(y, y_back, atol=1e-4)
