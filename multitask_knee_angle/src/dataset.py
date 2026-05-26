"""Sliding-window dataset construction and train/val/test splitting."""

import logging
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ── Sliding window helper ───────────────────────────────────────────────────

def build_windows(
    features: np.ndarray,         # shape [T, C]
    modes: np.ndarray,            # shape [T]
    knee_angles: np.ndarray,      # shape [T]
    window_size: int = 128,
    stride: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding-window samples from a continuous time series.

    Args:
        features: shape ``[num_timesteps, num_channels]``
        modes: shape ``[num_timesteps]``, integer activity labels
        knee_angles: shape ``[num_timesteps]``
        window_size: number of timesteps per window
        stride: step between consecutive windows

    Returns:
        X: shape ``[num_windows, num_channels, window_size]``
        y_cls: shape ``[num_windows]``, majority-vote MODE per window
        y_reg: shape ``[num_windows]``, last-timestep knee angle
    """
    T = features.shape[0]
    if T < window_size:
        raise ValueError(
            f"Data length ({T}) is shorter than window_size ({window_size})."
        )

    X_list, y_cls_list, y_reg_list = [], [], []

    for start in range(0, T - window_size + 1, stride):
        end = start + window_size

        win_feat = features[start:end, :]                    # [W, C]
        X_list.append(win_feat.T)                            # -> [C, W]

        # Classification label: majority vote
        win_modes = modes[start:end]
        cls_label = int(np.argmax(np.bincount(win_modes.astype(int))))
        y_cls_list.append(cls_label)

        # Regression label: last timestep (方式 A)
        y_reg_list.append(knee_angles[end - 1])

    X = np.stack(X_list, axis=0).astype(np.float32)          # [N, C, W]
    y_cls = np.array(y_cls_list, dtype=np.int64)
    y_reg = np.array(y_reg_list, dtype=np.float32).reshape(-1, 1)

    return X, y_cls, y_reg


# ── Chronological split ─────────────────────────────────────────────────────

def split_data_chronologically(
    features: np.ndarray,
    modes: np.ndarray,
    knee_angles: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple:
    """Split raw time series chronologically (no shuffling).

    Returns train/val/test tuples of (features, modes, knee_angles).
    """
    T = len(features)
    train_end = int(T * train_ratio)
    val_end = train_end + int(T * val_ratio)

    train_feat = features[:train_end]
    train_modes = modes[:train_end]
    train_knee = knee_angles[:train_end]

    val_feat = features[train_end:val_end]
    val_modes = modes[train_end:val_end]
    val_knee = knee_angles[train_end:val_end]

    test_feat = features[val_end:]
    test_modes = modes[val_end:]
    test_knee = knee_angles[val_end:]

    return (
        (train_feat, train_modes, train_knee),
        (val_feat, val_modes, val_knee),
        (test_feat, test_modes, test_knee),
    )


# ── Dataset ─────────────────────────────────────────────────────────────────

class KneeMultiTaskDataset(Dataset):
    """PyTorch Dataset for multi-task knee angle data.

    Each item is a dict with keys:
        - ``x``: ``[C, W]`` float32 tensor
        - ``cls_label``: ``int64`` scalar
        - ``reg_label``: ``[1]`` float32 tensor
    """

    def __init__(
        self,
        X: np.ndarray,          # [N, C, W]
        y_cls: np.ndarray,      # [N]
        y_reg: np.ndarray,      # [N, 1]
    ):
        self.X = torch.from_numpy(X)
        self.y_cls = torch.from_numpy(y_cls).long()
        self.y_reg = torch.from_numpy(y_reg).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict:
        return {
            "x": self.X[idx],             # [C, W]
            "cls_label": self.y_cls[idx], # scalar int64
            "reg_label": self.y_reg[idx], # [1] float32
        }


# ── Standardisation helpers ──────────────────────────────────────────────────

class FeatureScaler:
    """Wraps StandardScaler to handle [N, C, W] -> 2D -> back transformation.

    The scaler is fitted on data reshaped to [N*W, C], so each channel is
    standardised independently across all timesteps.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray) -> None:
        """Fit on X of shape [N, C, W]."""
        N, C, W = X.shape
        X_flat = X.transpose(0, 2, 1).reshape(-1, C)      # [N*W, C]
        self.scaler.fit(X_flat)
        self._fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X of shape [N, C, W]; return same shape."""
        N, C, W = X.shape
        X_flat = X.transpose(0, 2, 1).reshape(-1, C)
        X_scaled = self.scaler.transform(X_flat)
        return X_scaled.reshape(N, W, C).transpose(0, 2, 1)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    @property
    def mean_(self) -> np.ndarray:
        return self.scaler.mean_

    @property
    def scale_(self) -> np.ndarray:
        return self.scaler.scale_


class RegTargetScaler:
    """Optional StandardScaler for regression targets."""

    def __init__(self):
        self.scaler = StandardScaler()
        self._active = False

    def fit(self, y: np.ndarray) -> None:
        """y shape: [N, 1] or [N]."""
        y = y.reshape(-1, 1)
        self.scaler.fit(y)
        self._active = True

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = y.reshape(-1, 1)
        return self.scaler.transform(y).astype(np.float32).reshape(-1, 1)

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        y = y.reshape(-1, 1)
        return self.scaler.inverse_transform(y).astype(np.float32).reshape(-1, 1)

    @property
    def is_active(self) -> bool:
        return self._active
