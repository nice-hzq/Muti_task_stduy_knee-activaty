"""General-purpose utility functions."""

import os
import json
from typing import Any, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def ensure_dir(path: str) -> str:
    """Create directory if it does not exist; return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save a dictionary as a JSON file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def count_parameters(model) -> int:
    """Return the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Plotting helpers ────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: list, labels: list, save_path: str) -> None:
    """Plot and save a confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_knee_curve(y_true: np.ndarray, y_pred: np.ndarray,
                    save_path: str, max_points: int = 500) -> None:
    """Plot true vs predicted knee angle over sample index."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if len(y_true) > max_points:
        step = len(y_true) // max_points
        idx = np.arange(0, len(y_true), step)
    else:
        idx = np.arange(len(y_true))

    plt.figure(figsize=(12, 5))
    plt.plot(idx, y_true[idx], label="True", alpha=0.7, linewidth=0.8)
    plt.plot(idx, y_pred[idx], label="Predicted", alpha=0.7, linewidth=0.8)
    plt.xlabel("Sample index (subsampled)")
    plt.ylabel("Knee Angle (deg)")
    plt.title("Knee Angle: True vs Predicted")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_knee_scatter(y_true: np.ndarray, y_pred: np.ndarray,
                      save_path: str) -> None:
    """Plot predicted vs true scatter with identity line."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=4)
    r_min = min(y_true.min(), y_pred.min())
    r_max = max(y_true.max(), y_pred.max())
    plt.plot([r_min, r_max], [r_min, r_max], "r--", linewidth=1)
    plt.xlabel("True Knee Angle (deg)")
    plt.ylabel("Predicted Knee Angle (deg)")
    plt.title("Predicted vs True Knee Angle")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
