"""Classification and regression metrics for multi-task evaluation."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list = None
) -> dict:
    """Compute comprehensive classification metrics.

    Args:
        y_true: ground-truth class labels, shape [N]
        y_pred: predicted class labels, shape [N]
        labels: full list of class labels to include in the report
                (even if some are absent in y_true).

    Returns:
        dict with accuracy, macro/weighted precision/recall/f1,
        per-class report, and confusion matrix.
    """
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

    unique_actual = np.unique(y_true)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    # Macro / weighted – use zero_division=0 for safety
    for avg in ("macro", "weighted"):
        try:
            metrics[f"{avg}_precision"] = float(
                precision_score(y_true, y_pred, average=avg, labels=unique_actual, zero_division=0)
            )
            metrics[f"{avg}_recall"] = float(
                recall_score(y_true, y_pred, average=avg, labels=unique_actual, zero_division=0)
            )
            metrics[f"{avg}_f1"] = float(
                f1_score(y_true, y_pred, average=avg, labels=unique_actual, zero_division=0)
            )
        except ValueError:
            metrics[f"{avg}_precision"] = 0.0
            metrics[f"{avg}_recall"] = 0.0
            metrics[f"{avg}_f1"] = 0.0

    # Per-class report
    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    metrics["classification_report"] = report

    # Confusion matrix (always full 8×8)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["confusion_matrix_labels"] = labels

    return metrics


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics: MAE, RMSE, R².

    Args:
        y_true: shape [N] or [N, 1]
        y_pred: shape [N] or [N, 1]
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
