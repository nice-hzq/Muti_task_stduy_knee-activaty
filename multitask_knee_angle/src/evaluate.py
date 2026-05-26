"""Evaluation script – loads a trained checkpoint and evaluates on test data.

Usage::

    python -m src.evaluate \
        --csv_path data/raw/AB192_Circuit_033_post.csv \
        --checkpoint outputs/checkpoints/best_model.pth
"""

import argparse
import csv
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.config import Config
from src.data_utils import load_single_csv, load_csv_dir, clean_data, validate_mode_labels
from src.dataset import (
    build_windows, split_data_chronologically,
    KneeMultiTaskDataset, FeatureScaler, RegTargetScaler,
)
from src.model import CNNLSTMMultiTask
from src.metrics import compute_classification_metrics, compute_regression_metrics
from src.logger_utils import setup_logger
from src.utils import (
    ensure_dir, save_json,
    plot_confusion_matrix, plot_knee_curve, plot_knee_scatter,
)

logger = logging.getLogger("MultiTaskKnee")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate multi-task knee angle model")
    p.add_argument("--csv_path", type=str, default="",
                   help="Path to a single CSV file")
    p.add_argument("--csv_dir", type=str, default="",
                   help="Directory containing CSV files")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to best_model.pth checkpoint")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--window_size", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()

    ensure_dir(os.path.join(args.output_dir, "logs"))
    ensure_dir(os.path.join(args.output_dir, "figures"))
    ensure_dir(os.path.join(args.output_dir, "predictions"))
    ensure_dir(os.path.join(args.output_dir, "metrics"))
    log_dir = args.output_dir

    logger = setup_logger(log_dir)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    logger.info("Loaded checkpoint: best_epoch=%s, best_metric=%s",
                ckpt.get("best_epoch", "?"), ckpt.get("best_metric", "?"))

    num_classes = ckpt.get("num_classes", 8)
    num_channels = ckpt.get("num_channels", 7)
    window_size = args.window_size
    stride = args.stride

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    if args.csv_path:
        df = load_single_csv(args.csv_path)
    elif args.csv_dir:
        df = load_csv_dir(args.csv_dir)
    else:
        logger.error("Either --csv_path or --csv_dir must be provided.")
        sys.exit(1)

    validate_mode_labels(df, list(range(num_classes)), source="dataset")
    df = clean_data(df)

    feature_cols = ckpt.get("feature_cols", Config.feature_cols)
    features = df[feature_cols].values.astype(np.float32)
    modes = df["MODE"].values.astype(np.int64)
    knee_angles = df["LEFT_KNEE"].values.astype(np.float32)

    # Chronological split to isolate test set
    (tr_f, tr_m, tr_k), (va_f, va_m, va_k), (te_f, te_m, te_k) = \
        split_data_chronologically(features, modes, knee_angles,
                                   Config.train_ratio, Config.val_ratio)

    # Build windows on test portion only
    X_test, y_cls_test, y_reg_test = build_windows(
        te_f, te_m, te_k, window_size, stride)

    logger.info("Test windows: %d", len(X_test))

    # Standardise using saved stats
    feat_scaler = FeatureScaler()
    feat_scaler.scaler.mean_ = np.array(ckpt["train_feature_mean"])
    feat_scaler.scaler.scale_ = np.array(ckpt["train_feature_std"])
    feat_scaler._fitted = True
    X_test = feat_scaler.transform(X_test)

    reg_scaler = RegTargetScaler()
    if ckpt.get("reg_target_scale", 1.0) != 1.0:
        reg_scaler.scaler.mean_ = np.array([ckpt["reg_target_mean"]])
        reg_scaler.scaler.scale_ = np.array([ckpt["reg_target_scale"]])
        reg_scaler._active = True

    y_reg_test_scaled = y_reg_test
    if reg_scaler.is_active:
        y_reg_test_scaled = reg_scaler.transform(y_reg_test)

    # Build model
    model = CNNLSTMMultiTask(
        num_channels=num_channels, window_size=window_size,
        num_classes=num_classes,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Model loaded with %d parameters.", sum(p.numel() for p in model.parameters()))

    # Dataloader
    test_ds = KneeMultiTaskDataset(X_test, y_cls_test, y_reg_test_scaled)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Inference
    all_cls_true, all_cls_pred = [], []
    all_reg_true, all_reg_pred = [], []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            cls_logits, knee_pred = model(x)

            all_cls_true.append(batch["cls_label"].cpu().numpy())
            all_cls_pred.append(cls_logits.argmax(dim=1).cpu().numpy())
            all_reg_true.append(batch["reg_label"].cpu().numpy())
            all_reg_pred.append(knee_pred.cpu().numpy())

    cls_true = np.concatenate(all_cls_true)
    cls_pred = np.concatenate(all_cls_pred)
    reg_true = np.concatenate(all_reg_true)
    reg_pred = np.concatenate(all_reg_pred)

    if reg_scaler.is_active:
        reg_true_inv = reg_scaler.inverse_transform(reg_true).ravel()
        reg_pred_inv = reg_scaler.inverse_transform(reg_pred).ravel()
    else:
        reg_true_inv = reg_true.ravel()
        reg_pred_inv = reg_pred.ravel()

    # ── Classification ────────────────────────────────────────────────
    full_labels = list(range(num_classes))
    cls_metrics = compute_classification_metrics(cls_true, cls_pred, labels=full_labels)

    logger.info("Test Accuracy: %.4f", cls_metrics["accuracy"])
    logger.info("Test Macro-F1: %.4f", cls_metrics["macro_f1"])
    logger.info("Test Weighted-F1: %.4f", cls_metrics["weighted_f1"])

    save_json(cls_metrics, os.path.join(args.output_dir, "metrics", "test_metrics.json"))

    report_csv = os.path.join(args.output_dir, "metrics", "test_classification_report.csv")
    with open(report_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1_score", "support"])
        for label in full_labels:
            info = cls_metrics["classification_report"].get(str(label), {})
            writer.writerow([
                label,
                info.get("precision", 0),
                info.get("recall", 0),
                info.get("f1-score", 0),
                int(info.get("support", 0)),
            ])

    cm_path = os.path.join(args.output_dir, "figures", "confusion_matrix.png")
    plot_confusion_matrix(cls_metrics["confusion_matrix"], full_labels, cm_path)

    # ── Regression ────────────────────────────────────────────────────
    reg_metrics = compute_regression_metrics(reg_true_inv, reg_pred_inv)
    logger.info("Test MAE: %.4f  RMSE: %.4f  R²: %.4f",
                reg_metrics["mae"], reg_metrics["rmse"], reg_metrics["r2"])
    save_json(reg_metrics, os.path.join(args.output_dir, "metrics",
                                        "test_regression_metrics.json"))

    curve_path = os.path.join(args.output_dir, "figures", "knee_angle_curve.png")
    scatter_path = os.path.join(args.output_dir, "figures", "knee_angle_scatter.png")
    plot_knee_curve(reg_true_inv, reg_pred_inv, curve_path)
    plot_knee_scatter(reg_true_inv, reg_pred_inv, scatter_path)

    # ── Predictions CSV ───────────────────────────────────────────────
    pred_csv = os.path.join(args.output_dir, "predictions", "test_predictions.csv")
    with open(pred_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_index", "true_mode", "pred_mode",
            "true_knee_angle", "pred_knee_angle",
        ])
        for i in range(len(cls_true)):
            writer.writerow([
                i, int(cls_true[i]), int(cls_pred[i]),
                float(reg_true_inv[i]), float(reg_pred_inv[i]),
            ])

    logger.info("=== Evaluation complete. Outputs saved under %s ===", args.output_dir)


if __name__ == "__main__":
    main()
