# """Training script for the multi-task knee angle model.

# Usage::

#     python -m src.train --csv_path data/raw/AB192_Circuit_033_post.csv
#     python -m src.train --csv_dir ../data/Processed
# """

import argparse
import csv
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import Config, get_default_config
from src.data_utils import (
    load_single_csv, load_csv_dir, inspect_data, clean_data,
    validate_mode_labels,
)
from src.dataset import (
    build_windows, split_data_chronologically,
    KneeMultiTaskDataset, FeatureScaler, RegTargetScaler,
)
from src.model import CNNLSTMMultiTask
from src.metrics import compute_classification_metrics, compute_regression_metrics
from src.logger_utils import setup_logger
from src.utils import ensure_dir, save_json, count_parameters

logger = logging.getLogger("MultiTaskKnee")


# ── Argument parsing ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train multi-task knee angle model")

    # Data
    p.add_argument("--csv_path", type=str, default="",
                   help="Path to a single CSV file")
    p.add_argument("--csv_dir", type=str, default="",
                   help="Directory containing CSV files (used if --csv_path not set)")

    # Output
    p.add_argument("--output_dir", type=str, default="outputs")

    # Window
    p.add_argument("--window_size", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)

    # Training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_reg", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=1)

    # Misc
    p.add_argument("--scale_reg_target", type=str, default="True")
    p.add_argument("--device", type=str, default="")

    return p.parse_args()


# ── Training epoch ──────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion_cls, criterion_reg,
                lambda_reg, device):
    model.train()
    total_loss = total_cls = total_reg = 0.0
    n = 0

    for batch in loader:
        x = batch["x"].to(device)
        y_cls = batch["cls_label"].to(device)
        y_reg = batch["reg_label"].to(device)

        cls_logits, knee_pred = model(x)

        loss_cls = criterion_cls(cls_logits, y_cls)
        loss_reg = criterion_reg(knee_pred, y_reg)
        loss = loss_cls + lambda_reg * loss_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        n += bs
        total_loss += loss.item() * bs
        total_cls += loss_cls.item() * bs
        total_reg += loss_reg.item() * bs

    return total_loss / n, total_cls / n, total_reg / n


# ── Validation epoch ────────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(model, loader, criterion_cls, criterion_reg, lambda_reg, device,
              reg_scaler: RegTargetScaler = None):
    model.eval()
    total_loss = total_cls = total_reg = 0.0
    n = 0

    all_cls_true, all_cls_pred = [], []
    all_reg_true, all_reg_pred = [], []

    for batch in loader:
        x = batch["x"].to(device)
        y_cls = batch["cls_label"].to(device)
        y_reg = batch["reg_label"].to(device)

        cls_logits, knee_pred = model(x)

        loss_cls = criterion_cls(cls_logits, y_cls)
        loss_reg = criterion_reg(knee_pred, y_reg)
        loss = loss_cls + lambda_reg * loss_reg

        bs = x.size(0)
        n += bs
        total_loss += loss.item() * bs
        total_cls += loss_cls.item() * bs
        total_reg += loss_reg.item() * bs

        all_cls_true.append(y_cls.cpu().numpy())
        all_cls_pred.append(cls_logits.argmax(dim=1).cpu().numpy())
        all_reg_true.append(y_reg.cpu().numpy())
        all_reg_pred.append(knee_pred.cpu().numpy())

    cls_true = np.concatenate(all_cls_true)
    cls_pred = np.concatenate(all_cls_pred)
    reg_true = np.concatenate(all_reg_true)
    reg_pred = np.concatenate(all_reg_pred)

    # Inverse-transform regression if needed
    if reg_scaler is not None and reg_scaler.is_active:
        reg_true_inv = reg_scaler.inverse_transform(reg_true)
        reg_pred_inv = reg_scaler.inverse_transform(reg_pred)
    else:
        reg_true_inv = reg_true
        reg_pred_inv = reg_pred

    cls_metrics = compute_classification_metrics(
        cls_true, cls_pred, labels=list(range(8))
    )
    reg_metrics = compute_regression_metrics(reg_true_inv, reg_pred_inv)

    return {
        "total_loss": total_loss / n,
        "cls_loss": total_cls / n,
        "reg_loss": total_reg / n,
        "acc": cls_metrics["accuracy"],
        "macro_f1": cls_metrics["macro_f1"],
        "weighted_f1": cls_metrics["weighted_f1"],
        "mae": reg_metrics["mae"],
        "rmse": reg_metrics["rmse"],
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve config
    cfg = Config()
    for attr in (
        "window_size", "stride", "batch_size", "epochs", "lr",
        "lambda_reg", "dropout", "output_dir",
    ):
        setattr(cfg, attr, getattr(args, attr))
    cfg.lstm_hidden_size = args.hidden_size
    cfg.lstm_num_layers = args.num_layers
    cfg.scale_reg_target = args.scale_reg_target.lower() in ("true", "1", "yes")
    if args.device:
        cfg.device = args.device

    # ── Directories & logging ────────────────────────────────────────
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ensure_dir(os.path.join(cfg.output_dir, "logs", run_time))
    ckpt_dir = ensure_dir(os.path.join(cfg.output_dir, "checkpoints"))
    ensure_dir(os.path.join(cfg.output_dir, "figures"))
    ensure_dir(os.path.join(cfg.output_dir, "predictions"))
    ensure_dir(os.path.join(cfg.output_dir, "metrics"))

    logger = setup_logger(log_dir)
    logger.info("Run timestamp: %s", run_time)
    logger.info("Device: %s", cfg.device)
    logger.info("Configuration:\n  %s",
                "\n  ".join(f"{k}={v}" for k, v in sorted(vars(args).items())))

    # ── Load data ─────────────────────────────────────────────────────
    csv_path = args.csv_path or cfg.csv_path
    csv_dir = args.csv_dir or cfg.csv_dir
    if csv_path:
        df = load_single_csv(csv_path)
    elif csv_dir:
        df = load_csv_dir(csv_dir)
    else:
        logger.error("Either --csv_path or --csv_dir must be provided.")
        sys.exit(1)

    # Validate MODE labels
    validate_mode_labels(df, cfg.valid_mode_labels, source="dataset")

    # Clean & inspect
    df = clean_data(df)
    info = inspect_data(df)
    logger.info("Data summary:\n  %s",
                "\n  ".join(f"{k}={v}" for k, v in info.items()))
    save_json(info, os.path.join(log_dir, "data_summary.json"))

    # ── Extract arrays ────────────────────────────────────────────────
    features = df[cfg.feature_cols].values.astype(np.float32)
    modes = df[cfg.cls_label_col].values.astype(np.int64)
    knee_angles = df[cfg.reg_label_col].values.astype(np.float32)

    logger.info("Feature shape: %s, MODE range: [%d, %d], KNEE range: [%.2f, %.2f]",
                features.shape, modes.min(), modes.max(),
                knee_angles.min(), knee_angles.max())

    # ── Chronological split ───────────────────────────────────────────
    (tr_f, tr_m, tr_k), (va_f, va_m, va_k), (te_f, te_m, te_k) = \
        split_data_chronologically(features, modes, knee_angles,
                                   cfg.train_ratio, cfg.val_ratio)

    logger.info("After chronological split — train: %d, val: %d, test: %d",
                len(tr_f), len(va_f), len(te_f))

    # ── Build windows ─────────────────────────────────────────────────
    X_train, y_cls_train, y_reg_train = build_windows(
        tr_f, tr_m, tr_k, cfg.window_size, cfg.stride)
    X_val, y_cls_val, y_reg_val = build_windows(
        va_f, va_m, va_k, cfg.window_size, cfg.stride)
    X_test, y_cls_test, y_reg_test = build_windows(
        te_f, te_m, te_k, cfg.window_size, cfg.stride)

    logger.info("Windows — train: %d, val: %d, test: %d",
                len(X_train), len(X_val), len(X_test))
    logger.info("X_train shape: %s", X_train.shape)

    # ── Standardisation ───────────────────────────────────────────────
    feat_scaler = FeatureScaler()
    X_train = feat_scaler.fit_transform(X_train)
    X_val = feat_scaler.transform(X_val)
    X_test_scaled = feat_scaler.transform(X_test)

    reg_scaler = RegTargetScaler()
    if cfg.scale_reg_target:
        y_reg_train = reg_scaler.fit_transform(y_reg_train)
        y_reg_val = reg_scaler.transform(y_reg_val)
        y_reg_test = reg_scaler.transform(y_reg_test)
    else:
        y_reg_train = y_reg_train.reshape(-1, 1).astype(np.float32)
        y_reg_val = y_reg_val.reshape(-1, 1).astype(np.float32)
        y_reg_test = y_reg_test.reshape(-1, 1).astype(np.float32)

    # ── Datasets & loaders ────────────────────────────────────────────
    train_ds = KneeMultiTaskDataset(X_train, y_cls_train, y_reg_train)
    val_ds = KneeMultiTaskDataset(X_val, y_cls_val, y_reg_val)
    test_ds = KneeMultiTaskDataset(X_test_scaled, y_cls_test, y_reg_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────
    model = CNNLSTMMultiTask(
        num_channels=cfg.num_channels,
        window_size=cfg.window_size,
        num_classes=cfg.num_classes,
        conv_channels=cfg.conv_channels,
        conv_kernel=cfg.conv_kernel,
        lstm_hidden_size=cfg.lstm_hidden_size,
        lstm_num_layers=cfg.lstm_num_layers,
        lstm_bidirectional=cfg.lstm_bidirectional,
        fc_hidden_size=cfg.fc_hidden_size,
        dropout=cfg.dropout,
    ).to(cfg.device)

    logger.info("Model:\n%s", model)
    logger.info("Trainable parameters: %d", count_parameters(model))

    # ── Loss ──────────────────────────────────────────────────────────
    if cfg.use_class_weights:
        cls_counts = np.bincount(y_cls_train.ravel(), minlength=cfg.num_classes)
        # Avoid exploding weights for classes absent from the training set
        cls_counts = np.maximum(cls_counts, 1)
        cls_weights = cfg.num_classes / (cls_counts.astype(np.float32))
        cls_weights = cls_weights / cls_weights.sum() * cfg.num_classes
        cls_weights = torch.tensor(cls_weights, dtype=torch.float32).to(cfg.device)
        logger.info("Class weights: %s", cls_weights.tolist())
    else:
        cls_weights = None

    criterion_cls = nn.CrossEntropyLoss(weight=cls_weights)
    if cfg.reg_loss_type == "smooth_l1":
        criterion_reg = nn.SmoothL1Loss()
    else:
        criterion_reg = nn.MSELoss()

    # ── Optimiser & scheduler ─────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
    )

    # ── Training loop with early stopping ─────────────────────────────
    history_csv = os.path.join(log_dir, "training_history.csv")
    with open(history_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_total_loss", "train_cls_loss", "train_reg_loss",
            "val_total_loss", "val_cls_loss", "val_reg_loss",
            "val_acc", "val_macro_f1", "val_mae", "val_rmse", "lr",
        ])

    best_metric_val = -float("inf") if cfg.monitor_metric == "val_macro_f1" else float("inf")
    best_epoch = 0
    patience_counter = 0
    best_model_path = os.path.join(ckpt_dir, "best_model.pth")

    for epoch in range(1, cfg.epochs + 1):
        train_total, train_cls, train_reg = train_epoch(
            model, train_loader, optimizer, criterion_cls, criterion_reg,
            cfg.lambda_reg, cfg.device,
        )

        val_res = val_epoch(
            model, val_loader, criterion_cls, criterion_reg,
            cfg.lambda_reg, cfg.device, reg_scaler,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        logger.info(
            "Epoch %3d/%d | T-loss %.4f T-cls %.4f T-reg %.4f | "
            "V-loss %.4f V-cls %.4f V-reg %.4f | "
            "V-acc %.4f V-f1 %.4f V-mae %.4f V-rmse %.4f | lr %.1e",
            epoch, cfg.epochs, train_total, train_cls, train_reg,
            val_res["total_loss"], val_res["cls_loss"], val_res["reg_loss"],
            val_res["acc"], val_res["macro_f1"], val_res["mae"], val_res["rmse"],
            current_lr,
        )

        # Write history CSV
        with open(history_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_total, train_cls, train_reg,
                val_res["total_loss"], val_res["cls_loss"], val_res["reg_loss"],
                val_res["acc"], val_res["macro_f1"], val_res["mae"], val_res["rmse"],
                current_lr,
            ])

        # Scheduler step
        if cfg.monitor_metric == "val_macro_f1":
            scheduler.step(val_res["macro_f1"])
        else:
            scheduler.step(val_res["total_loss"])

        # Early stopping check
        monitor_val = (val_res["macro_f1"] if cfg.monitor_metric == "val_macro_f1"
                       else -val_res["total_loss"])
        improved = monitor_val > best_metric_val
        if improved:
            best_metric_val = monitor_val
            best_epoch = epoch
            patience_counter = 0

            # Save checkpoint
            ckpt = {
                "model_state_dict": model.state_dict(),
                "config": {k: v for k, v in vars(cfg).items()
                           if not k.startswith("_") and not callable(v)},
                "feature_cols": cfg.feature_cols,
                "train_feature_mean": feat_scaler.mean_.tolist(),
                "train_feature_std": feat_scaler.scale_.tolist(),
                "reg_target_mean": (reg_scaler.scaler.mean_[0]
                                    if reg_scaler.is_active else 0.0),
                "reg_target_scale": (reg_scaler.scaler.scale_[0]
                                     if reg_scaler.is_active else 1.0),
                "best_epoch": best_epoch,
                "best_metric": float(best_metric_val),
                "num_classes": cfg.num_classes,
                "window_size": cfg.window_size,
                "num_channels": cfg.num_channels,
            }
            torch.save(ckpt, best_model_path)
            logger.info("  >> Best model saved (epoch %d, %s=%.4f)",
                        best_epoch, cfg.monitor_metric, float(best_metric_val))
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

    # ── Final test evaluation ─────────────────────────────────────────
    logger.info("=== Final evaluation on test set ===")
    logger.info("Loading best model from %s", best_model_path)

    ckpt = torch.load(best_model_path, map_location=cfg.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_res = val_epoch(
        model, test_loader, criterion_cls, criterion_reg,
        cfg.lambda_reg, cfg.device, reg_scaler,
    )

    logger.info("Test — Acc: %.4f  Macro-F1: %.4f  Weighted-F1: %.4f",
                test_res["acc"], test_res["macro_f1"], test_res["weighted_f1"])
    logger.info("Test — MAE: %.4f  RMSE: %.4f",
                test_res["mae"], test_res["rmse"])

    # Save test metrics JSON
    save_json(test_res, os.path.join(cfg.output_dir, "metrics", "test_metrics.json"))

    # Detailed classification + regression on full test set
    _final_test_report(model, test_loader, cfg, reg_scaler)


def _final_test_report(model, loader, cfg: Config,
                       reg_scaler: RegTargetScaler):
    """Run detailed test evaluation and save all output files."""
    logger.info("Generating detailed test reports...")

    model.eval()
    all_cls_true, all_cls_pred = [], []
    all_reg_true, all_reg_pred = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(cfg.device)
            y_cls = batch["cls_label"]
            y_reg = batch["reg_label"]

            cls_logits, knee_pred = model(x)

            all_cls_true.append(y_cls.cpu().numpy())
            all_cls_pred.append(cls_logits.argmax(dim=1).cpu().numpy())
            all_reg_true.append(y_reg.cpu().numpy())
            all_reg_pred.append(knee_pred.cpu().numpy())

    cls_true = np.concatenate(all_cls_true)
    cls_pred = np.concatenate(all_cls_pred)
    reg_true = np.concatenate(all_reg_true)
    reg_pred = np.concatenate(all_reg_pred)

    # Inverse transform regression
    if reg_scaler.is_active:
        reg_true_inv = reg_scaler.inverse_transform(reg_true).ravel()
        reg_pred_inv = reg_scaler.inverse_transform(reg_pred).ravel()
    else:
        reg_true_inv = reg_true.ravel()
        reg_pred_inv = reg_pred.ravel()

    # ── Classification report ─────────────────────────────────────────
    full_labels = list(range(cfg.num_classes))
    cls_metrics = compute_classification_metrics(cls_true, cls_pred, labels=full_labels)

    save_json(cls_metrics, os.path.join(cfg.output_dir, "metrics", "test_metrics.json"))

    # CSV report
    report_csv = os.path.join(cfg.output_dir, "metrics", "test_classification_report.csv")
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
    logger.info("Saved classification report -> %s", report_csv)

    # Confusion matrix figure
    from src.utils import plot_confusion_matrix
    cm_path = os.path.join(cfg.output_dir, "figures", "confusion_matrix.png")
    plot_confusion_matrix(cls_metrics["confusion_matrix"], full_labels, cm_path)
    logger.info("Saved confusion matrix -> %s", cm_path)

    # ── Regression metrics ────────────────────────────────────────────
    reg_metrics = compute_regression_metrics(reg_true_inv, reg_pred_inv)
    save_json(reg_metrics, os.path.join(cfg.output_dir, "metrics",
                                        "test_regression_metrics.json"))
    logger.info("Test regression — MAE: %.4f  RMSE: %.4f  R²: %.4f",
                reg_metrics["mae"], reg_metrics["rmse"], reg_metrics["r2"])

    # Curve & scatter plots
    from src.utils import plot_knee_curve, plot_knee_scatter
    curve_path = os.path.join(cfg.output_dir, "figures", "knee_angle_curve.png")
    scatter_path = os.path.join(cfg.output_dir, "figures", "knee_angle_scatter.png")
    plot_knee_curve(reg_true_inv, reg_pred_inv, curve_path)
    plot_knee_scatter(reg_true_inv, reg_pred_inv, scatter_path)
    logger.info("Saved knee angle plots -> %s, %s", curve_path, scatter_path)

    # ── Predictions CSV ───────────────────────────────────────────────
    pred_csv = os.path.join(cfg.output_dir, "predictions", "test_predictions.csv")
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
    logger.info("Saved predictions -> %s", pred_csv)

    logger.info("=== All outputs saved under %s ===", cfg.output_dir)


if __name__ == "__main__":
    main()
