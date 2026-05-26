"""Prediction script – run inference using a trained checkpoint on new CSV data.

Usage::

    python -m src.predict \
        --csv_path data/raw/new_data.csv \
        --checkpoint outputs/checkpoints/best_model.pth \
        --output outputs/predictions/pred.csv
"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data_utils import load_single_csv, load_csv_dir, clean_data, validate_mode_labels
from src.dataset import build_windows, KneeMultiTaskDataset, FeatureScaler, RegTargetScaler
from src.model import CNNLSTMMultiTask
from src.logger_utils import setup_logger

logger = logging.getLogger("MultiTaskKnee")


def parse_args():
    p = argparse.ArgumentParser(description="Predict with multi-task knee angle model")
    p.add_argument("--csv_path", type=str, default="",
                   help="Path to a single CSV file")
    p.add_argument("--csv_dir", type=str, default="",
                   help="Directory containing CSV files")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default="outputs/predictions/pred.csv")
    p.add_argument("--window_size", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("outputs/logs", "Predict")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    num_classes = ckpt.get("num_classes", 8)
    num_channels = ckpt.get("num_channels", 7)
    feature_cols = ckpt.get("feature_cols", Config.feature_cols)

    # Load data
    if args.csv_path:
        df = load_single_csv(args.csv_path)
    elif args.csv_dir:
        df = load_csv_dir(args.csv_dir)
    else:
        logger.error("Either --csv_path or --csv_dir must be provided.")
        return

    validate_mode_labels(df, list(range(num_classes)), source="dataset")
    df = clean_data(df)

    features = df[feature_cols].values.astype(np.float32)
    modes = df["MODE"].values.astype(np.int64)
    knee_angles = df["LEFT_KNEE"].values.astype(np.float32)

    X, y_cls, y_reg = build_windows(features, modes, knee_angles,
                                    args.window_size, args.stride)
    logger.info("Built %d windows", len(X))

    # Standardise
    feat_scaler = FeatureScaler()
    feat_scaler.scaler.mean_ = np.array(ckpt["train_feature_mean"])
    feat_scaler.scaler.scale_ = np.array(ckpt["train_feature_std"])
    feat_scaler._fitted = True
    X = feat_scaler.transform(X)

    reg_scaler = RegTargetScaler()
    if ckpt.get("reg_target_scale", 1.0) != 1.0:
        reg_scaler.scaler.mean_ = np.array([ckpt["reg_target_mean"]])
        reg_scaler.scaler.scale_ = np.array([ckpt["reg_target_scale"]])
        reg_scaler._active = True

    # Model
    model = CNNLSTMMultiTask(
        num_channels=num_channels, window_size=args.window_size,
        num_classes=num_classes,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Dataset
    class PredDataset(torch.utils.data.Dataset):
        def __init__(self, X_arr):
            self.X = torch.from_numpy(X_arr).float()
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return {"x": self.X[idx]}

    loader = DataLoader(PredDataset(X), batch_size=args.batch_size, shuffle=False)

    all_cls, all_reg = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            cls_logits, knee_pred = model(x)
            all_cls.append(cls_logits.argmax(dim=1).cpu().numpy())
            all_reg.append(knee_pred.cpu().numpy())

    cls_pred = np.concatenate(all_cls)
    reg_pred = np.concatenate(all_reg)
    if reg_scaler.is_active:
        reg_pred = reg_scaler.inverse_transform(reg_pred)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["sample_index", "pred_mode", "pred_knee_angle"])
        for i in range(len(cls_pred)):
            writer.writerow([i, int(cls_pred[i]), float(reg_pred[i][0])])

    logger.info("Predictions saved to %s (%d samples)", args.output, len(cls_pred))


if __name__ == "__main__":
    main()
