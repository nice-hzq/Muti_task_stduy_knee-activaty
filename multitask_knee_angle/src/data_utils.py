"""Data reading, validation, and inspection utilities."""

import os
import logging
from glob import glob
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Required columns ────────────────────────────────────────────────────────

REQUIRED_COLS = [
    "LEFT_TA", "LEFT_MG", "LEFT_SOL",
    "LEFT_BF", "LEFT_ST", "LEFT_VL", "LEFT_RF",
    "LEFT_KNEE", "MODE",
]


# ── CSV loading ─────────────────────────────────────────────────────────────

def load_single_csv(csv_path: str) -> pd.DataFrame:
    """Load and perform basic validation on a single CSV.

    Raises:
        FileNotFoundError: if *csv_path* does not exist.
        ValueError: on missing required columns or invalid MODE labels.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Loaded %s — %d rows × %d cols", csv_path, len(df), len(df.columns))

    _validate_columns(df, csv_path)
    return df


def load_csv_dir(csv_dir: str) -> pd.DataFrame:
    """Load all CSV files under *csv_dir* and concatenate them.

    The directory is scanned recursively; every ``.csv`` file is read.
    """
    if not os.path.isdir(csv_dir):
        raise NotADirectoryError(f"Not a directory: {csv_dir}")

    pattern = os.path.join(csv_dir, "*.csv")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {csv_dir}")

    logger.info("Found %d CSV file(s) in %s", len(files), csv_dir)

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        _validate_columns(df, f)
        dfs.append(df)
        logger.info("  %s — %d rows", os.path.basename(f), len(df))

    combined = pd.concat(dfs, ignore_index=True)
    logger.info("Combined dataset: %d rows × %d cols", len(combined), len(combined.columns))
    return combined


# ── Data inspection ─────────────────────────────────────────────────────────

def inspect_data(df: pd.DataFrame) -> dict:
    """Run mandatory data checks and return a summary dict.

    Checks performed:
    1. Missing values
    2. Non-numeric columns
    3. MODE unique values and counts
    4. LEFT_KNEE value range
    """
    info: dict = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
    }

    # 1. Missing values
    na_count = df[REQUIRED_COLS].isna().sum().sum()
    info["missing_values"] = int(na_count)

    # 2. Non-numeric check on required columns
    non_numeric: List[str] = []
    for col in REQUIRED_COLS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)
    info["non_numeric_columns"] = non_numeric

    # 3. MODE inspection
    mode_col = "MODE"
    mode_counts = df[mode_col].value_counts().sort_index()
    info["mode_unique"] = sorted(df[mode_col].dropna().unique().tolist())
    info["mode_distribution"] = {int(k): int(v) for k, v in mode_counts.items()}

    # 4. LEFT_KNEE range
    knee = df["LEFT_KNEE"]
    info["knee_min"] = float(knee.min())
    info["knee_max"] = float(knee.max())
    info["knee_mean"] = float(knee.mean())

    return info


# ── Validation helpers ──────────────────────────────────────────────────────

def _validate_columns(df: pd.DataFrame, source: str) -> None:
    """Check required columns are present."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{os.path.basename(source)} is missing required columns: {missing}"
        )


def validate_mode_labels(df: pd.DataFrame, valid_labels: list, source: str = "") -> None:
    """Ensure all MODE values are within the expected range."""
    invalid = df["MODE"].dropna()
    invalid = invalid[~invalid.isin(valid_labels)]
    if len(invalid) > 0:
        unique_invalid = sorted(invalid.unique().tolist())
        raise ValueError(
            f"Invalid MODE labels found in {source}: {unique_invalid}"
        )


# ── Data-cleaning helpers ───────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing values in required columns and log the count."""
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLS)
    after = len(df)
    dropped = before - after
    if dropped > 0:
        logger.warning("Dropped %d rows with missing values in required columns.", dropped)
    return df
