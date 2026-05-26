"""Tests for data_utils module."""

import os
import tempfile

import pandas as pd
import pytest

from src.data_utils import (
    load_single_csv, load_csv_dir, inspect_data,
    clean_data, validate_mode_labels,
)


@pytest.fixture
def sample_csv_path():
    """Create a minimal valid CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(
            "LEFT_TA,LEFT_MG,LEFT_SOL,LEFT_BF,LEFT_ST,LEFT_VL,LEFT_RF,LEFT_KNEE,MODE\n"
            "0.1,0.2,0.3,0.4,0.5,0.6,0.7,45.0,0\n"
            "0.2,0.3,0.4,0.5,0.6,0.7,0.8,46.0,1\n"
            "0.3,0.4,0.5,0.6,0.7,0.8,0.9,47.0,0\n"
            "0.4,0.5,0.6,0.7,0.8,0.9,1.0,48.0,2\n"
            "0.5,0.6,0.7,0.8,0.9,1.0,1.1,49.0,2\n"
        )
    yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_csv_dir():
    """Create a temp directory with multiple CSV files."""
    tmpdir = tempfile.mkdtemp()
    for i, name in enumerate(["file_a.csv", "file_b.csv"]):
        path = os.path.join(tmpdir, name)
        with open(path, "w") as f:
            f.write(
                "LEFT_TA,LEFT_MG,LEFT_SOL,LEFT_BF,LEFT_ST,LEFT_VL,LEFT_RF,LEFT_KNEE,MODE\n"
                "0.1,0.2,0.3,0.4,0.5,0.6,0.7,45.0,0\n"
                "0.2,0.3,0.4,0.5,0.6,0.7,0.8,46.0,1\n"
            )
    yield tmpdir
    for f in os.listdir(tmpdir):
        os.unlink(os.path.join(tmpdir, f))
    os.rmdir(tmpdir)


class TestLoadSingleCSV:
    def test_reads_correctly(self, sample_csv_path):
        df = load_single_csv(sample_csv_path)
        assert len(df) == 5
        assert list(df.columns) == [
            "LEFT_TA", "LEFT_MG", "LEFT_SOL", "LEFT_BF",
            "LEFT_ST", "LEFT_VL", "LEFT_RF", "LEFT_KNEE", "MODE",
        ]

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_single_csv("/nonexistent/path.csv")

    def test_raises_on_missing_columns(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("LEFT_TA,LEFT_MG\n0.1,0.2\n")
        path = f.name
        try:
            with pytest.raises(ValueError):
                load_single_csv(path)
        finally:
            os.unlink(path)


class TestLoadCSVDir:
    def test_merges_multiple_files(self, sample_csv_dir):
        df = load_csv_dir(sample_csv_dir)
        assert len(df) == 4  # 2 files x 2 rows each


class TestInspectData:
    def test_returns_expected_keys(self, sample_csv_path):
        df = pd.read_csv(sample_csv_path)
        info = inspect_data(df)
        assert "num_rows" in info
        assert "mode_distribution" in info
        assert "knee_min" in info
        assert info["num_rows"] == 5

    def test_missing_value_detection(self, sample_csv_path):
        df = pd.read_csv(sample_csv_path)
        df.loc[0, "LEFT_KNEE"] = float("nan")
        info = inspect_data(df)
        assert info["missing_values"] == 1


class TestCleanData:
    def test_drops_na_rows(self):
        df = pd.DataFrame({
            "LEFT_TA": [1.0, 2.0, 3.0],
            "LEFT_MG": [1.0, 2.0, 3.0],
            "LEFT_SOL": [1.0, 2.0, 3.0],
            "LEFT_BF": [1.0, 2.0, 3.0],
            "LEFT_ST": [1.0, 2.0, 3.0],
            "LEFT_VL": [1.0, 2.0, 3.0],
            "LEFT_RF": [1.0, 2.0, 3.0],
            "LEFT_KNEE": [45.0, None, 47.0],
            "MODE": [0, 1, 2],
        })
        cleaned = clean_data(df)
        assert len(cleaned) == 2


class TestValidateModeLabels:
    def test_passes_for_valid_labels(self):
        df = pd.DataFrame({"MODE": [0, 1, 2, 3, 4, 5, 6, 7]})
        validate_mode_labels(df, list(range(8)))

    def test_raises_for_invalid_labels(self):
        df = pd.DataFrame({"MODE": [0, 1, 8]})
        with pytest.raises(ValueError):
            validate_mode_labels(df, list(range(8)))
