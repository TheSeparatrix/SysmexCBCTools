"""Shared fixtures for data module tests."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def minimal_df():
    """A small DataFrame resembling XN_SAMPLE structure."""
    return pd.DataFrame({
        "Sample No.": ["S001", "S002", "S003"],
        "WBC": [5.0, 6.1, 7.2],
        "RBC": [4.5, 4.8, 5.0],
    })


@pytest.fixture
def ancillary_fixture_dir():
    """Path to the ancillary fixture directory."""
    return Path(__file__).resolve().parent / "fixtures" / "ancillary"


@pytest.fixture
def processed_df():
    """Post-pipeline DataFrame for the ancillary fixture samples.

    QC-32941101 has two raw measurements; only the earliest survives.
    """
    return pd.DataFrame({
        "Sample No.": [
            "QC-32941101", "QC-43211103", "QC-41531101", "QC-43211102",
        ],
        "Date": ["2013/12/27", "2015/02/03", "2014/08/22", "2014/12/16"],
        "Time": ["09:52:04", "09:10:38", "09:08:26", "10:02:34"],
        "WBC(10^3/uL)": [2.92, 15.69, 6.45, 8.12],
    })


@pytest.fixture
def csv_file(tmp_path, minimal_df):
    """Write *minimal_df* to a CSV file and return its path."""
    path = tmp_path / "sample.csv"
    minimal_df.to_csv(path, index=False)
    return path


@pytest.fixture
def parquet_file(tmp_path, minimal_df):
    """Write *minimal_df* to a Parquet file and return its path.

    Skips automatically if pyarrow is not installed.
    """
    pytest.importorskip("pyarrow")
    path = tmp_path / "sample.parquet"
    minimal_df.to_parquet(path, index=False)
    return path


@pytest.fixture
def logger():
    """A silent logger (NullHandler only)."""
    log = logging.getLogger("test_data_module")
    log.handlers.clear()
    log.addHandler(logging.NullHandler())
    log.setLevel(logging.DEBUG)
    return log
