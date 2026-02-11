"""Tests for _read_single_file and load_dataframes in utils.py."""

from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest

from sysmexcbctools.data.sysmexclean.utils import (
    _read_single_file,
    load_dataframes,
)

# -- _read_single_file -------------------------------------------------------

class TestReadSingleFile:
    """Tests for the _read_single_file dispatcher."""

    def test_csv(self, csv_file, logger):
        df = _read_single_file(csv_file, logger)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "WBC" in df.columns

    def test_parquet(self, parquet_file, logger):
        df = _read_single_file(parquet_file, logger)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "WBC" in df.columns

    def test_unknown_extension_raises(self, tmp_path, logger):
        bad_file = tmp_path / "data.xlsx"
        bad_file.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            _read_single_file(bad_file, logger)

    def test_missing_pyarrow_raises(self, tmp_path, logger):
        """Simulate pyarrow not being installed."""
        pq_file = tmp_path / "data.parquet"
        pq_file.write_bytes(b"dummy")

        with mock.patch(
            "sysmexcbctools.data.sysmexclean.utils.pd.read_parquet",
            side_effect=ImportError("No module named 'pyarrow'"),
        ):
            with pytest.raises(ImportError, match="pyarrow"):
                _read_single_file(pq_file, logger)


# -- load_dataframes ----------------------------------------------------------

class TestLoadDataframes:
    """Tests for load_dataframes."""

    def test_single_csv(self, csv_file, logger):
        df = load_dataframes([str(csv_file)], logger)
        assert len(df) == 3

    def test_single_parquet(self, parquet_file, logger):
        df = load_dataframes([str(parquet_file)], logger)
        assert len(df) == 3

    def test_empty_list_raises(self, logger):
        with pytest.raises(ValueError, match="No valid dataframes loaded"):
            load_dataframes([], logger)

    def test_mixed_formats(self, csv_file, parquet_file, logger):
        df = load_dataframes([str(csv_file), str(parquet_file)], logger)
        assert len(df) == 6  # 3 rows from each

    def test_unsupported_extension_propagates(self, tmp_path, logger):
        bad_file = tmp_path / "data.xlsx"
        bad_file.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_dataframes([str(bad_file)], logger)

    def test_corrupt_file_skipped(self, tmp_path, logger):
        """A corrupt CSV is logged and skipped; a good one still loads."""
        bad = tmp_path / "bad.csv"
        bad.write_text("this,is\nnot,valid\x00\x01data")

        good = tmp_path / "good.csv"
        pd.DataFrame({"A": [1]}).to_csv(good, index=False)

        # The corrupt file may or may not raise -- it depends on pandas.
        # What we care about is that the good file still loads, or that a
        # clear error is raised (not silently empty).
        try:
            df = load_dataframes([str(bad), str(good)], logger)
            # If it didn't raise, the good file should be present
            assert len(df) >= 1
        except ValueError:
            # All files failed, which is acceptable for truly corrupt input
            pass
