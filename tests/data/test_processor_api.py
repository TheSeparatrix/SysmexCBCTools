"""Tests for XNSampleProcessor.process_files dispatch logic."""

from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest

from sysmexcbctools.data.sysmexclean.processor_api import XNSampleProcessor


@pytest.fixture
def processor(tmp_path):
    """A processor configured for testing (silent, no file logging)."""
    return XNSampleProcessor(
        output_dir=str(tmp_path),
        verbose=0,
        log_to_file=False,
    )


class TestProcessFilesDispatch:
    """Verify that process_files correctly dispatches on input type.

    We mock ``_process_pipeline`` so these tests exercise only the
    loading/dispatch logic, not the full cleaning pipeline.
    """

    def test_dataframe_input(self, processor, minimal_df):
        """A DataFrame should bypass file loading entirely."""
        with mock.patch.object(
            processor, "_process_pipeline", return_value=minimal_df
        ) as mock_pipeline:
            result = processor.process_files(minimal_df)

        mock_pipeline.assert_called_once()
        # The first positional arg to _process_pipeline should be our df
        passed_df = mock_pipeline.call_args[0][0]
        pd.testing.assert_frame_equal(passed_df, minimal_df)
        pd.testing.assert_frame_equal(result, minimal_df)

    def test_single_csv_string(self, processor, csv_file):
        """A single string path should be wrapped in a list and loaded."""
        sentinel = pd.DataFrame({"out": [1]})
        with mock.patch.object(
            processor, "_process_pipeline", return_value=sentinel
        ):
            result = processor.process_files(str(csv_file))

        pd.testing.assert_frame_equal(result, sentinel)

    def test_csv_list(self, processor, csv_file):
        """A list of paths should be concatenated and loaded."""
        sentinel = pd.DataFrame({"out": [1]})
        with mock.patch.object(
            processor, "_process_pipeline", return_value=sentinel
        ):
            result = processor.process_files([str(csv_file)])

        pd.testing.assert_frame_equal(result, sentinel)

    def test_parquet_input(self, processor, parquet_file):
        """A parquet path should be loaded via _read_single_file."""
        sentinel = pd.DataFrame({"out": [1]})
        with mock.patch.object(
            processor, "_process_pipeline", return_value=sentinel
        ):
            result = processor.process_files(str(parquet_file))

        pd.testing.assert_frame_equal(result, sentinel)

    def test_last_processed_stored(self, processor, minimal_df):
        """After processing, last_processed_ should hold the result."""
        with mock.patch.object(
            processor, "_process_pipeline", return_value=minimal_df
        ):
            processor.process_files(minimal_df)

        pd.testing.assert_frame_equal(processor.last_processed_, minimal_df)
