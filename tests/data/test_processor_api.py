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


class TestAncillaryCopyValidation:
    """Validation for copy_output_data and copy_sct parameters."""

    def test_copy_output_data_with_dataframe_raises(self, processor, minimal_df):
        with pytest.raises(ValueError, match="not a DataFrame"):
            processor.process_files(
                minimal_df, copy_output_data=True, save_output=True
            )

    def test_copy_sct_with_dataframe_no_archive_raises(self, processor, minimal_df):
        with pytest.raises(ValueError, match="not a DataFrame"):
            processor.process_files(
                minimal_df, copy_sct=True, save_output=True
            )

    def test_copy_output_data_without_save_raises(self, processor, csv_file):
        with pytest.raises(ValueError, match="save_output=True"):
            processor.process_files(
                str(csv_file), copy_output_data=True, save_output=False
            )

    def test_copy_sct_without_save_raises(self, processor, csv_file):
        with pytest.raises(ValueError, match="save_output=True"):
            processor.process_files(
                str(csv_file), copy_sct=True, save_output=False
            )

    def test_sct_archive_with_dataframe_allowed(self, processor, minimal_df):
        """copy_sct + sct_archive_files should NOT raise with DataFrame input."""
        with mock.patch.object(
            processor, "_process_pipeline", return_value=minimal_df
        ), mock.patch(
            "sysmexcbctools.data.sysmexclean.ancillary.build_matching_keys",
            return_value=set(),
        ), mock.patch(
            "sysmexcbctools.data.sysmexclean.ancillary.reconstruct_sct_from_archives",
            return_value=0,
        ) as mock_reconstruct:
            processor.process_files(
                minimal_df,
                save_output=True,
                copy_sct=True,
                sct_archive_files=["archive.csv"],
            )
        mock_reconstruct.assert_called_once()

    def test_sct_archive_without_save_raises(self, processor, minimal_df):
        with pytest.raises(ValueError, match="save_output=True"):
            processor.process_files(
                minimal_df,
                copy_sct=True,
                save_output=False,
                sct_archive_files=["archive.csv"],
            )

    def test_ancillary_flags_default_false(self, processor, minimal_df):
        """By default, no ancillary files should be created."""
        with mock.patch.object(
            processor, "_process_pipeline", return_value=minimal_df
        ):
            processor.process_files(minimal_df, save_output=False)

        assert "output_data" not in processor.diagnostic_files_
        assert "sct_dir" not in processor.diagnostic_files_


class TestSctArchiveDispatch:
    """Verify that sct_archive_files routes to the correct function."""

    def test_archive_dispatches_to_reconstruct(self, processor, minimal_df):
        """With sct_archive_files, should call reconstruct_sct_from_archives."""
        with mock.patch.object(
            processor, "_process_pipeline", return_value=minimal_df
        ), mock.patch(
            "sysmexcbctools.data.sysmexclean.ancillary.build_matching_keys",
            return_value=set(),
        ), mock.patch(
            "sysmexcbctools.data.sysmexclean.ancillary.reconstruct_sct_from_archives",
            return_value=3,
        ) as mock_reconstruct, mock.patch(
            "sysmexcbctools.data.sysmexclean.ancillary.copy_matching_sct_files",
        ) as mock_copy:
            processor.process_files(
                minimal_df,
                save_output=True,
                copy_sct=True,
                sct_archive_files=["a.csv", "b.parquet"],
            )
        mock_reconstruct.assert_called_once()
        mock_copy.assert_not_called()

    def test_no_archive_dispatches_to_copy(self, processor, csv_file, minimal_df):
        """Without sct_archive_files, should call copy_matching_sct_files."""
        with mock.patch.object(
            processor, "_process_pipeline", return_value=minimal_df
        ), mock.patch(
            "sysmexcbctools.data.sysmexclean.ancillary.build_matching_keys",
            return_value=set(),
        ), mock.patch(
            "sysmexcbctools.data.sysmexclean.ancillary.reconstruct_sct_from_archives",
        ) as mock_reconstruct, mock.patch(
            "sysmexcbctools.data.sysmexclean.ancillary.copy_matching_sct_files",
            return_value=2,
        ) as mock_copy:
            processor.process_files(
                str(csv_file),
                save_output=True,
                copy_sct=True,
            )
        mock_copy.assert_called_once()
        mock_reconstruct.assert_not_called()

    def test_string_archive_normalised_to_list(self, processor, minimal_df):
        """A single string sct_archive_files should be normalised to a list."""
        with mock.patch.object(
            processor, "_process_pipeline", return_value=minimal_df
        ), mock.patch(
            "sysmexcbctools.data.sysmexclean.ancillary.build_matching_keys",
            return_value=set(),
        ), mock.patch(
            "sysmexcbctools.data.sysmexclean.ancillary.reconstruct_sct_from_archives",
            return_value=0,
        ) as mock_reconstruct:
            processor.process_files(
                minimal_df,
                save_output=True,
                copy_sct=True,
                sct_archive_files="single_archive.csv",
            )
        # First positional arg should be a list
        archive_arg = mock_reconstruct.call_args[0][0]
        assert archive_arg == ["single_archive.csv"]


class TestOutputFormat:
    """Verify that output_format controls the saved file extension."""

    def test_csv_default(self, processor, minimal_df, tmp_path):
        with mock.patch.object(
            processor, "_process_pipeline", return_value=minimal_df
        ):
            processor.process_files(
                minimal_df, save_output=True, dataset_name="d",
            )
        saved = list(tmp_path.glob("*.csv"))
        assert len(saved) == 1
        assert saved[0].suffix == ".csv"

    def test_parquet_output(self, processor, minimal_df, tmp_path):
        pytest.importorskip("pyarrow")
        with mock.patch.object(
            processor, "_process_pipeline", return_value=minimal_df
        ):
            processor.process_files(
                minimal_df,
                save_output=True,
                dataset_name="d",
                output_format="parquet",
            )
        saved = list(tmp_path.glob("*.parquet"))
        assert len(saved) == 1
        # Round-trip check
        restored = pd.read_parquet(saved[0])
        assert len(restored) == len(minimal_df)

    def test_invalid_format_raises(self, processor, minimal_df):
        with mock.patch.object(
            processor, "_process_pipeline", return_value=minimal_df
        ):
            with pytest.raises(ValueError, match="Unsupported output_format"):
                processor.process_files(
                    minimal_df,
                    save_output=True,
                    dataset_name="d",
                    output_format="xml",
                )
