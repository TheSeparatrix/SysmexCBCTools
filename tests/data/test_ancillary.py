"""Tests for ancillary file matching and copying functions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sysmexcbctools.data.sysmexclean.ancillary import (
    _ensure_parquet,
    _find_overflow_files,
    _normalize_overflow_filename,
    _parse_sct_decoded_filename,
    _parse_sct_filename,
    _read_and_merge_sct,
    build_matching_keys,
    copy_matching_sct_files,
    derive_source_dirs,
    filter_output_data,
    reconstruct_sct_from_archives,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "ancillary"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ancillary_fixture_dir():
    """Path to the ancillary fixture directory."""
    return FIXTURE_DIR


@pytest.fixture
def processed_df():
    """DataFrame resembling post-pipeline output for the fixture data.

    Contains only samples that would survive processing.  QC-32941101
    appears twice in the raw data (2013/12/27 and 2014/01/10); the
    pipeline keeps only the earliest measurement.
    """
    return pd.DataFrame({
        "Sample No.": ["QC-32941101", "QC-43211103", "QC-41531101", "QC-43211102"],
        "Date": ["2013/12/27", "2015/02/03", "2014/08/22", "2014/12/16"],
        "Time": ["09:52:04", "09:10:38", "09:08:26", "10:02:34"],
        "WBC(10^3/uL)": [2.92, 15.69, 6.45, 8.12],
    })


# ---------------------------------------------------------------------------
# build_matching_keys
# ---------------------------------------------------------------------------

class TestBuildMatchingKeys:

    def test_correct_key_set(self, processed_df):
        keys = build_matching_keys(processed_df)
        assert keys == {
            ("QC-32941101", "20131227_095204"),
            ("QC-43211103", "20150203_091038"),
            ("QC-41531101", "20140822_090826"),
            ("QC-43211102", "20141216_100234"),
        }

    def test_handles_both_date_formats(self):
        """Verify that both zero-padded and non-padded dates work."""
        df = pd.DataFrame({
            "Sample No.": ["S001", "S002"],
            "Date": ["2013/04/16", "2013/4/6"],
            "Time": ["09:05:03", "9:5:3"],
        })
        keys = build_matching_keys(df)
        assert keys == {
            ("S001", "20130416_090503"),
            ("S002", "20130406_090503"),
        }

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["Sample No.", "Date", "Time"])
        keys = build_matching_keys(df)
        assert keys == set()


# ---------------------------------------------------------------------------
# derive_source_dirs
# ---------------------------------------------------------------------------

class TestDeriveSourceDirs:

    def test_single_path(self, tmp_path):
        dirs = derive_source_dirs([str(tmp_path / "XN_SAMPLE.csv")])
        assert dirs == [tmp_path]

    def test_deduplication(self, tmp_path):
        dirs = derive_source_dirs([
            str(tmp_path / "a.csv"),
            str(tmp_path / "b.csv"),
        ])
        assert len(dirs) == 1
        assert dirs[0] == tmp_path

    def test_multiple_dirs(self, tmp_path):
        d1 = tmp_path / "machine1"
        d2 = tmp_path / "machine2"
        d1.mkdir()
        d2.mkdir()
        dirs = derive_source_dirs([
            str(d1 / "XN_SAMPLE.csv"),
            str(d2 / "XN_SAMPLE.csv"),
        ])
        assert len(dirs) == 2


# ---------------------------------------------------------------------------
# _parse_sct_filename
# ---------------------------------------------------------------------------

class TestParseSctFilename:

    def test_standard_filename(self):
        fname = "WDF_[XN-10^11036][00-15_5][20150203_091038][           QC-43211103].116.csv"
        result = _parse_sct_filename(fname)
        assert result == ("QC-43211103", "20150203_091038")

    def test_different_channel(self):
        fname = "PLTF_[XN-10^11041][00-15_5][20140822_090826][           QC-41531101].116.csv"
        result = _parse_sct_filename(fname)
        assert result == ("QC-41531101", "20140822_090826")

    def test_overflow_file_returns_none(self):
        fname = "RET_[XN-10^11036][00-15_5][20150203_091038][           QC-43211103].116(1).csv"
        assert _parse_sct_filename(fname) is None

    def test_malformed_filename_returns_none(self):
        assert _parse_sct_filename("not_a_valid_file.csv") is None
        assert _parse_sct_filename("README.md") is None


# ---------------------------------------------------------------------------
# _find_overflow_files
# ---------------------------------------------------------------------------

class TestFindOverflowFiles:

    def test_finds_overflow_siblings(self, ancillary_fixture_dir):
        base = (
            ancillary_fixture_dir / "SCT"
            / "WDF_[XN-10^11036][00-15_5][20150203_091038][           QC-43211103].116.csv"
        )
        overflows = _find_overflow_files(base)
        assert len(overflows) == 2
        assert "116(1).csv" in str(overflows[0])
        assert "116(2).csv" in str(overflows[1])

    def test_single_overflow(self, ancillary_fixture_dir):
        base = (
            ancillary_fixture_dir / "SCT"
            / "RET_[XN-10^11036][00-15_5][20150203_091038][           QC-43211103].116.csv"
        )
        overflows = _find_overflow_files(base)
        assert len(overflows) == 1

    def test_no_overflow(self, ancillary_fixture_dir):
        base = (
            ancillary_fixture_dir / "SCT"
            / "WNR_[XN-10^11036][00-15_5][20131227_095204][           QC-32941101].116.csv"
        )
        overflows = _find_overflow_files(base)
        assert overflows == []


# ---------------------------------------------------------------------------
# _read_and_merge_sct
# ---------------------------------------------------------------------------

class TestReadAndMergeSct:

    def test_base_only(self, ancillary_fixture_dir):
        base = (
            ancillary_fixture_dir / "SCT"
            / "WNR_[XN-10^11036][00-15_5][20131227_095204][           QC-32941101].116.csv"
        )
        df = _read_and_merge_sct(base, [])
        assert len(df) == 5  # 5 base rows

    def test_with_overflow(self, ancillary_fixture_dir):
        base = (
            ancillary_fixture_dir / "SCT"
            / "WDF_[XN-10^11036][00-15_5][20150203_091038][           QC-43211103].116.csv"
        )
        overflows = _find_overflow_files(base)
        df = _read_and_merge_sct(base, overflows)
        # 5 base + 3 overflow(1) + 2 overflow(2) = 10
        assert len(df) == 10


# ---------------------------------------------------------------------------
# filter_output_data
# ---------------------------------------------------------------------------

class TestFilterOutputData:

    def test_correct_row_filtering(self, ancillary_fixture_dir, processed_df, tmp_path, logger):
        keys = build_matching_keys(processed_df)
        out = tmp_path / "OutputData_filtered.csv"
        n = filter_output_data([ancillary_fixture_dir], keys, out, logger)
        assert n == 4
        result = pd.read_csv(out)
        assert set(result["Sample No."]) == {
            "QC-32941101", "QC-43211103", "QC-41531101", "QC-43211102",
        }

    def test_excludes_non_matching(self, ancillary_fixture_dir, tmp_path, logger):
        """Only the 20131227 measurement of QC-32941101 should match."""
        keys = {("QC-32941101", "20131227_095204")}
        out = tmp_path / "od.csv"
        n = filter_output_data([ancillary_fixture_dir], keys, out, logger)
        assert n == 1

    def test_missing_file_warns(self, tmp_path, logger):
        keys = {("S001", "20200101_120000")}
        out = tmp_path / "od.csv"
        n = filter_output_data([tmp_path], keys, out, logger)
        assert n == 0
        # No output file created when source dirs have no OutputData.csv
        # and no explicit columns were requested.
        assert not out.exists()

    def test_empty_keys(self, ancillary_fixture_dir, tmp_path, logger):
        out = tmp_path / "od.csv"
        n = filter_output_data([ancillary_fixture_dir], set(), out, logger)
        assert n == 0

    def test_deduplicates_across_source_dirs(
        self, ancillary_fixture_dir, tmp_path, logger,
    ):
        """Same OutputData.csv listed twice should not produce duplicate rows."""
        keys = {("QC-32941101", "20131227_095204")}
        out = tmp_path / "od.csv"
        n = filter_output_data(
            [ancillary_fixture_dir, ancillary_fixture_dir], keys, out, logger,
        )
        assert n == 1
        result = pd.read_csv(out)
        assert len(result) == 1

    def test_column_filtering(self, ancillary_fixture_dir, processed_df, tmp_path, logger):
        """Only requested columns should appear in the output."""
        keys = build_matching_keys(processed_df)
        out = tmp_path / "od.csv"
        cols = ["Sample No.", "AnalyzeDate", "AnalyzeTime", "RBC"]
        n = filter_output_data(
            [ancillary_fixture_dir], keys, out, logger, columns=cols,
        )
        assert n == 4
        result = pd.read_csv(out)
        assert list(result.columns) == cols


# ---------------------------------------------------------------------------
# copy_matching_sct_files
# ---------------------------------------------------------------------------

class TestCopyMatchingSctFiles:

    def test_basic_copy(self, ancillary_fixture_dir, tmp_path, logger):
        """Matching files for one sample should be written."""
        keys = {("QC-41531101", "20140822_090826")}
        out = tmp_path / "SCT"
        out.mkdir()
        n = copy_matching_sct_files([ancillary_fixture_dir], keys, str(out), logger)
        # 4 channels (PLTF, RET, WDF, WNR), all base only
        assert n == 4
        assert len(list(out.iterdir())) == 4

    def test_with_overflow_consolidation(self, ancillary_fixture_dir, tmp_path, logger):
        """Overflow files should be concatenated into the base output."""
        keys = {("QC-43211103", "20150203_091038")}
        out = tmp_path / "SCT"
        out.mkdir()
        n = copy_matching_sct_files([ancillary_fixture_dir], keys, str(out), logger)
        # 4 channels
        assert n == 4

        # Check that the WDF output has consolidated rows (5 + 3 + 2 = 10)
        wdf_files = [f for f in out.iterdir() if f.name.startswith("WDF_")]
        assert len(wdf_files) == 1
        wdf_df = pd.read_csv(wdf_files[0])
        assert len(wdf_df) == 10

        # No overflow files in the output
        overflow_files = [f for f in out.iterdir() if "(" in f.name]
        assert overflow_files == []

    def test_non_matching_skipped(self, ancillary_fixture_dir, tmp_path, logger):
        keys = {("NONEXISTENT", "20200101_120000")}
        out = tmp_path / "SCT"
        out.mkdir()
        n = copy_matching_sct_files([ancillary_fixture_dir], keys, str(out), logger)
        assert n == 0

    def test_missing_sct_dir(self, tmp_path, logger):
        keys = {("S001", "20200101_120000")}
        out = tmp_path / "SCT"
        out.mkdir()
        n = copy_matching_sct_files([tmp_path], keys, str(out), logger)
        assert n == 0

    def test_deduplicates_across_source_dirs(
        self, ancillary_fixture_dir, tmp_path, logger,
    ):
        """Same SCT dir listed twice should not re-copy files."""
        keys = {("QC-41531101", "20140822_090826")}
        out = tmp_path / "SCT"
        out.mkdir()
        n = copy_matching_sct_files(
            [ancillary_fixture_dir, ancillary_fixture_dir],
            keys, str(out), logger,
        )
        # 4 channels, written once (not 8)
        assert n == 4


# ---------------------------------------------------------------------------
# _normalize_overflow_filename
# ---------------------------------------------------------------------------

class TestNormalizeOverflowFilename:

    def test_base_unchanged(self):
        fname = "WDF_[XN-10^11036][00-15_5][20150203_091038][           QC-43211103].116.csv"
        assert _normalize_overflow_filename(fname) == fname

    def test_overflow_stripped(self):
        fname = "WDF_[XN-10^11036][00-15_5][20150203_091038][           QC-43211103].116(1).csv"
        expected = "WDF_[XN-10^11036][00-15_5][20150203_091038][           QC-43211103].116.csv"
        assert _normalize_overflow_filename(fname) == expected

    def test_higher_overflow_index(self):
        fname = "RET_[XN-10^11036][00-15_5][20150203_091038][           QC-43211103].116(42).csv"
        expected = "RET_[XN-10^11036][00-15_5][20150203_091038][           QC-43211103].116.csv"
        assert _normalize_overflow_filename(fname) == expected

    def test_non_sct_unchanged(self):
        assert _normalize_overflow_filename("OutputData.csv") == "OutputData.csv"
        assert _normalize_overflow_filename("README.md") == "README.md"


# ---------------------------------------------------------------------------
# _parse_sct_decoded_filename
# ---------------------------------------------------------------------------

class TestParseSctDecodedFilename:

    def test_base_filename(self):
        fname = "WDF_[XN-10^11036][00-15_5][20150203_091038][           QC-43211103].116.csv"
        assert _parse_sct_decoded_filename(fname) == ("QC-43211103", "20150203_091038")

    def test_overflow_filename(self):
        fname = "WDF_[XN-10^11036][00-15_5][20150203_091038][           QC-43211103].116(1).csv"
        assert _parse_sct_decoded_filename(fname) == ("QC-43211103", "20150203_091038")

    def test_malformed_returns_none(self):
        assert _parse_sct_decoded_filename("garbage.csv") is None


# ---------------------------------------------------------------------------
# _ensure_parquet
# ---------------------------------------------------------------------------

class TestEnsureParquet:

    @pytest.fixture(autouse=True)
    def _require_duckdb(self):
        pytest.importorskip("duckdb")

    def test_csv_converts_to_parquet(self, tmp_path, logger):
        """A CSV archive produces a sibling .parquet file."""
        df = pd.DataFrame({"a": range(10), "b": range(10, 20)})
        csv_path = tmp_path / "archive.csv"
        df.to_csv(csv_path, index=False)

        result = _ensure_parquet(csv_path, logger)
        assert result == tmp_path / "archive.parquet"
        assert result.exists()
        restored = pd.read_parquet(result)
        # all_varchar=true means every column is stored as string
        pd.testing.assert_frame_equal(
            restored, df.astype(str), check_dtype=False,
        )

    def test_existing_parquet_not_reconverted(self, tmp_path, logger):
        """If .parquet already exists alongside CSV, return it without re-creating."""
        csv_path = tmp_path / "archive.csv"
        csv_path.write_text("a,b\n1,2\n")
        pq_path = tmp_path / "archive.parquet"
        # Write a distinct Parquet so we can verify it was NOT overwritten
        pd.DataFrame({"x": [99]}).to_parquet(pq_path, index=False)

        result = _ensure_parquet(csv_path, logger)
        assert result == pq_path
        restored = pd.read_parquet(result)
        assert list(restored.columns) == ["x"]  # original, not from CSV

    def test_parquet_input_returned_as_is(self, tmp_path, logger):
        """A .parquet archive returns its own path unchanged."""
        pq_path = tmp_path / "archive.parquet"
        pd.DataFrame({"a": [1]}).to_parquet(pq_path, index=False)

        assert _ensure_parquet(pq_path, logger) == pq_path

    def test_pq_extension_accepted(self, tmp_path, logger):
        """A .pq archive returns its own path unchanged."""
        pq_path = tmp_path / "archive.pq"
        pd.DataFrame({"a": [1]}).to_parquet(pq_path, index=False)

        assert _ensure_parquet(pq_path, logger) == pq_path

    def test_bad_extension_raises(self, tmp_path, logger):
        """Unsupported extension raises ValueError."""
        bad_path = tmp_path / "archive.xlsx"
        bad_path.touch()
        with pytest.raises(ValueError, match="Unsupported archive extension"):
            _ensure_parquet(bad_path, logger)


# ---------------------------------------------------------------------------
# reconstruct_sct_from_archives
# ---------------------------------------------------------------------------

def _make_archive_df(
    decoded_filename, channel, sample_no, date_time, data_cols,
    *, analyzer="XN-10^36677", unknown="00-22-123",
):
    """Helper to build a small archive-style DataFrame."""
    df = pd.DataFrame(data_cols)
    df["decoded_filename"] = decoded_filename
    df["channel"] = channel
    df["analyzer"] = analyzer
    df["unknown"] = unknown
    df["sample_no"] = sample_no
    df["date_time"] = date_time
    return df


class TestReconstructSctFromArchives:

    @pytest.fixture(autouse=True)
    def _require_duckdb(self):
        pytest.importorskip("duckdb")

    @pytest.fixture
    def wdf_base_filename(self):
        return "WDF_[XN-10^11036][00-15_5][20150203_091038][           QC-43211103].116.csv"

    @pytest.fixture
    def pltf_base_filename(self):
        return "PLTF_[XN-10^11041][00-15_5][20140822_090826][           QC-41531101].116.csv"

    @pytest.fixture
    def matching_keys(self):
        return {
            ("QC-43211103", "20150203_091038"),
            ("QC-41531101", "20140822_090826"),
        }

    def _write_archive(self, path, frames):
        combined = pd.concat(frames, ignore_index=True)
        ext = path.suffix.lower()
        if ext == ".csv":
            combined.to_csv(path, index=False)
        else:
            combined.to_parquet(path, index=False)
        return combined

    def test_basic_reconstruction(self, tmp_path, logger, wdf_base_filename, matching_keys):
        """Rows are reconstructed with correct content and columns."""
        archive_df = _make_archive_df(
            decoded_filename=wdf_base_filename,
            channel="WDF",
            sample_no="QC-43211103",
            date_time="20150203_091038",
            data_cols={
                "repeatcount": [1, 1, 1],
                "particleid": [10, 20, 30],
                "fsc": [100.0, 200.0, 300.0],
                "ssc": [50.0, 60.0, 70.0],
                "sfl": [1.1, 2.2, 3.3],
                "fscw": [0.5, 0.6, 0.7],
            },
        )
        archive_path = tmp_path / "archive.csv"
        self._write_archive(archive_path, [archive_df])

        out_dir = tmp_path / "SCT"
        out_dir.mkdir()
        n = reconstruct_sct_from_archives(
            [str(archive_path)], matching_keys, str(out_dir), logger,
        )
        assert n == 1
        result = pd.read_csv(out_dir / wdf_base_filename)
        assert len(result) == 3
        assert list(result.columns) == [
            "RepeatCount", "ParticleID", "FSC", "SSC", "SFL", "FSCW",
        ]

    def test_overflow_consolidation(self, tmp_path, logger, wdf_base_filename, matching_keys):
        """Overflow files (.116(N).csv) are merged into one base file."""
        overflow_fn = wdf_base_filename.replace(".116.csv", ".116(1).csv")
        base_df = _make_archive_df(
            decoded_filename=wdf_base_filename,
            channel="WDF",
            sample_no="QC-43211103",
            date_time="20150203_091038",
            data_cols={"repeatcount": [1, 1], "fsc": [100.0, 200.0]},
        )
        overflow_df = _make_archive_df(
            decoded_filename=overflow_fn,
            channel="WDF",
            sample_no="QC-43211103",
            date_time="20150203_091038",
            data_cols={"repeatcount": [1], "fsc": [300.0]},
        )
        archive_path = tmp_path / "archive.csv"
        self._write_archive(archive_path, [base_df, overflow_df])

        out_dir = tmp_path / "SCT"
        out_dir.mkdir()
        n = reconstruct_sct_from_archives(
            [str(archive_path)], matching_keys, str(out_dir), logger,
        )
        assert n == 1
        result = pd.read_csv(out_dir / wdf_base_filename)
        # Base (2) + overflow (1) = 3 rows
        assert len(result) == 3

    def test_metadata_columns_stripped(self, tmp_path, logger, wdf_base_filename, matching_keys):
        """Archive metadata columns should not appear in the output."""
        archive_df = _make_archive_df(
            decoded_filename=wdf_base_filename,
            channel="WDF",
            sample_no="QC-43211103",
            date_time="20150203_091038",
            data_cols={"repeatcount": [1], "fsc": [100.0]},
        )
        archive_path = tmp_path / "archive.csv"
        self._write_archive(archive_path, [archive_df])

        out_dir = tmp_path / "SCT"
        out_dir.mkdir()
        reconstruct_sct_from_archives(
            [str(archive_path)], matching_keys, str(out_dir), logger,
        )
        result = pd.read_csv(out_dir / wdf_base_filename)
        for col in (
            "decoded_filename", "channel", "analyzer", "unknown",
            "date_time", "sample_no",
        ):
            assert col not in result.columns

    def test_all_nan_columns_dropped(self, tmp_path, logger, wdf_base_filename, matching_keys):
        """Columns that are all-NaN should be dropped (channel-specific restoration)."""
        import numpy as np
        archive_df = _make_archive_df(
            decoded_filename=wdf_base_filename,
            channel="WDF",
            sample_no="QC-43211103",
            date_time="20150203_091038",
            data_cols={
                "repeatcount": [1, 2],
                "fsc": [100.0, 200.0],
                "phase": [np.nan, np.nan],  # WDF doesn't have Phase
                "fsclog": [np.nan, np.nan],  # WDF doesn't have FSClog
            },
        )
        archive_path = tmp_path / "archive.csv"
        self._write_archive(archive_path, [archive_df])

        out_dir = tmp_path / "SCT"
        out_dir.mkdir()
        reconstruct_sct_from_archives(
            [str(archive_path)], matching_keys, str(out_dir), logger,
        )
        result = pd.read_csv(out_dir / wdf_base_filename)
        assert "Phase" not in result.columns
        assert "FSClog" not in result.columns
        assert "RepeatCount" in result.columns
        assert "FSC" in result.columns

    def test_column_case_restored(self, tmp_path, logger, wdf_base_filename, matching_keys):
        """Lowercase archive columns should be renamed to original casing."""
        archive_df = _make_archive_df(
            decoded_filename=wdf_base_filename,
            channel="WDF",
            sample_no="QC-43211103",
            date_time="20150203_091038",
            data_cols={"repeatcount": [1], "particleid": [10], "sflx2": [1.5]},
        )
        archive_path = tmp_path / "archive.csv"
        self._write_archive(archive_path, [archive_df])

        out_dir = tmp_path / "SCT"
        out_dir.mkdir()
        reconstruct_sct_from_archives(
            [str(archive_path)], matching_keys, str(out_dir), logger,
        )
        result = pd.read_csv(out_dir / wdf_base_filename)
        assert "RepeatCount" in result.columns
        assert "ParticleID" in result.columns
        assert "SFLx2" in result.columns

    def test_non_matching_excluded(self, tmp_path, logger, wdf_base_filename):
        """Rows for non-matching samples should not produce output files."""
        archive_df = _make_archive_df(
            decoded_filename=wdf_base_filename,
            channel="WDF",
            sample_no="QC-43211103",
            date_time="20150203_091038",
            data_cols={"repeatcount": [1], "fsc": [100.0]},
        )
        archive_path = tmp_path / "archive.csv"
        self._write_archive(archive_path, [archive_df])

        out_dir = tmp_path / "SCT"
        out_dir.mkdir()
        # Keys that don't match anything in the archive
        n = reconstruct_sct_from_archives(
            [str(archive_path)],
            {("NONEXISTENT", "20200101_120000")},
            str(out_dir),
            logger,
        )
        assert n == 0
        assert list(out_dir.iterdir()) == []

    def test_deduplication_across_archives(
        self, tmp_path, logger, wdf_base_filename, matching_keys,
    ):
        """The same file appearing in two archives should only be written once."""
        archive_df = _make_archive_df(
            decoded_filename=wdf_base_filename,
            channel="WDF",
            sample_no="QC-43211103",
            date_time="20150203_091038",
            data_cols={"repeatcount": [1], "fsc": [100.0]},
        )
        a1 = tmp_path / "archive1.csv"
        a2 = tmp_path / "archive2.csv"
        self._write_archive(a1, [archive_df])
        self._write_archive(a2, [archive_df])

        out_dir = tmp_path / "SCT"
        out_dir.mkdir()
        n = reconstruct_sct_from_archives(
            [str(a1), str(a2)], matching_keys, str(out_dir), logger,
        )
        assert n == 1

    def test_empty_archive_returns_zero(self, tmp_path, logger, matching_keys):
        """An archive with no matching rows should return 0."""
        df = pd.DataFrame({
            "decoded_filename": [],
            "channel": [],
            "sample_no": [],
            "date_time": [],
            "repeatcount": [],
            "fsc": [],
        })
        archive_path = tmp_path / "empty.csv"
        df.to_csv(archive_path, index=False)

        out_dir = tmp_path / "SCT"
        out_dir.mkdir()
        n = reconstruct_sct_from_archives(
            [str(archive_path)], matching_keys, str(out_dir), logger,
        )
        assert n == 0

    def test_missing_decoded_filename_warns(self, tmp_path, logger, matching_keys):
        """An archive without 'decoded_filename' should warn and skip."""
        df = pd.DataFrame({"repeatcount": [1], "fsc": [100.0]})
        archive_path = tmp_path / "bad.parquet"
        df.to_parquet(archive_path, index=False)

        out_dir = tmp_path / "SCT"
        out_dir.mkdir()
        n = reconstruct_sct_from_archives(
            [str(archive_path)], matching_keys, str(out_dir), logger,
        )
        assert n == 0
