"""Tests for ancillary file matching and copying functions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sysmexcbctools.data.sysmexclean.ancillary import (
    _find_overflow_files,
    _parse_sct_filename,
    _read_and_merge_sct,
    build_matching_keys,
    copy_matching_sct_files,
    derive_source_dirs,
    filter_output_data,
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

    def test_correct_row_filtering(self, ancillary_fixture_dir, processed_df, logger):
        keys = build_matching_keys(processed_df)
        source_dirs = [ancillary_fixture_dir]
        result = filter_output_data(source_dirs, keys, logger)
        assert len(result) == 4
        assert set(result["Sample No."]) == {
            "QC-32941101", "QC-43211103", "QC-41531101", "QC-43211102",
        }

    def test_excludes_non_matching(self, ancillary_fixture_dir, logger):
        """Only the 20131227 measurement of QC-32941101 should match."""
        keys = {("QC-32941101", "20131227_095204")}
        result = filter_output_data([ancillary_fixture_dir], keys, logger)
        assert len(result) == 1

    def test_missing_file_warns(self, tmp_path, logger):
        keys = {("S001", "20200101_120000")}
        result = filter_output_data([tmp_path], keys, logger)
        assert result.empty

    def test_empty_keys(self, ancillary_fixture_dir, logger):
        result = filter_output_data([ancillary_fixture_dir], set(), logger)
        assert len(result) == 0


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
