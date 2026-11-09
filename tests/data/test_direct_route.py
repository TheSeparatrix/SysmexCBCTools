"""The direct route: ``process_files(bc_sysmex_dataset_dir=...)``.

Every dataset here is made by bloodcounts-sysmex-format itself from the vendored
QC fixture.  Its two measurements are of QC material, which cleaning drops as
technical samples, so most datasets give them made-up sample numbers instead
(``S-0001`` ...), in a temporary copy, to keep them through cleaning.
"""

from __future__ import annotations

import subprocess
import sys
import warnings
from importlib import metadata
from pathlib import Path

import pandas as pd
import pytest

from sysmexcbctools.data import XNSampleProcessor
from tests.data.conftest import haas_export
from tests.data.test_processing_record import _records

pytest.importorskip("bc_sysmex_format")

KEY = ["Sample No.", "Date", "Time"]
# The fixture's two measurements, in file (and datetime) order.
DATETIMES = ["20131227_095204", "20140110_090629"]


def build_dataset(root: Path, hospital: str, sample_nos=None) -> Path:
    from bc_sysmex_format.db import SysmexDataset

    csv = haas_export(root.parent / f"{root.name}_haas", sample_nos)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with SysmexDataset(root) as ds:
            ds.ingest_xn_sample(csv, hospital)
    return root


@pytest.fixture(scope="module")
def dataset_a(tmp_path_factory):
    return build_dataset(tmp_path_factory.mktemp("a") / "ds", "A", ["S-0001", "S-0002"])


@pytest.fixture(scope="module")
def dataset_b(tmp_path_factory):
    return build_dataset(tmp_path_factory.mktemp("b") / "ds", "B", ["S-0003", "S-0004"])


def clean(tmp_path, *args, **kwargs) -> pd.DataFrame:
    processor = XNSampleProcessor(output_dir=str(tmp_path / "out"), verbose=0)
    return processor.process_files(*args, **kwargs)


def test_direct_and_export_routes_give_identical_cleaned_output(dataset_a, tmp_path):
    """Identical but for one column the export renders differently.

    Export writes rows in storage order, so rows are compared by key.
    """
    export = tmp_path / "export"
    subprocess.run(
        [Path(sys.executable).parent / "bc-sysmex-format", "export", dataset_a, export,
         "--samples-sql", "SELECT DISTINCT sample_no FROM measurements",
         "--include", "xn_sample"],
        check=True, capture_output=True,
    )

    direct = clean(tmp_path, bc_sysmex_dataset_dir=str(dataset_a))
    exported = clean(tmp_path, str(export / "XN_SAMPLE.csv")).sort_values(
        KEY, ignore_index=True
    )

    assert list(direct["measurement_id"]) == [
        f"S-0001_{DATETIMES[0]}", f"S-0002_{DATETIMES[1]}"
    ]
    # The export writes Reception Date as DuckDB renders a TIMESTAMP; the
    # direct route keeps Sysmex's own format, as the HAAS route reads it.
    reception = "Reception Date"
    assert list(exported[reception]) == ["2013-12-27 08:53:04", "2014-01-10 09:04:19"]
    assert list(direct[reception]) == ["2013/12/27 08:53:04", "2014/01/10 09:04:19"]
    pd.testing.assert_frame_equal(
        direct.drop(columns=["measurement_id", reception]).sort_values(KEY, ignore_index=True),
        exported.drop(columns=reception),
    )


def test_haas_route_differs_from_direct_only_in_named_ways(dataset_a, tmp_path):
    direct = clean(tmp_path, bc_sysmex_dataset_dir=str(dataset_a))
    haas = clean(tmp_path, str(dataset_a.parent / "ds_haas" / "XN_SAMPLE.csv"))

    # 1. The direct route carries each row's measurement_id, first.
    assert direct.columns[0] == "measurement_id"
    direct = direct.drop(columns="measurement_id")
    # 2. The dataset stores a Q-Flag word (DISCRETE, ERROR) in a numeric column
    #    as NULL, so the flag it would have encoded is 0.
    flag = "Q-Flag(Giant PLT?)_disc"
    assert list(haas[flag]) == [1, 1]
    assert list(direct[flag]) == [0, 0]
    # Nothing else.
    pd.testing.assert_frame_equal(direct.drop(columns=flag), haas.drop(columns=flag))


@pytest.mark.parametrize(
    "filters, expected",
    [
        ({}, ["S-0001", "S-0002", "S-0003", "S-0004"]),
        ({"hospital_id": "B"}, ["S-0003", "S-0004"]),
        # Compared stripped, as on the HAAS route.
        ({"sample_nos": [" S-0002 ", "S-0003"]}, ["S-0002", "S-0003"]),
        ({"hospital_id": "A", "sample_nos": ["S-0002", "S-0003"]}, ["S-0002"]),
    ],
)
def test_several_datasets_are_combined_and_filtered(
    dataset_a, dataset_b, tmp_path, filters, expected
):
    df = clean(tmp_path, bc_sysmex_dataset_dir=[dataset_a, str(dataset_b)], **filters)

    assert sorted(df["Sample No."]) == expected


def test_filters_matching_nothing_are_refused(dataset_a, tmp_path):
    with pytest.raises(ValueError, match="No measurement"):
        clean(tmp_path, bc_sysmex_dataset_dir=str(dataset_a), hospital_id="B")


def test_a_measurement_in_two_datasets_is_refused(dataset_a, tmp_path):
    other = build_dataset(tmp_path / "other", "C", ["S-0009", "S-0002"])

    with pytest.raises(ValueError, match=f"S-0002_{DATETIMES[1]}"):
        clean(tmp_path, bc_sysmex_dataset_dir=[dataset_a, other])

    # The record says what was read before the refusal, and why the run failed.
    [record] = _records(tmp_path / "out")
    text = record.read_text(encoding="utf-8")
    assert [line for line in text.splitlines() if line.startswith("read ")] == [
        f"read 1 S-0001_{DATETIMES[0]}",
        f"read 1 S-0002_{DATETIMES[1]}",
        f"read 2 S-0002_{DATETIMES[1]}",
    ]
    assert "measurements_read: 3\noutcome: failure\n" in text
    assert "Traceback" in text and "is in two of the given datasets" in text


@pytest.mark.parametrize("path", ["missing", "ds/database.duckdb"])
def test_a_path_that_is_not_a_dataset_directory_is_refused(dataset_a, tmp_path, path):
    target = dataset_a.parent / path

    with pytest.raises(FileNotFoundError, match="No dataset"):
        clean(tmp_path, bc_sysmex_dataset_dir=str(target))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"input_files": "XN_SAMPLE.csv", "bc_sysmex_dataset_dir": "ds"}, "exactly one"),
        ({}, "exactly one"),
        ({"input_files": "XN_SAMPLE.csv", "hospital_id": "A"}, "hospital_id"),
    ],
)
def test_inputs_are_checked(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        clean(tmp_path, **kwargs)


def test_the_direct_route_requires_an_output_directory(dataset_a):
    processor = XNSampleProcessor(output_dir=None, verbose=0)

    with pytest.raises(ValueError, match="output_dir"):
        processor.process_files(bc_sysmex_dataset_dir=str(dataset_a))


def test_the_processing_record_names_datasets_reader_and_measurements(
    dataset_a, tmp_path
):
    qc = build_dataset(tmp_path / "qc", "QC")
    output_dir = tmp_path / "out"
    df = clean(tmp_path, bc_sysmex_dataset_dir=[str(dataset_a), str(qc)])

    [record] = _records(output_dir)
    lines = record.read_text(encoding="utf-8").splitlines()
    header = dict(line.split(": ", 1) for line in lines if ": " in line)

    assert header["reader"] == (
        f"bloodcounts-sysmex-format {metadata.version('bloodcounts-sysmex-format')}"
    )
    assert header["dataset 1"] == str(dataset_a.resolve())
    assert header["dataset 2"] == str(qc.resolve())
    assert not [k for k in header if k.startswith("input ")]
    assert [line for line in lines if line.startswith("read ")] == [
        f"read 1 S-0001_{DATETIMES[0]}",
        f"read 1 S-0002_{DATETIMES[1]}",
        f"read 2 QC-32941101_{DATETIMES[0]}",
        f"read 2 QC-32941101_{DATETIMES[1]}",
    ]
    assert header["measurements_read"] == "4"
    assert header["outcome"] == "success"
    # Dropped rows are keyed by measurement_id too.
    dropped = pd.read_csv(header["dropped_rows"], dtype=str)
    assert dropped.to_dict("records") == [
        {"measurement_id": f"QC-32941101_{dt}", "Sample No.": "QC-32941101",
         "Date": date, "Time": time, "step": "remove_technical_samples",
         "reason": "technical sample number (starts with 'QC-')"}
        for dt, date, time in (
            (DATETIMES[0], "2013/12/27", "09:52:04"),
            (DATETIMES[1], "2014/01/10", "09:06:29"),
        )
    ]
    assert sorted(df["Sample No."]) == ["S-0001", "S-0002"]
