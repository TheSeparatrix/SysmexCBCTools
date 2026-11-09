"""The processing record a cleaning run leaves in its output directory.

See ``GLOSSARY.md``: how each measurement was processed -- the settings and
inputs, every row dropped and why, every column removed and why, and the
outcome.  Written whenever the processor has an output directory.
"""

from __future__ import annotations

import getpass
import json
import socket
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

import sysmexcbctools
from sysmexcbctools.data import XNSampleProcessor

REPO = Path(__file__).resolve().parents[2]

_FBC = ["WBC(10^3/uL)", "RBC(10^6/uL)", "HGB(g/dL)", "HCT(%)", "PLT(10^3/uL)"]
_CLUMPS = "IP SUS(PLT)PLT Clumps?"


def _row(sample, date, time, fbc=(6.0, 4.5, 14.0, 42.0, 250.0), **extra):
    row = {
        "Sample No.": sample,
        "Date": date,
        "Time": time,
        "Measurement Mode": "WB",
        "Discrete": "CBC+DIFF",
        "Patient ID": "P",  # removed as unused
        "(Reserved)": "0",  # removed as unused
        **dict(zip(_FBC, fbc)),
        "MCV(fL)": "90.1",
        "RET%(%)": "0.00",  # zero fill: RET never ran under CBC+DIFF
        "HGB_SI(mmol/L)": "8.7",  # removed as redundant
        "IP SUS(RBC)Turbidity/HGB Interf?": "",
        "IP SUS(RBC)RBC Agglutination?": "",
        _CLUMPS: "",
    }
    row.update(extra)
    return row


@pytest.fixture
def xn_sample(tmp_path) -> Path:
    first = _row("A001", "2024/01/02", "09:00:00", **{"MCV(fL)": "----"})
    rows = [
        first,
        first,  # exact duplicate
        _row("A001", "2024/01/03", "10:00:00"),  # agreeing repeat
        _row("QC-123", "2024/01/02", "08:00:00"),  # technical sample
        _row("B002", "2024/01/02", "11:00:00", **{_CLUMPS: "1"}),  # clot
        _row("C003", "2024/01/02", "12:00:00", fbc=("----", 4.8, 15.0, 44.0, 260.0)),
        # Two measurements of D004 that disagree on every FBC parameter.
        _row("D004", "2024/01/02", "13:00:00", fbc=(5.0, 4.0, 12.0, 38.0, 200.0)),
        _row("D004", "2024/01/03", "13:00:00", fbc=(500.0, 9.0, 25.0, 80.0, 900.0)),
    ]
    path = tmp_path / "in" / "XN_SAMPLE.csv"
    path.parent.mkdir()
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _records(output_dir: Path) -> list[Path]:
    return sorted((output_dir / "logs").glob("processing_record_*.log"))


def _parse(path: Path) -> tuple[dict[str, str], list[dict]]:
    """``({key: value}, [removed_column entries])`` of a record."""
    header, removed = {}, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(" ") or ": " not in line:
            continue  # traceback
        key, value = line.split(": ", 1)
        if key == "removed_column":
            removed.append(json.loads(value))
        else:
            header[key] = value
    return header, removed


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    assert parsed.utcoffset() == timedelta(0)
    return parsed


@pytest.mark.parametrize("log_to_file", [False, True])
def test_a_run_records_how_each_measurement_was_processed(
    xn_sample, tmp_path, log_to_file
):
    output_dir = tmp_path / "out"
    before = datetime.now(timezone.utc)
    processor = XNSampleProcessor(
        output_dir=str(output_dir), log_to_file=log_to_file, verbose=0
    )
    df = processor.process_files(str(xn_sample), dataset_name="cohort")

    [record] = _records(output_dir)
    header, removed = _parse(record)
    commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    # Header
    assert before <= _utc(header["started_utc"]) <= _utc(header["ended_utc"])
    assert header["user"] == getpass.getuser()
    assert header["host"] == socket.gethostname()
    assert header["tool"] == f"sysmexcbctools {sysmexcbctools.__version__} (git {commit})"
    settings = json.loads(header["settings"])
    assert settings == {
        "remove_clotintube": True,
        "remove_multimeasurementsamples": True,
        "remove_correlated": False,
        "std_threshold": 1.0,
        "keep_drop_rows": False,
        "make_dummy_marks": False,
        "drop_empty_columns": True,
        "use_memory_optimized": True,
        "enable_memory_monitoring": True,
        "output_dir": str(output_dir),
        "output_prefix": "XN_SAMPLE_processed",
        "log_to_file": log_to_file,
        "verbose": 0,
        "config_path": None,
    }
    assert json.loads(header["arguments"])["dataset_name"] == "cohort"
    assert header["input 1"] == str(xn_sample.resolve())
    assert header["output_dir"] == str(output_dir.resolve())

    # Every dropped row, with the step and the reason
    dropped = pd.read_csv(header["dropped_rows"], dtype=str)
    assert dropped.to_dict("records") == [
        {"Sample No.": "A001", "Date": "2024/01/02", "Time": "09:00:00",
         "step": "remove_duplicate_rows", "reason": "exact duplicate of another row"},
        {"Sample No.": "QC-123", "Date": "2024/01/02", "Time": "08:00:00",
         "step": "remove_technical_samples",
         "reason": "technical sample number (starts with 'QC-')"},
        {"Sample No.": "B002", "Date": "2024/01/02", "Time": "11:00:00",
         "step": "remove_clot_in_tube_samples", "reason": f"clot in tube: {_CLUMPS}"},
        {"Sample No.": "A001", "Date": "2024/01/03", "Time": "10:00:00",
         "step": "handle_multiple_measurements",
         "reason": "repeat measurement agreeing with the one kept"},
        {"Sample No.": "D004", "Date": "2024/01/02", "Time": "13:00:00",
         "step": "handle_multiple_measurements",
         "reason": "repeat measurements disagree; none kept"},
        {"Sample No.": "D004", "Date": "2024/01/03", "Time": "13:00:00",
         "step": "handle_multiple_measurements",
         "reason": "repeat measurements disagree; none kept"},
    ]
    assert sorted(df["Sample No."]) == ["A001", "C003"]

    # Every removed column, with the step and the reason
    assert removed == [
        {"column": "Patient ID", "step": "remove_unused_columns",
         "reason": "not needed or not interpretable"},
        {"column": "(Reserved)", "step": "remove_unused_columns",
         "reason": "not needed or not interpretable"},
        {"column": "HGB_SI(mmol/L)", "step": "remove_redundant_columns",
         "reason": "redundant HGB column"},
        {"column": "RET%(%)", "step": "remove_empty_columns",
         "reason": "empty for every row"},
    ]

    # Per-column counts of the cells each value-changing step touched
    assert json.loads(header["cells encode_flags filled_0"]) == {
        "IP SUS(RBC)Turbidity/HGB Interf?": 6,
        "IP SUS(RBC)RBC Agglutination?": 6,
        _CLUMPS: 5,
    }
    assert json.loads(header["cells encode_flags set_nan"]) == {}
    assert json.loads(header["cells mask_unmeasured_channels set_nan"]) == {
        "RET%(%)": 6
    }
    # A placeholder in an FBC column is coerced before the non-numeric cleanup.
    assert json.loads(header["cells coerce_fbc_to_numeric set_nan"]) == {
        "WBC(10^3/uL)": 1
    }
    assert json.loads(header["cells clean_non_numeric_values set_nan"]) == {
        "MCV(fL)": 1
    }

    assert header["outcome"] == "success"


def test_keep_drop_rows_records_only_real_rows(xn_sample, tmp_path):
    """The unoptimised multiple-measurement step once flagged column labels as
    rows, adding one phantom row per column to the frame and the record."""
    output_dir = tmp_path / "out"
    processor = XNSampleProcessor(
        output_dir=str(output_dir), keep_drop_rows=True, use_memory_optimized=False,
        verbose=0,
    )
    processor.process_files(str(xn_sample))

    [record] = _records(output_dir)
    dropped = pd.read_csv(_parse(record)[0]["dropped_rows"], dtype=str)
    assert set(dropped["Sample No."]) <= {"A001", "D004"}


def test_a_failed_run_records_its_failure(tmp_path):
    output_dir = tmp_path / "out"
    processor = XNSampleProcessor(output_dir=str(output_dir), verbose=0)

    with pytest.raises(ValueError, match="No valid dataframes"):
        processor.process_files(str(tmp_path / "missing.csv"))

    [record] = _records(output_dir)
    text = record.read_text(encoding="utf-8")
    header, _ = _parse(record)
    assert header["outcome"] == "failure"
    assert "Traceback" in text and "No valid dataframes loaded" in text
    assert _utc(header["started_utc"]) <= _utc(header["ended_utc"])
    assert text.splitlines()[-1].startswith("ended_utc: ")


def test_without_an_output_directory_nothing_is_written(xn_sample, tmp_path, monkeypatch):
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    processor = XNSampleProcessor(output_dir=None, verbose=0)

    df = processor.process_files(str(xn_sample))

    assert sorted(df["Sample No."]) == ["A001", "C003"]
    assert list(workdir.iterdir()) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cwd", "in"]
    with pytest.raises(ValueError, match="output_dir"):
        processor.process_files(str(xn_sample), save_output=True)
