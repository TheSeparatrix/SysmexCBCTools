"""The processing record of one cleaning run (see ``GLOSSARY.md``).

How each measurement was processed, for the researcher: a plain-text
``key: value`` file in ``<output_dir>/logs/`` opening with the run header
(UTC start, user, host, tool version, settings, inputs, output location),
then every removed column and the per-column counts of the value-changing
steps, closing with the outcome (traceback on failure) and UTC end.  Every
dropped row goes to a sidecar CSV (row key, step, reason) named in the header.

On the direct route the header names the dataset directories and the
bloodcounts-sysmex-format version, a ``read <dataset> <measurement_id>`` line
names each measurement read, and rows are keyed by ``measurement_id`` too: the
audit record the format's ADR 0002 asks of a tool writing per-measurement data
out of a dataset.

Both files are written as the run goes -- the record line-buffered, the CSV
appended step by step -- so a crashed run leaves everything up to the crash.
"""

from __future__ import annotations

import getpass
import json
import os
import socket
import subprocess
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

#: Columns that name a row in the dropped-row CSV.
ROW_KEY = ["Sample No.", "Date", "Time"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(*args) -> str:
    package = Path(__file__).resolve().parent
    try:
        return subprocess.run(
            ["git", "-C", str(package), *args], capture_output=True, text=True
        ).stdout.strip()
    except OSError:  # no git
        return ""


def _tool() -> str:
    """Name and version, plus the git commit only when run from this tool's own
    checkout: an install inside some other repository is not described by that
    repository's commit."""
    version = metadata.version("sysmexcbctools")
    checkout = Path(__file__).resolve().parents[3]
    toplevel = _git("rev-parse", "--show-toplevel")
    commit = _git("rev-parse", "HEAD") if toplevel and Path(toplevel).resolve() == checkout else ""
    return f"sysmexcbctools {version}" + (f" (git {commit})" if commit else "")


class ProcessingRecord:
    """Writes to an open record; without one (no output directory) does nothing."""

    def __init__(self, f=None, rows_path: Path | None = None, row_key=ROW_KEY):
        self._f = f
        self.rows_path = rows_path
        self.row_key = row_key
        self.measurements_read = 0

    def __bool__(self) -> bool:
        return self._f is not None

    def write(self, key: str, value) -> None:
        if self._f is not None:
            self._f.write(f"{key}: {value}\n")

    def dropped(self, step: str, rows: pd.DataFrame, reasons) -> None:
        """Append *rows* (the dropped rows, as they were) to the sidecar CSV."""
        if self._f is not None and len(rows):
            out = rows.reindex(columns=self.row_key).assign(step=step, reason=reasons)
            out.to_csv(self.rows_path, mode="a", header=False, index=False)

    def read(self, dataset_index: int, measurement_id: str) -> None:
        """Name a measurement read from the *dataset_index*-th (1-based) dataset."""
        if self._f is not None:
            self._f.write(f"read {dataset_index} {measurement_id}\n")
            self.measurements_read += 1

    def removed_column(self, column: str, step: str, reason: str) -> None:
        self.write(
            "removed_column", json.dumps({"column": column, "step": step, "reason": reason})
        )

    def cells(self, step: str, change: str, counts: pd.Series) -> None:
        """Per-column counts of the cells *step* changed this way (nonzero only)."""
        self.write(
            f"cells {step} {change}", json.dumps({c: int(n) for c, n in counts.items() if n})
        )


@contextmanager
def processing_record(
    output_dir, settings: dict, arguments: dict, inputs, logger, dataset_dirs=None
):
    """Open the record for one run and yield it as a :class:`ProcessingRecord`.

    *inputs* is a list of file paths or a DataFrame, or None on the direct
    route, which gives *dataset_dirs* instead.  With no *output_dir* nothing is
    written.  An exception is recorded as a failure with its traceback and
    re-raised.
    """
    if output_dir is None:
        yield ProcessingRecord()
        return

    logdir = Path(output_dir).resolve() / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = logdir / f"processing_record_{stamp}_{os.getpid()}.log"
    rows_path = path.with_name(f"{path.stem}_dropped_rows.csv")
    row_key = ROW_KEY if dataset_dirs is None else ["measurement_id", *ROW_KEY]
    pd.DataFrame(columns=[*row_key, "step", "reason"]).to_csv(rows_path, index=False)

    with open(path, "x", buffering=1, encoding="utf-8") as f:
        rec = ProcessingRecord(f, rows_path, row_key)
        rec.write("started_utc", _utc_now())
        try:
            rec.write("user", getpass.getuser())
            rec.write("host", socket.gethostname())
            rec.write("tool", _tool())
            if dataset_dirs is not None:
                version = metadata.version("bloodcounts-sysmex-format")
                rec.write("reader", f"bloodcounts-sysmex-format {version}")
            rec.write("settings", json.dumps(settings, default=str))
            rec.write("arguments", json.dumps(arguments, default=str))
            if dataset_dirs is not None:
                for i, p in enumerate(dataset_dirs, 1):
                    rec.write(f"dataset {i}", Path(p).resolve())
            elif isinstance(inputs, pd.DataFrame):
                shape = f"{inputs.shape[0]} rows, {inputs.shape[1]} columns"
                rec.write("input 1", f"DataFrame ({shape})")
            else:
                for i, p in enumerate(inputs, 1):
                    rec.write(f"input {i}", Path(p).resolve())
            rec.write("output_dir", Path(output_dir).resolve())
            rec.write("dropped_rows", rows_path)
            logger.info(f"Processing record: {path}")
            yield rec
        except BaseException:
            if dataset_dirs is not None:
                rec.write("measurements_read", rec.measurements_read)
            rec.write("outcome", "failure")
            f.writelines("  " + line for line in traceback.format_exc().splitlines(True))
            raise
        else:
            if dataset_dirs is not None:
                rec.write("measurements_read", rec.measurements_read)
            rec.write("outcome", "success")
        finally:
            rec.write("ended_utc", _utc_now())
            logger.info(f"Processing record: {path}")
