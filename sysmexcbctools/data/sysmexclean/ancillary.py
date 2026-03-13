"""Ancillary file matching and copying for OutputData.csv and SCT files.

Provides functions to filter OutputData.csv rows and copy SCT files that
match the samples surviving the XNSampleProcessor pipeline.  Matching uses
a composite key of (Sample No., normalised datetime) so that duplicate
measurements of the same sample are correctly distinguished.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Matching key construction
# ---------------------------------------------------------------------------

def build_matching_keys(df: pd.DataFrame) -> set[tuple[str, str]]:
    """Build a set of composite keys from the processed DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame with ``Sample No.``, ``Date`` (``YYYY/MM/DD``),
        and ``Time`` (``HH:MM:SS``) columns.

    Returns
    -------
    keys : set of (str, str)
        Set of ``(sample_no, datetime_str)`` tuples where *datetime_str* is
        formatted as ``YYYYMMDD_HHMMSS``.
    """
    keys: set[tuple[str, str]] = set()
    for _, row in df.iterrows():
        dt = datetime.strptime(
            f"{row['Date']} {row['Time']}", "%Y/%m/%d %H:%M:%S"
        )
        dt_str = dt.strftime("%Y%m%d_%H%M%S")
        keys.add((str(row["Sample No."]).strip(), dt_str))
    return keys


def derive_source_dirs(input_files: list[str]) -> list[Path]:
    """Extract unique parent directories from input file paths.

    Parameters
    ----------
    input_files : list of str
        Paths to XN_SAMPLE.csv files.

    Returns
    -------
    dirs : list of Path
        Unique parent directories, in discovery order.
    """
    seen: set[Path] = set()
    dirs: list[Path] = []
    for f in input_files:
        p = Path(f).resolve().parent
        if p not in seen:
            seen.add(p)
            dirs.append(p)
    return dirs


# ---------------------------------------------------------------------------
# SCT filename parsing
# ---------------------------------------------------------------------------

# Regex mirrors parse_sysmex_raw_filename() in
# sysmexcbctools/transfer/sysmexalign/load_and_preprocess.py:236-274
_SCT_FILENAME_RE = re.compile(
    r"""
    ^
    [A-Z]+                              # channel name
    _\[[^\]]+\]                         # [analyser ID]
    \[[^\]]+\]                          # [middle section]
    \[(\d{8}_\d{6})\]                   # [datetime] -- captured
    \[\s*([^\]]+)\]                     # [sample number] -- captured
    \.116\.csv$                         # extension (base file only)
    """,
    re.VERBOSE,
)


def _parse_sct_filename(filename: str) -> tuple[str, str] | None:
    """Extract sample number and datetime from an SCT base filename.

    Parameters
    ----------
    filename : str
        Basename of the SCT file (e.g.
        ``WDF_[XN-10^11036][00-15_5][20150203_091038][  QC-43211103].116.csv``).

    Returns
    -------
    result : tuple of (str, str) or None
        ``(sample_no_stripped, datetime_YYYYMMDD_HHMMSS)`` on success,
        ``None`` if *filename* does not match the expected pattern.
    """
    m = _SCT_FILENAME_RE.match(filename)
    if m is None:
        return None
    dt_str, sample_no = m.groups()
    return (sample_no.strip(), dt_str)


# ---------------------------------------------------------------------------
# Overflow file handling
# ---------------------------------------------------------------------------

def _find_overflow_files(base_path: Path) -> list[Path]:
    """Find overflow sibling files for a base ``.116.csv`` file.

    Sysmex creates ``.116(1).csv``, ``.116(2).csv``, etc. when a file
    exceeds a size limit.

    Parameters
    ----------
    base_path : Path
        Path to the base ``.116.csv`` file.

    Returns
    -------
    overflow_paths : list of Path
        Sorted overflow file paths.  Empty if none exist.
    """
    stem = str(base_path)[:-4]  # strip .csv
    overflow: list[Path] = []
    for n in range(1, 100):
        candidate = Path(f"{stem}({n}).csv")
        if candidate.exists():
            overflow.append(candidate)
        elif n > 1:
            break
    return overflow


def _read_and_merge_sct(
    base_path: Path, overflow_paths: list[Path]
) -> pd.DataFrame:
    """Read a base SCT CSV and concatenate any overflow files.

    Parameters
    ----------
    base_path : Path
        Path to the base ``.116.csv`` file.
    overflow_paths : list of Path
        Paths to overflow files (may be empty).

    Returns
    -------
    merged : pd.DataFrame
        Vertically concatenated data.
    """
    frames = [pd.read_csv(base_path)]
    for ovf in overflow_paths:
        frames.append(pd.read_csv(ovf))
    return pd.concat(frames, axis=0, ignore_index=True)


# ---------------------------------------------------------------------------
# OutputData.csv filtering
# ---------------------------------------------------------------------------

def filter_output_data(
    source_dirs: list[Path],
    matching_keys: set[tuple[str, str]],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Load and filter OutputData.csv files to matching samples.

    Parameters
    ----------
    source_dirs : list of Path
        Directories that may contain an ``OutputData.csv``.
    matching_keys : set of (str, str)
        Composite keys ``(sample_no, YYYYMMDD_HHMMSS)`` of surviving samples.
    logger : logging.Logger
        Logger for warnings.

    Returns
    -------
    filtered : pd.DataFrame
        Concatenated, filtered rows from all source directories.
    """
    frames: list[pd.DataFrame] = []

    for d in source_dirs:
        od_path = d / "OutputData.csv"
        if not od_path.exists():
            logger.warning("OutputData.csv not found in %s -- skipping", d)
            continue

        od = pd.read_csv(od_path, low_memory=False)

        keep_mask = pd.Series(False, index=od.index)
        for idx, row in od.iterrows():
            try:
                dt = datetime.strptime(
                    f"{row['AnalyzeDate']} {row['AnalyzeTime']}",
                    "%Y/%m/%d %H:%M:%S",
                )
            except (ValueError, TypeError):
                continue
            dt_str = dt.strftime("%Y%m%d_%H%M%S")
            sample_no = str(row["Sample No."]).strip()
            if (sample_no, dt_str) in matching_keys:
                keep_mask.at[idx] = True

        filtered = od.loc[keep_mask]
        logger.info(
            "OutputData.csv in %s: kept %d / %d rows",
            d, len(filtered), len(od),
        )
        frames.append(filtered)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# SCT file copying
# ---------------------------------------------------------------------------

def copy_matching_sct_files(
    source_dirs: list[Path],
    matching_keys: set[tuple[str, str]],
    output_sct_dir: str,
    logger: logging.Logger,
) -> int:
    """Copy matching SCT files, merging overflow files into the base.

    Parameters
    ----------
    source_dirs : list of Path
        Directories that may contain an ``SCT/`` subdirectory.
    matching_keys : set of (str, str)
        Composite keys ``(sample_no, YYYYMMDD_HHMMSS)`` of surviving samples.
    output_sct_dir : str
        Destination directory for consolidated SCT files.
    logger : logging.Logger
        Logger for warnings.

    Returns
    -------
    n_written : int
        Number of files written.
    """
    n_written = 0
    out_dir = Path(output_sct_dir)

    for d in source_dirs:
        sct_dir = d / "SCT"
        if not sct_dir.is_dir():
            logger.warning("SCT/ directory not found in %s -- skipping", d)
            continue

        for entry in os.scandir(sct_dir):
            if not entry.is_file():
                continue

            parsed = _parse_sct_filename(entry.name)
            if parsed is None:
                # Overflow files (.116(N).csv) and non-matching names are
                # skipped here; overflows are picked up via _find_overflow_files
                continue

            sample_no, dt_str = parsed
            if (sample_no, dt_str) not in matching_keys:
                continue

            base_path = Path(entry.path)
            overflow_paths = _find_overflow_files(base_path)
            merged = _read_and_merge_sct(base_path, overflow_paths)
            dest = out_dir / entry.name
            merged.to_csv(dest, index=False)
            n_written += 1

    logger.info("Wrote %d SCT files to %s", n_written, output_sct_dir)
    return n_written
