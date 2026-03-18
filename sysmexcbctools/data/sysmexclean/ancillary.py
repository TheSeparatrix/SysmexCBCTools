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
from collections.abc import Iterator
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

_OVERFLOW_SUFFIX_RE = re.compile(r"\.116\(\d+\)\.csv$")

# Metadata columns added during archive consolidation (not in original SCT files)
_ARCHIVE_METADATA_COLS = frozenset({
    "decoded_filename", "channel", "date_time", "sample_no",
})

# Lowercase archive column name -> original SCT column name
_ARCHIVE_COLUMN_MAP: dict[str, str] = {
    "repeatcount": "RepeatCount",
    "phase": "Phase",
    "particleid": "ParticleID",
    "fsc": "FSC",
    "ssc": "SSC",
    "sfl": "SFL",
    "fscw": "FSCW",
    "fsclog": "FSClog",
    "sflx2": "SFLx2",
}


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
# Archive helpers (for reconstructing SCT files from consolidated archives)
# ---------------------------------------------------------------------------

def _normalize_overflow_filename(filename: str) -> str:
    """Strip overflow suffix from an SCT filename.

    Parameters
    ----------
    filename : str
        SCT filename, possibly with an overflow suffix such as
        ``.116(1).csv``.

    Returns
    -------
    normalized : str
        Filename with the overflow suffix replaced by ``.116.csv``.
        If the filename has no overflow suffix it is returned unchanged.
    """
    return _OVERFLOW_SUFFIX_RE.sub(".116.csv", filename)


def _parse_sct_decoded_filename(filename: str) -> tuple[str, str] | None:
    """Parse a ``decoded_filename`` value from an SCT archive.

    Normalises overflow suffixes before delegating to
    :func:`_parse_sct_filename`.

    Parameters
    ----------
    filename : str
        Value from the ``decoded_filename`` column.

    Returns
    -------
    result : tuple of (str, str) or None
        ``(sample_no, datetime_str)`` on success, ``None`` on failure.
    """
    return _parse_sct_filename(_normalize_overflow_filename(filename))


def _iter_archive_chunks(
    path: str | Path, chunk_size: int,
) -> "Iterator[pd.DataFrame]":
    """Yield DataFrames from a CSV or Parquet archive file.

    Parameters
    ----------
    path : str or Path
        Path to a ``.csv`` or ``.parquet``/``.pq`` archive.
    chunk_size : int
        Approximate number of rows per chunk.

    Yields
    ------
    chunk : pd.DataFrame

    Raises
    ------
    ValueError
        If the file extension is not recognised.
    ImportError
        If reading Parquet and *pyarrow* is not installed.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".csv":
        with pd.read_csv(path, chunksize=chunk_size) as reader:
            yield from reader
    elif ext in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "pyarrow is required to read Parquet archives. "
                "Install it with: pip install pyarrow"
            )
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=chunk_size):
            yield batch.to_pandas()
    else:
        raise ValueError(
            f"Unsupported archive extension '{ext}'. "
            "Expected .csv, .parquet, or .pq."
        )


def reconstruct_sct_from_archives(
    archive_files: list[str],
    matching_keys: set[tuple[str, str]],
    output_sct_dir: str | Path,
    logger: logging.Logger,
    *,
    chunk_size: int = 500_000,
) -> int:
    """Reconstruct individual SCT files from consolidated archive CSVs.

    The archives contain rows from many samples with lowercase column
    names and extra metadata columns.  This function filters to matching
    samples, drops metadata and all-NaN columns, restores original
    column casing, and writes one CSV per original base filename.

    Parameters
    ----------
    archive_files : list of str
        Paths to archive files (``.csv``, ``.parquet``, or ``.pq``).
    matching_keys : set of (str, str)
        Composite keys ``(sample_no, YYYYMMDD_HHMMSS)`` of surviving
        samples.
    output_sct_dir : str or Path
        Destination directory for reconstructed SCT files.
    logger : logging.Logger
        Logger for warnings and progress messages.
    chunk_size : int, default=500_000
        Rows per chunk when streaming archives.

    Returns
    -------
    n_written : int
        Number of SCT files written.
    """
    out_dir = Path(output_sct_dir)
    accumulated: dict[str, list[pd.DataFrame]] = {}
    written_files: set[str] = set()

    for archive_path in archive_files:
        if not Path(archive_path).exists():
            logger.warning("Archive file not found: %s -- skipping", archive_path)
            continue

        for chunk in _iter_archive_chunks(archive_path, chunk_size):
            if "decoded_filename" not in chunk.columns:
                logger.warning(
                    "Archive %s missing 'decoded_filename' column -- skipping chunk",
                    archive_path,
                )
                continue

            # Normalise filenames and parse keys
            chunk = chunk.copy()
            chunk["_base_filename"] = chunk["decoded_filename"].apply(
                _normalize_overflow_filename
            )
            chunk["_parsed"] = chunk["_base_filename"].apply(_parse_sct_filename)

            for base_fn, group in chunk.groupby("_base_filename"):
                parsed = group["_parsed"].iloc[0]
                if parsed is None:
                    continue
                sample_no, dt_str = parsed
                if (sample_no, dt_str) not in matching_keys:
                    continue
                if base_fn in written_files:
                    continue

                rows = group.drop(columns=["_base_filename", "_parsed"])
                accumulated.setdefault(base_fn, []).append(rows)

    # Write accumulated data
    n_written = 0
    for base_fn, frames in accumulated.items():
        if base_fn in written_files:
            continue

        merged = pd.concat(frames, axis=0, ignore_index=True)

        # Drop metadata columns
        meta_to_drop = [c for c in merged.columns if c in _ARCHIVE_METADATA_COLS]
        merged = merged.drop(columns=meta_to_drop)

        # Drop all-NaN columns (restores channel-specific column sets)
        merged = merged.dropna(axis=1, how="all")

        # Restore column casing
        merged = merged.rename(
            columns={c: _ARCHIVE_COLUMN_MAP.get(c, c) for c in merged.columns}
        )

        merged.to_csv(out_dir / base_fn, index=False)
        written_files.add(base_fn)
        n_written += 1

    logger.info(
        "Reconstructed %d SCT files from archives to %s",
        n_written, output_sct_dir,
    )
    return n_written


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
    output_path: str | Path,
    logger: logging.Logger,
    *,
    columns: list[str] | None = None,
) -> int:
    """Load, filter, and incrementally write OutputData.csv rows to disk.

    Processes one source directory at a time and appends matching rows to
    *output_path*, keeping peak memory low.  Rows that share a composite
    key with a row already written are silently skipped (deduplication).

    Parameters
    ----------
    source_dirs : list of Path
        Directories that may contain an ``OutputData.csv``.
    matching_keys : set of (str, str)
        Composite keys ``(sample_no, YYYYMMDD_HHMMSS)`` of surviving samples.
    output_path : str or Path
        Destination CSV file.  Created (or overwritten) on first write,
        then appended to.
    logger : logging.Logger
        Logger for warnings.
    columns : list of str, optional
        If given, only these columns are written to *output_path*.  The
        key columns (``Sample No.``, ``AnalyzeDate``, ``AnalyzeTime``)
        are always read from disk for matching but may be excluded from
        the output if not listed here.

    Returns
    -------
    n_written : int
        Total number of rows written.
    """
    output_path = Path(output_path)
    seen_keys: set[tuple[str, str]] = set()
    total_written = 0
    header_written = False

    _KEY_COLS = {"Sample No.", "AnalyzeDate", "AnalyzeTime"}

    for d in source_dirs:
        od_path = d / "OutputData.csv"
        if not od_path.exists():
            logger.warning("OutputData.csv not found in %s -- skipping", d)
            continue

        # Determine which columns to load from disk
        if columns is not None:
            available = set(pd.read_csv(od_path, nrows=0).columns)
            load_cols = list((_KEY_COLS | set(columns)) & available)
            missing = set(columns) - available
            if missing:
                logger.warning(
                    "OutputData.csv in %s missing requested columns: %s",
                    d, ", ".join(sorted(missing)),
                )
        else:
            load_cols = None

        od = pd.read_csv(od_path, usecols=load_cols, low_memory=False)

        # Filter rows: must match a surviving sample and not already written
        keep_indices: list[int] = []
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
            key = (sample_no, dt_str)
            if key in matching_keys and key not in seen_keys:
                keep_indices.append(idx)
                seen_keys.add(key)

        filtered = od.loc[keep_indices]

        # Select only requested output columns
        if columns is not None:
            out_cols = [c for c in columns if c in filtered.columns]
            filtered = filtered[out_cols]

        logger.info(
            "OutputData.csv in %s: kept %d / %d rows",
            d, len(filtered), len(od),
        )

        if not filtered.empty:
            filtered.to_csv(
                output_path,
                mode="a" if header_written else "w",
                header=not header_written,
                index=False,
            )
            header_written = True
            total_written += len(filtered)

        # Free memory before loading next file
        del od, filtered

    if not header_written and columns is not None:
        # Write an empty file with the requested column header so
        # downstream code always finds a valid CSV.
        pd.DataFrame(columns=columns).to_csv(output_path, index=False)

    return total_written


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

            # Skip duplicates across source directories
            dest = out_dir / entry.name
            if dest.exists():
                logger.debug(
                    "SCT file %s already written -- skipping duplicate",
                    entry.name,
                )
                continue

            base_path = Path(entry.path)
            overflow_paths = _find_overflow_files(base_path)
            merged = _read_and_merge_sct(base_path, overflow_paths)
            merged.to_csv(dest, index=False)
            n_written += 1

    logger.info("Wrote %d SCT files to %s", n_written, output_sct_dir)
    return n_written
