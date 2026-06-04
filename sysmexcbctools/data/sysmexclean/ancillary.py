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
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]

# Explicit duckdb ``read_csv`` options for OutputData.csv.
#
# OutputData.csv is an instrument-generated log and is occasionally
# malformed (truncated lines, stray quotes, very long rows).  DuckDB's
# automatic dialect sniffer raises ``InvalidInputException`` when it cannot
# confidently detect the parsing dialect of such files.  We therefore pin
# the dialect (comma-delimited, standard double-quote quoting/escaping)
# instead of relying on the sniffer, and enable tolerant parsing so that
# individual bad rows are skipped rather than aborting the whole read.
#
#   - ``all_varchar``    : read every column as text (no type inference)
#   - ``delim``/``quote``/``escape`` : pin the dialect, bypassing the sniffer
#   - ``ignore_errors``  : skip rows that do not conform to the dialect
#   - ``null_padding``   : pad short rows with NULLs instead of failing
#   - ``strict_mode``    : tolerate rows that break the CSV standard
#   - ``max_line_size``  : allow very long lines (instrument logs can be wide)
_OUTPUTDATA_READ_OPTS = (
    "all_varchar=true, delim=',', quote='\"', escape='\"', "
    "ignore_errors=true, null_padding=true, strict_mode=false, "
    "max_line_size=10000000"
)

# ---------------------------------------------------------------------------
# Sample number normalisation
# ---------------------------------------------------------------------------

# Characters stripped from both ends of a sample number before matching.
# Sample numbers occasionally carry leading/trailing separators (spaces,
# hyphens, underscores) that differ between the XN_SAMPLE table, the
# OutputData.csv log, and the SCT filename encoding.  Normalising all three
# the same way prevents spurious match failures.  ``str.strip`` with this
# character set removes any combination of these characters from each end.
_SAMPLE_NO_STRIP = " -_"


def _normalize_sample_no(value: object) -> str:
    """Normalise a single sample number for matching.

    Casts to ``str`` and strips leading/trailing spaces, hyphens, and
    underscores (in any combination).
    """
    return str(value).strip(_SAMPLE_NO_STRIP)


def _normalize_sample_no_series(values: pd.Series) -> pd.Series:
    """Vectorised :func:`_normalize_sample_no` for a pandas Series."""
    return values.astype(str).str.strip(_SAMPLE_NO_STRIP)


def _log_unmatched_keys(
    matching_keys: set[tuple[str, str]],
    matched_keys: set[tuple[str, str]],
    label: str,
    logger: logging.Logger,
    *,
    preview: int = 20,
) -> None:
    """Debug-log which surviving samples had no match in an ancillary source.

    Reconciles the surviving XN_SAMPLE keys (*matching_keys*) against the
    keys that were actually found in an ancillary source (*matched_keys*).
    A large unmatched count for SCT relative to OutputData is the typical
    signature of missing or unparseable SCT files.  Emitted at ``DEBUG``
    level so it stays out of normal output unless ``verbose >= 2``.

    Parameters
    ----------
    matching_keys : set of (str, str)
        All surviving ``(sample_no, datetime)`` keys from XN_SAMPLE.
    matched_keys : set of (str, str)
        Keys found in this ancillary source.
    label : str
        Human-readable source name (e.g. ``"OutputData"`` or ``"SCT"``).
    logger : logging.Logger
        Logger to emit debug messages on.
    preview : int, default=20
        Maximum number of unmatched keys to list.
    """
    n_total = len(matching_keys)
    unmatched = matching_keys - matched_keys
    n_matched = n_total - len(unmatched)
    logger.debug(
        "%s reconciliation: %d / %d surviving samples matched, %d unmatched",
        label, n_matched, n_total, len(unmatched),
    )
    if unmatched:
        sample = sorted(unmatched)[:preview]
        logger.debug(
            "%s: surviving samples with no match (showing %d of %d): %s",
            label, len(sample), len(unmatched), sample,
        )


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
    if len(df) == 0:
        return set()

    combined = df["Date"].astype(str) + " " + df["Time"].astype(str)
    dt = pd.to_datetime(combined, format="%Y/%m/%d %H:%M:%S", errors="raise")
    dt_strs = dt.dt.strftime("%Y%m%d_%H%M%S")
    sample_nos = _normalize_sample_no_series(df["Sample No."])
    return set(zip(sample_nos, dt_strs, strict=True))


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
    "decoded_filename", "channel", "analyzer", "unknown",
    "date_time", "sample_no",
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
    return (_normalize_sample_no(sample_no), dt_str)


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


def _ensure_parquet(
    archive_path: str | Path, logger: logging.Logger,
) -> Path:
    """Convert a CSV archive to Parquet if needed, returning the Parquet path.

    For ``.parquet`` or ``.pq`` files the path is returned unchanged.
    For ``.csv`` files a sibling ``.parquet`` is created (streamed via
    duckdb so memory stays bounded) and its path is returned.  If the
    sibling already exists the conversion is skipped.

    Parameters
    ----------
    archive_path : str or Path
        Path to a ``.csv``, ``.parquet``, or ``.pq`` archive.
    logger : logging.Logger
        Logger for conversion messages.

    Returns
    -------
    parquet_path : Path
        Path to the Parquet file (original or newly created).

    Raises
    ------
    ValueError
        If the file extension is not recognised.
    """
    path = Path(archive_path)
    ext = path.suffix.lower()

    if ext in {".parquet", ".pq"}:
        return path

    if ext != ".csv":
        raise ValueError(
            f"Unsupported archive extension '{ext}'. "
            "Expected .csv, .parquet, or .pq."
        )

    parquet_path = path.with_suffix(".parquet")
    if parquet_path.exists():
        logger.info(
            "Using existing Parquet conversion: %s", parquet_path.name,
        )
        return parquet_path

    logger.info("Converting CSV archive to Parquet: %s", path.name)
    csv_escaped = str(path).replace("'", "''")
    pq_escaped = str(parquet_path).replace("'", "''")
    # Read all columns as VARCHAR to avoid type-inference failures (e.g.
    # sample_no looks numeric in the first rows but contains text later).
    duckdb.sql(
        f"COPY (SELECT * FROM read_csv('{csv_escaped}', all_varchar=true)) "
        f"TO '{pq_escaped}' (FORMAT PARQUET)"
    )
    return parquet_path


def _build_sct_index(
    archive_paths: list[Path],
    matching_keys: set[tuple[str, str]],
    logger: logging.Logger,
) -> dict[str, tuple[Path, list[str]]]:
    """Build an index of matching SCT base filenames across archives.

    Reads only the ``decoded_filename`` column from each archive
    (extremely fast via Parquet column projection), then filters against
    *matching_keys*.

    Parameters
    ----------
    archive_paths : list of Path
        Paths to Parquet archive files.
    matching_keys : set of (str, str)
        Composite keys ``(sample_no, YYYYMMDD_HHMMSS)``.
    logger : logging.Logger
        Logger for warnings.

    Returns
    -------
    index : dict[str, tuple[Path, list[str]]]
        Mapping of ``base_filename`` to
        ``(archive_path, [decoded_filename_variants])``.  If the same
        *base_filename* appears in multiple archives only the first
        archive encountered is used.
    """
    index: dict[str, tuple[Path, list[str]]] = {}

    for archive_path in archive_paths:
        path_escaped = str(archive_path).replace("'", "''")
        try:
            filenames = duckdb.sql(
                f"SELECT DISTINCT decoded_filename "
                f"FROM '{path_escaped}'"
            ).fetchdf()["decoded_filename"]
        except duckdb.BinderException:
            logger.warning(
                "Archive %s missing 'decoded_filename' column -- skipping",
                archive_path,
            )
            continue

        # Group decoded_filename variants by normalised base filename
        base_groups: dict[str, list[str]] = defaultdict(list)
        for fn in filenames:
            base_fn = _normalize_overflow_filename(fn)
            base_groups[base_fn].append(fn)

        for base_fn, variants in base_groups.items():
            if base_fn in index:
                continue  # first-archive-wins deduplication

            parsed = _parse_sct_filename(base_fn)
            if parsed is None:
                continue

            sample_no, dt_str = parsed
            if (sample_no, dt_str) in matching_keys:
                index[base_fn] = (archive_path, variants)

    return index


def reconstruct_sct_from_archives(
    archive_files: list[str],
    matching_keys: set[tuple[str, str]],
    output_sct_dir: str | Path,
    logger: logging.Logger,
    *,
    chunk_size: int = 500_000,
) -> int:
    """Reconstruct individual SCT files from consolidated archives.

    Uses a three-phase approach for efficiency:

    1. **Normalise** -- convert any CSV archives to Parquet (streamed,
       bounded memory).
    2. **Index** -- read only the ``decoded_filename`` column to identify
       which files to reconstruct.
    3. **Write** -- fetch each file's rows via Parquet predicate pushdown
       and write them as individual SCT CSVs with a progress bar.

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
        Kept for backward compatibility; ignored.

    Returns
    -------
    n_written : int
        Number of SCT files written.

    Raises
    ------
    ImportError
        If *duckdb* is not installed.
    """
    if duckdb is None:
        raise ImportError(
            "duckdb is required for SCT archive reconstruction. "
            "Install it with: pip install 'sysmexcbctools[data]'"
        )

    out_dir = Path(output_sct_dir)
    n_archives = len(archive_files)
    logger.info(
        "Processing %d SCT archive file(s) for reconstruction", n_archives,
    )

    # Phase 1 -- Normalise: ensure all archives are Parquet
    parquet_paths: list[Path] = []
    for archive_path in archive_files:
        if not Path(archive_path).exists():
            logger.warning(
                "Archive file not found: %s -- skipping", archive_path,
            )
            continue
        parquet_paths.append(_ensure_parquet(archive_path, logger))

    # Phase 2 -- Index: lightweight scan of decoded_filename only
    index = _build_sct_index(parquet_paths, matching_keys, logger)
    logger.info("Found %d matching SCT files to reconstruct", len(index))

    matched_keys = {
        parsed
        for base_fn in index
        if (parsed := _parse_sct_filename(base_fn)) is not None
    }
    _log_unmatched_keys(matching_keys, matched_keys, "SCT (archive)", logger)

    # Phase 3 -- Write: one file at a time with progress bar
    n_written = 0
    for base_fn, (archive_path, variants) in tqdm(
        index.items(), desc="Reconstructing SCT files", unit="file",
    ):
        dest = out_dir / base_fn
        if dest.exists():
            continue

        path_escaped = str(archive_path).replace("'", "''")
        placeholders = ", ".join(
            f"'{v.replace(chr(39), chr(39) + chr(39))}'" for v in variants
        )
        df = duckdb.sql(
            f"SELECT * FROM '{path_escaped}' "
            f"WHERE decoded_filename IN ({placeholders})"
        ).fetchdf()

        # Drop metadata columns
        meta_to_drop = [c for c in df.columns if c in _ARCHIVE_METADATA_COLS]
        df = df.drop(columns=meta_to_drop)

        # Drop all-NaN columns (restores channel-specific column sets)
        df = df.dropna(axis=1, how="all")

        # Restore column casing
        df = df.rename(
            columns={c: _ARCHIVE_COLUMN_MAP.get(c, c) for c in df.columns}
        )

        df.to_csv(dest, index=False)
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
    chunk_rows: int = 200_000,
) -> int:
    """Stream, filter, and incrementally write OutputData rows to disk.

    Each source directory's ``OutputData.csv`` is scanned in bounded chunks
    via duckdb (all columns read as ``VARCHAR`` to avoid type-inference
    issues), filtered against *matching_keys* with vectorised datetime
    parsing, and appended to *output_path*.  Peak memory is therefore
    independent of file size.  Rows that share a composite key with a row
    already written are silently skipped (deduplication).

    The output format is inferred from *output_path*'s extension:

    - ``.csv`` -- appended row-wise via pandas.
    - ``.parquet`` / ``.pq`` -- written as a single Parquet file, one
      row-group per streamed chunk via ``pyarrow.parquet.ParquetWriter``.
      Each column is written as a string (consistent with the
      ``all_varchar=true`` read).  Recommended for large datasets --
      the resulting file is several-fold smaller and round-trips faster.

    Parameters
    ----------
    source_dirs : list of Path
        Directories that may contain an ``OutputData.csv``.
    matching_keys : set of (str, str)
        Composite keys ``(sample_no, YYYYMMDD_HHMMSS)`` of surviving samples.
    output_path : str or Path
        Destination path (``.csv``, ``.parquet``, or ``.pq``).  Created
        or overwritten on first write.
    logger : logging.Logger
        Logger for warnings.
    columns : list of str, optional
        If given, only these columns are written to *output_path*.  The
        key columns (``Sample No.``, ``AnalyzeDate``, ``AnalyzeTime``)
        are always read from disk for matching but may be excluded from
        the output if not listed here.
    chunk_rows : int, default=200_000
        Approximate number of rows per streamed chunk.  Larger values
        trade memory for fewer pandas round-trips.

    Returns
    -------
    n_written : int
        Total number of rows written.

    Raises
    ------
    ImportError
        If *duckdb* is not installed, or if parquet output is requested
        but *pyarrow* is not installed.
    ValueError
        If *output_path* has an unsupported extension.
    """
    if duckdb is None:
        raise ImportError(
            "duckdb is required for OutputData.csv filtering. "
            "Install it with: pip install 'sysmexcbctools[data]'"
        )

    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        write_format = "csv"
    elif suffix in {".parquet", ".pq"}:
        write_format = "parquet"
    else:
        raise ValueError(
            f"Unsupported output extension '{suffix}' for {output_path}. "
            f"Supported: .csv, .parquet, .pq"
        )

    pq_writer = None  # lazy-initialised ParquetWriter
    pq_schema = None
    if write_format == "parquet":
        try:
            import pyarrow as pa  # noqa: F401
            import pyarrow.parquet as pq  # noqa: F401
        except ImportError:
            raise ImportError(
                "Writing parquet output requires pyarrow. "
                "Install it with: pip install pyarrow"
            )

    seen_keys: set[tuple[str, str]] = set()
    total_written = 0
    header_written = False  # tracks whether any output has been produced

    _KEY_COLS = {"Sample No.", "AnalyzeDate", "AnalyzeTime"}

    for d in source_dirs:
        od_path = d / "OutputData.csv"
        if not od_path.exists():
            logger.warning("OutputData.csv not found in %s -- skipping", d)
            continue

        path_escaped = str(od_path).replace("'", "''")

        # Probe header to determine available columns and column order.
        # A genuinely corrupted file can defeat even the pinned-dialect read
        # (e.g. an unreadable header line); skip it with a warning rather
        # than aborting the whole run.
        try:
            header_df = duckdb.sql(
                f"SELECT * FROM read_csv('{path_escaped}', "
                f"{_OUTPUTDATA_READ_OPTS}) LIMIT 0"
            ).fetchdf()
        except duckdb.Error as exc:
            logger.warning(
                "OutputData.csv in %s could not be read (%s) -- skipping",
                d, exc,
            )
            continue
        available = set(header_df.columns)

        missing_keys = _KEY_COLS - available
        if missing_keys:
            logger.warning(
                "OutputData.csv in %s missing key column(s) %s -- skipping",
                d, ", ".join(sorted(missing_keys)),
            )
            continue

        if columns is not None:
            load_cols = [c for c in header_df.columns
                         if c in (_KEY_COLS | set(columns))]
            missing = set(columns) - available
            if missing:
                logger.warning(
                    "OutputData.csv in %s missing requested columns: %s",
                    d, ", ".join(sorted(missing)),
                )
        else:
            load_cols = list(header_df.columns)

        col_list = ", ".join(f'"{c}"' for c in load_cols)
        # ``_OUTPUTDATA_READ_OPTS`` pins the dialect and enables tolerant
        # parsing so malformed rows (e.g. truncated lines with fewer columns
        # than the header) are skipped rather than aborting the read.
        # OutputData.csv is an instrument-generated log and occasionally
        # contains such rows; a row missing most of its columns would fail
        # the key match anyway, so silently skipping is safe.
        select_sql = (
            f"SELECT {col_list} "
            f"FROM read_csv('{path_escaped}', {_OUTPUTDATA_READ_OPTS})"
        )

        kept_in_file = 0
        seen_in_file = 0

        # Stream rows via duckdb's arrow iterator in bounded batches.
        # Use the connection-level ``fetch_record_batch`` rather than a
        # relation method: ``DuckDBPyRelation.__getattr__`` is overloaded to
        # do column lookups, so attribute names that differ across versions
        # (``to_arrow_reader`` / ``fetch_arrow_reader``) raise a confusing
        # "no such column" error on older installations. Going via the
        # connection sidesteps that and is available in every version.
        # A corrupted file can still raise mid-stream; skip the rest of it
        # with a warning rather than aborting the whole run.  Rows already
        # written for this file are preserved.  The read is driven one batch
        # at a time so peak memory stays bounded -- the iterator is never
        # fully materialised.
        try:
            reader = duckdb.execute(select_sql).fetch_record_batch(chunk_rows)
        except duckdb.Error as exc:
            logger.warning(
                "OutputData.csv in %s failed during read (%s) -- skipping",
                d, exc,
            )
            continue

        while True:
            # Fetch the next batch in its own guard so a mid-stream read
            # error is distinguished from an error in our row processing.
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break
            except duckdb.Error as exc:
                logger.warning(
                    "OutputData.csv in %s failed mid-stream (%s) -- "
                    "keeping rows already written, skipping the rest",
                    d, exc,
                )
                break

            chunk = batch.to_pandas()
            if chunk.empty:
                continue
            seen_in_file += len(chunk)

            combined = (
                chunk["AnalyzeDate"].astype(str)
                + " "
                + chunk["AnalyzeTime"].astype(str)
            )
            dt = pd.to_datetime(
                combined, format="%Y/%m/%d %H:%M:%S", errors="coerce",
            )
            sample_nos = _normalize_sample_no_series(chunk["Sample No."])
            dt_strs = dt.dt.strftime("%Y%m%d_%H%M%S")

            candidates = pd.Series(
                list(zip(sample_nos, dt_strs, strict=True)),
                index=chunk.index,
            )
            in_match = dt.notna() & candidates.isin(matching_keys)
            if not in_match.any():
                continue

            matched_pairs = candidates[in_match]
            not_seen = ~matched_pairs.isin(seen_keys)
            keep_pairs = matched_pairs[not_seen]
            if keep_pairs.empty:
                continue
            seen_keys.update(keep_pairs.tolist())

            filtered = chunk.loc[keep_pairs.index]
            if columns is not None:
                filtered = filtered[[c for c in columns
                                     if c in filtered.columns]]

            if write_format == "csv":
                filtered.to_csv(
                    output_path,
                    mode="a" if header_written else "w",
                    header=not header_written,
                    index=False,
                )
            else:
                import pyarrow as pa
                import pyarrow.parquet as pq
                table = pa.Table.from_pandas(filtered, preserve_index=False)
                if pq_writer is None:
                    pq_schema = table.schema
                    pq_writer = pq.ParquetWriter(str(output_path), pq_schema)
                else:
                    # Ensure consistent schema across chunks
                    table = table.cast(pq_schema, safe=False)
                pq_writer.write_table(table)

            header_written = True
            total_written += len(filtered)
            kept_in_file += len(filtered)

        logger.info(
            "OutputData.csv in %s: kept %d / %d rows",
            d, kept_in_file, seen_in_file,
        )

    if pq_writer is not None:
        pq_writer.close()

    _log_unmatched_keys(matching_keys, seen_keys, "OutputData", logger)

    if not header_written and columns is not None:
        # Write an empty file with the requested column header so
        # downstream code always finds a valid output.
        empty = pd.DataFrame(columns=columns)
        if write_format == "csv":
            empty.to_csv(output_path, index=False)
        else:
            empty.astype(str).to_parquet(output_path, index=False)

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
    matched_keys: set[tuple[str, str]] = set()

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
            matched_keys.add((sample_no, dt_str))

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

    _log_unmatched_keys(matching_keys, matched_keys, "SCT", logger)
    logger.info("Wrote %d SCT files to %s", n_written, output_sct_dir)
    return n_written
