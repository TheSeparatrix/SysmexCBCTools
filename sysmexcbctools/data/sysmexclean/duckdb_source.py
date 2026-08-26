"""Read XN_SAMPLE data from a bloodcounts-sysmex-format DuckDB dataset.

Provides an alternative to loading ``XN_SAMPLE.csv`` from disk: the same
table is read out of the DuckDB database at the root of a
``bloodcounts-sysmex-format`` dataset.  The returned DataFrame is shaped to
match ``pd.read_csv("XN_SAMPLE.csv")`` column-for-column, so it can be
handed straight to :meth:`XNSampleProcessor.process_files` and the rest of
the cleaning pipeline needs no changes.

Only the standard library, pandas and duckdb are used here -- the dataset's
DuckDB file is read directly, so ``bc_sysmex_format`` itself is not a
required import.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .constants import ID_COLUMNS

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Name of the DuckDB file at the root of a bloodcounts-sysmex-format dataset.
DATABASE_FILENAME = "database.duckdb"

# The dataset stores these three columns as native DATE/TIME/TIMESTAMP types,
# but XN_SAMPLE.csv holds them as strings in Sysmex's own formats.  They are
# rendered back to those exact formats on read: downstream code parses them
# with explicit formats and ``errors="raise"`` (see
# ``ancillary.build_survivor_keys``), so a different rendering is a hard
# failure, not a cosmetic difference.
_DATETIME_FORMATS = {
    "Date": "%Y/%m/%d",
    "Reception Date": "%Y/%m/%d %H:%M:%S",
}

# ``Time`` is a bare TIME; DuckDB's VARCHAR cast already yields "HH:MM:SS".
_TIME_COLUMN = "Time"

# XN_SAMPLE.csv has a duplicated header (many repeated parameter names) and one
# trailing empty header cell.  DuckDB and pandas resolve those differently, so
# the DuckDB names are translated back to the pandas ones on read:
#
#   duplicate ``X`` -> DuckDB ``X_1``,        pandas ``X.1``
#   blank at pos N  -> DuckDB ``columnN``,    pandas ``Unnamed: N``
_DUCKDB_UNNAMED_RE = re.compile(r"^column(\d+)$")
_DUCKDB_DUPLICATE_RE = re.compile(r"^(.*)_(\d+)$")

# Rows are returned in this order.  A DuckDB table has no inherent row order, so
# without an explicit sort the result would vary between runs; sorting also
# matters because the cleaning pipeline's duplicate-resolution sort is not
# stable.  This particular key reproduces the order Sysmex writes XN_SAMPLE.csv
# in (grouped by analyser, then chronological), but the guarantee this function
# makes is determinism, not byte-identical row order with any given CSV.
_ORDER_BY = ('"Analyzer ID"', '"Date"', '"Time"', '"Sample No."')
# Same key, unquoted -- used for the pandas-level sort applied when combining
# more than one dataset (see read_xn_sample_from_duckdb).
_ORDER_BY_COLUMNS = [c.strip('"') for c in _ORDER_BY]


def _to_read_csv_column_names(columns: list[str]) -> list[str]:
    """Translate DuckDB XN_SAMPLE column names into ``pd.read_csv`` names.

    Assumes no genuine column name ends in ``_<digits>`` while a column of the
    same stem appears earlier -- true of the Sysmex XN_SAMPLE header, where
    every such name is DuckDB's own de-duplication suffix.
    """
    seen: dict[str, int] = {}
    names: list[str] = []

    for position, name in enumerate(columns):
        unnamed = _DUCKDB_UNNAMED_RE.match(name)
        duplicate = _DUCKDB_DUPLICATE_RE.match(name)

        if unnamed is not None and int(unnamed.group(1)) == position:
            base = f"Unnamed: {position}"
        elif duplicate is not None and duplicate.group(1) in seen:
            base = duplicate.group(1)
        else:
            base = name

        occurrence = seen.get(base, 0)
        seen[base] = occurrence + 1
        names.append(base if occurrence == 0 else f"{base}.{occurrence}")

    return names


def resolve_database_path(dataset_root: str | Path) -> Path:
    """Return the path to the DuckDB file for *dataset_root*.

    Accepts either a dataset directory (the usual case) or a direct path to
    the ``.duckdb`` file itself.

    Raises
    ------
    FileNotFoundError
        If no DuckDB file exists at the resolved location.
    """
    path = Path(dataset_root)
    if path.suffix.lower() != ".duckdb":
        path = path / DATABASE_FILENAME

    if not path.is_file():
        raise FileNotFoundError(
            f"No bloodcounts-sysmex-format database found at {path}. "
            f"Expected a dataset directory containing {DATABASE_FILENAME}, "
            f"or a path to the .duckdb file itself."
        )
    return path


def _build_select(hospital_id: str | None, sample_nos: list[str] | None) -> str:
    """Build the XN_SAMPLE SELECT, reshaping columns to CSV form.

    ``* EXCLUDE ... REPLACE ...`` keeps the table's native column order (which
    is the CSV's column order) while dropping the dataset-internal
    ``measurement_id`` key and re-rendering the date/time columns, without
    naming the other ~490 columns.
    """
    replacements = [
        f"strftime(\"{col}\", '{fmt}') AS \"{col}\""
        for col, fmt in _DATETIME_FORMATS.items()
    ]
    replacements.append(f'CAST("{_TIME_COLUMN}" AS VARCHAR) AS "{_TIME_COLUMN}"')

    sql = (
        f'SELECT * EXCLUDE (measurement_id) REPLACE ({", ".join(replacements)}) '
        f'FROM "XN_SAMPLE"'
    )

    conditions = []
    if hospital_id is not None:
        # hospital_id lives on ``measurements``, not on the wide table.
        conditions.append(
            "measurement_id IN "
            "(SELECT measurement_id FROM measurements WHERE hospital_id = ?)"
        )
    if sample_nos is not None:
        conditions.append('"Sample No." IN (SELECT sn FROM _requested_samples)')

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    sql += " ORDER BY " + ", ".join(_ORDER_BY)

    return sql


def _read_one_root(
    db_path: Path,
    sample_nos: list[str] | None,
    hospital_id: str | None,
) -> pd.DataFrame:
    """Read XN_SAMPLE from one already-resolved ``.duckdb`` file."""
    logger.info("Reading XN_SAMPLE from %s", db_path)

    sql = _build_select(hospital_id, sample_nos)
    params = [hospital_id] if hospital_id is not None else []

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        if sample_nos is not None:
            con.register(
                "_requested_samples",
                pd.DataFrame({"sn": [str(s) for s in sample_nos]}),
            )
        df = con.execute(sql, params).fetchdf()
    finally:
        con.close()

    df.columns = _to_read_csv_column_names(list(df.columns))

    # Identifier columns are always strings so that purely numeric cohorts are
    # not silently inferred as int/float -- matching ``utils._read_single_file``.
    for col in ID_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str)

    logger.info("Loaded XN_SAMPLE with shape %s from %s", df.shape, db_path)
    return df


def read_xn_sample_from_duckdb(
    dataset_root: str | Path | Sequence[str | Path],
    sample_nos: list[str] | None = None,
    hospital_id: str | None = None,
) -> pd.DataFrame:
    """Load the ``XN_SAMPLE`` table from one or more bloodcounts-sysmex-format datasets.

    Parameters
    ----------
    dataset_root : str, Path, or sequence of str/Path
        A dataset directory containing ``database.duckdb`` (or a direct path
        to that file), or a sequence of several -- e.g. one dataset per
        hospital, to be combined. ``sample_nos`` / ``hospital_id`` are
        applied identically to each. Datasets transferred via
        ``bc-sysmex-format package --obfuscate`` (each site using its own
        secret) already have collision-safe ``Sample No.`` values across
        hospitals, so combining them needs no further reconciliation; see
        the duplicate check below for the case where that doesn't hold.
    sample_nos : list of str, optional
        If given, only rows whose ``"Sample No."`` appears in this list are
        returned.  Entries are cast to ``str`` before comparison, so integer
        sample numbers are handled transparently.
    hospital_id : str, optional
        If given, only rows belonging to this hospital are returned.

    Returns
    -------
    df : pd.DataFrame
        The XN_SAMPLE rows, with the same columns, column order and string
        formatting that ``pd.read_csv("XN_SAMPLE.csv")`` would produce.
        Suitable for passing directly as ``input_files`` to
        :meth:`XNSampleProcessor.process_files`. When more than one
        ``dataset_root`` is given, rows are the union of all of them, sorted
        by the same key as the single-dataset case for determinism.

    Raises
    ------
    ImportError
        If duckdb is not installed.
    FileNotFoundError
        If no DuckDB file exists at one of the given roots.
    ValueError
        If more than one dataset is given and the same ``(Sample No., Date,
        Time)`` appears in more than one of them -- refusing to guess which
        is authoritative rather than silently dropping or duplicating rows.
    """
    if duckdb is None:
        raise ImportError(
            "Reading from a DuckDB dataset requires duckdb. "
            "Install it with: pip install 'sysmexcbctools[data]'"
        )

    roots = (
        [dataset_root]
        if isinstance(dataset_root, (str, Path))
        else list(dataset_root)
    )
    if not roots:
        raise ValueError(
            "dataset_root must be a path, or a non-empty sequence of paths."
        )

    db_paths = [resolve_database_path(root) for root in roots]
    frames = [_read_one_root(p, sample_nos, hospital_id) for p in db_paths]

    if len(frames) == 1:
        return frames[0]

    df = pd.concat(frames, ignore_index=True)

    dup_mask = df.duplicated(subset=["Sample No.", "Date", "Time"], keep=False)
    if dup_mask.any():
        dup_keys = df.loc[dup_mask, ["Sample No.", "Date", "Time"]].drop_duplicates()
        dupes_preview = dup_keys.head(20).to_dict("records")
        raise ValueError(
            f"{len(dup_keys)} (Sample No., Date, Time) combination(s) appear in "
            f"more than one of the {len(roots)} combined datasets -- refusing "
            f"to guess which is authoritative. First few: {dupes_preview}. This is "
            f"expected to be impossible for datasets transferred via "
            f"`bc-sysmex-format package --obfuscate` with distinct per-site "
            f"secrets; check the source datasets."
        )

    df = df.sort_values(by=_ORDER_BY_COLUMNS, kind="stable").reset_index(drop=True)

    logger.info(
        "Combined XN_SAMPLE from %d datasets: shape %s", len(roots), df.shape
    )
    return df
