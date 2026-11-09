"""The direct route: XN_SAMPLE from Sysmex datasets, through the read API only.

Imported only when ``bc_sysmex_dataset_dir`` is given, so the HAAS route needs
neither bloodcounts-sysmex-format nor duckdb.
"""

from __future__ import annotations

import duckdb
import pandas as pd
from bc_sysmex_format import reader

BATCH_SIZE = 10_000

# The dataset stores these as DATE / TIMESTAMP; XN_SAMPLE.csv holds Sysmex's
# strings, which later steps parse with these exact formats.
_DATETIME_FORMATS = {"Date": "%Y/%m/%d", "Reception Date": "%Y/%m/%d %H:%M:%S"}


def read_xn_sample(dataset_dirs, hospital_id, sample_nos, on_read):
    """XN_SAMPLE of the selected measurements, shaped as the HAAS route loads it.

    ``measurement_id`` first, then the stored XN_SAMPLE columns in their order,
    with ``Date``, ``Time`` and ``Reception Date`` as Sysmex's strings and
    ``Sample No.`` a string.  Measurements are selected from ``measurements()``
    by *hospital_id* and *sample_nos* (compared stripped, as on the HAAS route)
    and read *BATCH_SIZE* at a time, in ``measurement_id`` order per dataset;
    ``on_read(i, measurement_id)`` is called for each XN_SAMPLE row read from the
    *i*-th (1-based) of *dataset_dirs*.

    Raises ``FileNotFoundError`` for a path that is not a dataset directory and
    ``ValueError`` for a measurement in more than one of *dataset_dirs*.
    """
    frames, seen = [], {}
    for i, root in enumerate(dataset_dirs, 1):
        # xn_sample runs on its own cursor: the batches are still being fetched.
        with reader.open_dataset(root) as con, con.cursor() as cur:
            rel = reader.measurements(con)
            if hospital_id is not None:
                rel = rel.filter(
                    duckdb.ColumnExpression("hospital_id")
                    == duckdb.ConstantExpression(str(hospital_id))
                )
            if sample_nos is not None:
                rel = rel.filter(
                    duckdb.FunctionExpression("trim", duckdb.ColumnExpression("sample_no"))
                    .isin(*(duckdb.ConstantExpression(str(s).strip()) for s in sample_nos))
                )
            # xn_sample returns rows in measurement_id order; batches follow it.
            rel = rel.order("measurement_id").select("measurement_id")
            while batch := rel.fetchmany(BATCH_SIZE):
                frame = reader.xn_sample(cur, [mid for (mid,) in batch])
                for mid in frame["measurement_id"]:
                    on_read(i, mid)
                    if mid in seen:
                        raise ValueError(
                            f"Measurement {mid} is in two of the given datasets "
                            f"({dataset_dirs[seen[mid] - 1]} and {root}); refusing to "
                            "guess which is authoritative."
                        )
                    seen[mid] = i
                frames.append(frame)

    if not frames:
        raise ValueError("No measurement in the given datasets matches the filters.")
    df = pd.concat(frames, ignore_index=True)
    for col, fmt in _DATETIME_FORMATS.items():
        df[col] = df[col].dt.strftime(fmt)
    df["Time"] = df["Time"].map(str, na_action="ignore")
    df["Sample No."] = df["Sample No."].astype(str)
    # BIGINT with NULLs comes back Int64, whose pd.NA the cleaning steps cannot
    # compare; read_csv gives such a column float64 with NaN.
    nullable = df.columns[df.dtypes == "Int64"]
    df[nullable] = df[nullable].astype("float64")
    return df
