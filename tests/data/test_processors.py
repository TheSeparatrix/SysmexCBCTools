"""Tests for individual cleaning steps in ``sysmexclean.processors``."""

from __future__ import annotations

import io
import logging

import pandas as pd

from sysmexcbctools.data.sysmexclean.processors import clean_non_numeric_values


def _quiet_logger() -> logging.Logger:
    logger = logging.getLogger("test_processors")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def _read(text: str) -> pd.DataFrame:
    """Read a CSV the way the cleaner's callers do, dtypes and all."""
    return pd.read_csv(io.StringIO(text))


def test_placeholders_become_nan():
    """Sysmex's ``----`` and blank placeholders must read as missing.

    Regression test for the dtype these steps guard on.  A column pandas does
    not parse as a number is ``object`` under pandas 2 but ``str`` from pandas
    3.0, so a check against ``"object"`` alone stops firing and the placeholders
    survive into the cleaned output as literal strings -- silently, because both
    of the pipeline's data sources degrade the same way.
    """
    df = _read(
        "measured,not_ordered,rack\n"
        "1.5,----,      \n"
        "2.5,----,A1\n"
    )
    assert not pd.api.types.is_numeric_dtype(df["not_ordered"]), "fixture is wrong"

    out = clean_non_numeric_values(df, _quiet_logger())

    assert out["not_ordered"].isna().all()
    assert pd.isna(out.loc[0, "rack"])
    assert out.loc[1, "rack"] == "A1"


def test_numeric_text_columns_are_converted():
    """A text column that is entirely numeric is coerced to a number."""
    df = _read("a,b\n1,x\n2,y\n").astype(str)

    out = clean_non_numeric_values(df, _quiet_logger())

    assert pd.api.types.is_numeric_dtype(out["a"])
    assert out["a"].tolist() == [1, 2]
    assert out["b"].tolist() == ["x", "y"]


def test_numeric_columns_are_left_alone():
    """Columns pandas already parsed as numbers are not touched."""
    df = _read("a,b\n1.5,2\n2.5,3\n")

    out = clean_non_numeric_values(df, _quiet_logger())

    assert out["a"].tolist() == [1.5, 2.5]
    assert out["b"].tolist() == [2, 3]
