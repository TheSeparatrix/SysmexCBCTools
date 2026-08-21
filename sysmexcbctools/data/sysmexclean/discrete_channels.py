"""Which XN_SAMPLE columns each analyser channel populates.

A Sysmex XN analyser runs only the channels the operator ordered, and the
``Discrete`` column of ``XN_SAMPLE.csv`` records that order.  Fields belonging
to a channel that never ran are not left blank in the export: they are filled
with zeros, or occasionally with a constant artefact of computing on zeros
(``LFR(%) = 100.0``, ``RET-He(pg) = 5.3``).  Either way the value reads as a
measurement when it is nothing of the kind, so :func:`mask_unmeasured` replaces
it with NaN during cleaning.

Two tables compose to decide this, both taken from the Sysmex data dictionary:

1. :data:`DISCRETE_TO_CHANNELS` -- which mechanical channels each ``Discrete``
   application token drives (sheet ``Discrete vs Channels``).
2. ``COLUMN_CHANNELS`` in :mod:`sysmexcbctools.data.sysmexclean._channel_table`
   -- which mechanical channels populate each ``XN_SAMPLE`` column (sheet
   ``500set``).  That module is **generated**; see its docstring.

A column is *measured* when it is always-on (:data:`ALWAYS`) or when any channel
that populates it is active.  A column carrying no attribution at all is never
masked -- see :func:`unmeasured_columns`.

This supersedes the per-combination column lists compiled by Allerdien Visser
(Amsterdam UMC), which this module replaces.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from ._channel_table import ALWAYS, COLUMN_CHANNELS

__all__ = [
    "DISCRETE_TO_CHANNELS",
    "WHOLE_BLOOD_MODE",
    "channels_for_column",
    "mask_unmeasured",
    "parse_discrete",
    "unmeasured_by_discrete",
    "unmeasured_columns",
]

# Which mechanical channels each ``Discrete`` application token drives, keyed on
# the token exactly as ``Discrete`` spells it.  From the ``Discrete vs
# Channels`` sheet of the dictionary.
DISCRETE_TO_CHANNELS: dict[str, frozenset[str]] = {
    "CBC": frozenset({"CBC-RBC/PLT", "CBC-HGB", "CBC-WNR"}),
    "DIFF": frozenset({"CBC-WNR", "DIFF/WDF"}),
    "RET": frozenset({"CBC-RBC/PLT", "RET"}),
    "PLT-F": frozenset({"CBC-RBC/PLT", "PLT-F"}),
    "WPC": frozenset({"CBC-WNR", "DIFF/WDF", "WPC"}),
}

# Compact single-letter ``Discrete``.  ``W`` for WPC is inferred: no export seen
# so far carries a compact-form WPC order to confirm it.
_COMPACT_TO_TOKEN: dict[str, str] = {
    "C": "CBC",
    "D": "DIFF",
    "R": "RET",
    "P": "PLT-F",
    "W": "WPC",
}

# The compact form is recognised only when the whole value is compact letters.
# Without this, ``"FREE SELECT"`` would be read letter by letter as ``C`` + ``R``
# and mask DIFF and PLT-F on a row whose order we cannot actually decode.
_COMPACT_RE = re.compile(r"^[CDRPW]+$")

# A free-selection order does not name its channels in a form we can decode, so
# rows carrying one are left exactly as the analyser exported them.
_FREE_SELECT = "FREE SELECT"

#: ``Measurement Mode`` value for whole blood.  Body-fluid rows have their own
#: parameter set and a different channel story, so they are never masked.
WHOLE_BLOOD_MODE = "WB"

# pandas mangles duplicated CSV header names by appending ``.1``, ``.2``, ...
# and ``normalize_duplicate_columns`` spells the same thing ``_1``, ``_2``.
# Strip either form before looking a column up.
_MANGLED_SUFFIX = re.compile(r"[._]\d+$")

# A mark column ``X/M`` (data-quality flag for parameter ``X``) is measured
# exactly when its value column is.  Applying the value column's attribution
# rather than the mark row's own is one rule instead of a special case, and it
# corrects the dictionary's only observed error: ``PLT-I/M`` is attributed there
# to PLT-F, but ``PLT-I`` is a CBC-RBC/PLT parameter whose marks are real under
# ``CBC+DIFF``.
_MARK_SUFFIX = re.compile(r"/M$")


def parse_discrete(discrete: str | None) -> list[str]:
    """Parse a Discrete column value into its active measurement components.

    Parameters
    ----------
    discrete : str or None
        Value such as ``"CBC+DIFF+RET+PLT-F"`` or ``"CDRP"`` (compact form).
        The compact letter for WPC is inferred as ``W``; no export carrying a
        compact-form WPC order has been seen to confirm it.

    Returns
    -------
    list of str
        The tokens named by the field, in channel order (e.g.
        ``["CBC", "DIFF", "RET", "PLT-F"]``).  Empty when the value is missing,
        names nothing recognised, or is a free selection -- in every one of
        those cases the caller must leave the row untouched.
    """
    if not discrete or not isinstance(discrete, str):
        return []

    discrete = discrete.strip()
    if _FREE_SELECT in discrete.upper():
        return []

    if "+" in discrete:
        tokens = [t.strip() for t in discrete.split("+")]
    elif _COMPACT_RE.match(discrete.upper()):
        tokens = [_COMPACT_TO_TOKEN[ch] for ch in discrete.upper()]
    else:
        tokens = [discrete]

    return [t for t in tokens if t in DISCRETE_TO_CHANNELS]


def channels_for_column(
    column: str, all_columns: frozenset[str]
) -> frozenset[str] | None:
    """Return the channels populating ``column``, or None if unattributed.

    Parameters
    ----------
    column : str
        An ``XN_SAMPLE`` column name, possibly carrying a duplicated-header
        suffix (``X.1`` from pandas, ``X_1`` from
        ``processors.normalize_duplicate_columns``).
    all_columns : frozenset of str
        Every column of the frame, used to resolve a mark column ``X/M`` to its
        value column ``X(...)``.

    Returns
    -------
    frozenset of str or None
        The channel names, possibly containing :data:`ALWAYS`.  ``None`` when
        the dictionary gives the column no positive attribution, in which case
        it must never be masked.
    """
    base = _MANGLED_SUFFIX.sub("", column)
    stem = _MARK_SUFFIX.sub("", base)
    if stem != base:
        # A mark column: prefer its value column's attribution.
        for candidate in all_columns:
            other = _MANGLED_SUFFIX.sub("", candidate)
            if other.startswith(f"{stem}(") and not other.endswith("/M"):
                channels = COLUMN_CHANNELS.get(other)
                if channels:
                    return channels
    return COLUMN_CHANNELS.get(base)


def unmeasured_columns(
    discrete_components: Iterable[str], columns: Iterable[str]
) -> list[str]:
    """List the columns no active channel could have populated.

    Parameters
    ----------
    discrete_components : iterable of str
        Active ``Discrete`` tokens, as returned by :func:`parse_discrete`
        (e.g. ``["CBC", "DIFF", "RET", "PLT-F"]``).
    columns : iterable of str
        The columns of the frame being cleaned.

    Returns
    -------
    list of str
        Columns to set NaN, in input order.  A column the dictionary does not
        attribute to any channel is never returned: only a positive attribution
        justifies discarding a value.
    """
    active: set[str] = set()
    for component in discrete_components:
        active |= DISCRETE_TO_CHANNELS.get(component, frozenset())
    all_columns = frozenset(columns)
    unmeasured = []
    for column in columns:
        channels = channels_for_column(column, all_columns)
        if channels is None or ALWAYS in channels:
            continue
        if not (channels & active):
            unmeasured.append(column)
    return unmeasured


def unmeasured_by_discrete(
    df: pd.DataFrame, *, discrete_col: str = "Discrete"
) -> dict[str, list[str]]:
    """Map each ``Discrete`` value in *df* to the columns it leaves unmeasured.

    This is the masking plan :func:`mask_unmeasured` applies, exposed so a
    consumer of an already-cleaned file can reproduce it from the retained
    ``Discrete`` column and tell a structural NaN ("this channel never ran")
    from a genuinely missing value.

    Parameters
    ----------
    df : pandas.DataFrame
        An ``XN_SAMPLE`` frame.
    discrete_col : str, optional
        Column holding the ``Discrete`` order.  Default ``"Discrete"``.

    Returns
    -------
    dict of {str: list of str}
        Keyed on the ``Discrete`` value.  Values that name nothing decodable
        (missing, unrecognised, or a free selection) are absent from the result,
        as are values that leave nothing unmeasured.
    """
    if discrete_col not in df.columns:
        return {}

    plan: dict[str, list[str]] = {}
    for discrete in df[discrete_col].dropna().unique():
        components = parse_discrete(discrete)
        if not components:
            continue
        unmeasured = unmeasured_columns(components, df.columns)
        if unmeasured:
            plan[discrete] = unmeasured
    return plan


def mask_unmeasured(
    df: pd.DataFrame,
    *,
    discrete_col: str = "Discrete",
    mode_col: str = "Measurement Mode",
) -> tuple[pd.DataFrame, int]:
    """Set the unmeasured-channel columns of an XN_SAMPLE frame to NaN.

    Blanks, per ``Discrete`` value, every column that no active channel could
    have populated -- turning a fabricated ``RET%(%) = 0.00`` on a ``CBC+DIFF``
    sample into a NaN that reads as "not measured".

    Non-whole-blood rows, free-selection rows and rows whose ``Discrete`` names
    nothing recognised are left untouched, as is any column the dictionary does
    not attribute.

    Parameters
    ----------
    df : pandas.DataFrame
        An ``XN_SAMPLE`` frame.  Modified in place and returned.
    discrete_col : str, optional
        Column holding the ``Discrete`` order.  Default ``"Discrete"``.
    mode_col : str, optional
        Column holding the measurement mode.  Default ``"Measurement Mode"``.
        When absent, every row is treated as whole blood.

    Returns
    -------
    pandas.DataFrame
        The masked frame.
    int
        Number of cells set to NaN, for logging.
    """
    plan = unmeasured_by_discrete(df, discrete_col=discrete_col)
    if not plan:
        return df, 0

    if mode_col in df.columns:
        eligible = df[mode_col].isna() | (
            df[mode_col].astype(str).str.strip() == WHOLE_BLOOD_MODE
        )
    else:
        eligible = pd.Series(True, index=df.index)

    # Integer columns cannot hold NaN.  Widen the ones about to be masked up
    # front, once: assigning NaN into an int column raises under pandas >= 3.0
    # rather than silently upcasting, and ``encode_flags`` has already cast the
    # flag columns to int by the time this runs.
    int_cols = [
        c
        for c in {c for cols in plan.values() for c in cols}
        if pd.api.types.is_integer_dtype(df[c])
    ]
    if int_cols:
        df[int_cols] = df[int_cols].astype("float64")

    n_masked = 0
    for discrete, columns in plan.items():
        rows = eligible & (df[discrete_col] == discrete)
        if not rows.any():
            continue
        # Count only the cells that actually held something.
        n_masked += int(df.loc[rows, columns].notna().to_numpy().sum())
        df.loc[rows, columns] = np.nan

    return df, n_masked
