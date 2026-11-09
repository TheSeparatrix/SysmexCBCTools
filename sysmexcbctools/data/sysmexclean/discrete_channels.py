"""Which XN_SAMPLE columns each analyser channel populates.

A Sysmex XN analyser runs only the channels the operator ordered, and the
``Discrete`` column of ``XN_SAMPLE.csv`` records that order.  Fields belonging
to a channel that never ran are not left blank in the export: they are filled
with zeros, or occasionally with a constant artefact of computing on zeros
(``LFR(%) = 100.0``, ``RET-He(pg) = 5.3``).  Either way the value reads as a
measurement when it is nothing of the kind, so :func:`mask_unmeasured` replaces
it with NaN during cleaning.

The rule is bloodcounts-sysmex-format's (``bc_sysmex_format.discrete_channels``),
copied so the HAAS route does not depend on the format: everything from
:data:`DISCRETE_TO_CHANNELS` to :func:`unmeasured_columns` is the format's code,
and a test compares the two column for column whenever the format is installed.
Change the rule there first, then copy it here.

Two tables compose to decide this, both taken from the AUMC (Amsterdam UMC) XN
data dictionary, ``SysmexDictionary_2025 Aug25.xlsx`` -- compiled by AUMC from
Sysmex documentation, not issued by Sysmex:

1. :data:`DISCRETE_TO_CHANNELS` -- which mechanical channels each ``Discrete``
   application token drives (sheet ``Discrete vs Channels``).
2. ``COLUMN_CHANNELS`` in :mod:`sysmexcbctools.data.sysmexclean._channel_table`
   -- which mechanical channels populate each ``XN_SAMPLE`` column (sheet
   ``500set``).  That module is a **generated** copy of the format's; see its
   docstring.

A column is *measured* when it is always-on (:data:`ALWAYS`) or when any channel
that populates it is active -- or, for a value derived from several channels
(``REQUIRES_ALL``: ``NEUT#`` = % from WDF x WBC from WNR, the ``DLT-*`` ratios),
when every one is.  A column carrying no attribution at all is never masked --
see :func:`is_measured`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from ._channel_table import ALWAYS, COLUMN_CHANNELS, REQUIRES_ALL

__all__ = [
    "ALWAYS",
    "COLUMN_CHANNELS",
    "DISCRETE_TO_CHANNELS",
    "WHOLE_BLOOD_MODE",
    "channels_for_column",
    "is_measured",
    "mask_unmeasured",
    "parse_discrete",
    "unmeasured_by_discrete",
    "unmeasured_columns",
]

# Which mechanical channels each ``Discrete`` application token drives, keyed on
# the canonical component names :func:`parse_discrete` produces (so ``PLTF``,
# not ``PLT-F``).  From the ``Discrete vs Channels`` sheet of the dictionary.
DISCRETE_TO_CHANNELS: dict[str, frozenset[str]] = {
    "CBC": frozenset({"CBC-RBC/PLT", "CBC-HGB", "CBC-WNR"}),
    "DIFF": frozenset({"CBC-WNR", "DIFF/WDF"}),
    "RET": frozenset({"CBC-RBC/PLT", "RET"}),
    "PLTF": frozenset({"CBC-RBC/PLT", "PLT-F"}),
    "WPC": frozenset({"CBC-WNR", "DIFF/WDF", "WPC"}),
}

# ``Discrete`` spells PLT-F with a hyphen; ``measurement_components`` and
# :data:`DISCRETE_TO_CHANNELS` use the hyphen-free form.
_DISCRETE_TO_COMPONENT: dict[str, str] = {
    "CBC": "CBC",
    "DIFF": "DIFF",
    "RET": "RET",
    "PLT-F": "PLTF",
    "WPC": "WPC",
}

# Compact single-letter ``Discrete``, accepted only when the whole value is made
# of these letters (``FREE SELECT`` is not ``R`` + ``C``).  ``W`` for WPC is
# inferred: no export seen so far carries a compact-form WPC order to confirm it.
_COMPACT_TO_DISCRETE: dict[str, str] = {
    "C": "CBC",
    "D": "DIFF",
    "R": "RET",
    "P": "PLT-F",
    "W": "WPC",
}

#: ``Measurement Mode`` value for whole blood.  Body-fluid rows have their own
#: parameter set and a different channel story, so they are never masked.
WHOLE_BLOOD_MODE = "WB"

# Real exports repeat 35 XN_SAMPLE headers.  The dataset no longer stores the
# repeats, but a frame read with plain ``pd.read_csv`` --
# ``scripts/check_discrete_mask.py`` -- has pandas' ``.1``, ``.2``, and a dataset
# built before ``repair`` dropped them has ``_1``, ``_2``.  Strip either form
# before looking a column up.
_MANGLED_SUFFIX = re.compile(r"[._]\d+$")

# Corrections to the dictionary, applied over the generated table.
_OVERRIDES: dict[str, frozenset[str]] = {
    # Every channel cell is NO, but AUMC's own definition reads "total nucleated
    # cell count (WBC + NRBC) calculated from the WNR channel", and real exports
    # carry TNC = TNC-N.
    "TNC(10^3/uL)": frozenset({"CBC-WNR"}),
    # Missing from the dictionary; every other differential-count ABN flag
    # (Neutrophilia, Lymphopenia, ...) is WDF.
    "IP ABN(WBC)Neutoropenia": frozenset({"DIFF/WDF"}),
}
_CHANNELS: dict[str, frozenset[str]] = {**COLUMN_CHANNELS, **_OVERRIDES}

# A mark column ``X/M`` (data-quality flag for parameter ``X``) is measured
# exactly when its value column ``X(unit)`` is.  Applying the value column's
# attribution rather than the mark row's own is one rule instead of a special
# case, and it corrects a dictionary error: ``PLT-I/M`` is attributed there to
# PLT-F, but ``PLT-I`` is a CBC-RBC/PLT parameter whose marks are real under
# ``CBC+DIFF``.  The value column must be exactly ``X(...)`` -- a prefix match
# would also find ``X(hsA)(unit)``, which has different channels.
_VALUE_COLUMN = re.compile(r"(.+)\([^()]*\)")
_MARK_TO_VALUE: dict[str, str] = {
    f"{m.group(1)}/M": key for key in _CHANNELS if (m := _VALUE_COLUMN.fullmatch(key))
}


def parse_discrete(discrete: str | None) -> list[str]:
    """Parse a Discrete column value into its active measurement components.

    Parameters
    ----------
    discrete : str or None
        Value such as ``"CBC+DIFF+RET+PLT-F"`` or ``"CDRP"`` (compact form),
        in any case.  The compact letter for WPC is inferred as ``W``; no export
        carrying a compact-form WPC order has been seen to confirm it.

    Returns
    -------
    list of str
        The ``measurement_components.component_type`` values named by the field,
        in channel order (e.g. ``["CBC", "DIFF", "RET", "PLTF"]``).  Empty when
        the value is missing or holds anything unrecognised (``FREE SELECT``,
        ``CBC+FOO``): naming only some of the channels that ran would mask the
        others.
    """
    if not discrete or not isinstance(discrete, str):
        return []

    value = discrete.strip().upper()
    if "+" in value or value in _DISCRETE_TO_COMPONENT:
        tokens = [t.strip() for t in value.split("+")]
    elif set(value) <= _COMPACT_TO_DISCRETE.keys():
        tokens = [_COMPACT_TO_DISCRETE[ch] for ch in value]
    else:
        return []

    if any(t not in _DISCRETE_TO_COMPONENT for t in tokens):
        return []
    return list(dict.fromkeys(_DISCRETE_TO_COMPONENT[t] for t in tokens))


def _attribution_key(column: str) -> str:
    """The dictionary entry deciding ``column``: its value column for a mark."""
    base = _MANGLED_SUFFIX.sub("", column)
    return _MARK_TO_VALUE.get(base, base)


def channels_for_column(column: str) -> frozenset[str] | None:
    """Return the channels populating ``column``, or None if unattributed.

    Parameters
    ----------
    column : str
        An ``XN_SAMPLE`` column name, possibly carrying a duplicated-header
        suffix (``X.1`` from pandas, ``X_1`` from the schema).  A mark column
        ``X/M`` resolves to its value column ``X(unit)``.

    Returns
    -------
    frozenset of str or None
        The channel names, possibly containing :data:`ALWAYS`.  ``None`` when
        the dictionary gives the column no positive attribution, in which case
        it must never be masked.
    """
    return _CHANNELS.get(_attribution_key(column))


def is_measured(column: str, active: set[str] | frozenset[str]) -> bool | None:
    """Whether the active channels could have populated ``column``.

    Parameters
    ----------
    column : str
        An ``XN_SAMPLE`` column name (see :func:`channels_for_column`).
    active : set of str
        The mechanical channels that ran.

    Returns
    -------
    bool or None
        ``None`` when the column is unattributed (never mask it).  Otherwise
        True when it is always-on, when every channel of a ``REQUIRES_ALL``
        column ran, or when any channel of any other column ran.
    """
    key = _attribution_key(column)
    channels = _CHANNELS.get(key)
    if channels is None:
        return None
    if ALWAYS in channels:
        return True
    if key in REQUIRES_ALL:
        return channels <= active
    return bool(channels & active)


def unmeasured_columns(
    discrete_components: Iterable[str], columns: Iterable[str]
) -> list[str]:
    """List the columns no active channel could have populated.

    Parameters
    ----------
    discrete_components : iterable of str
        Active measurement components, as returned by :func:`parse_discrete`
        (e.g. ``["CBC", "DIFF", "RET", "PLTF"]``).
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
    return [c for c in columns if is_measured(c, active) is False]


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
