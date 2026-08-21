"""Discrete-driven NaN-filling of columns whose analyser channel never ran.

``XN_SAMPLE.csv`` zero-fills the fields of channels absent from the ``Discrete``
order, so a ``CBC+DIFF`` sample carries ``RET%(%) = 0.00`` -- indistinguishable
from a genuine reticulocyte count of zero.  ``discrete_channels.mask_unmeasured``
replaces those with NaN during cleaning; these tests pin the rule, its
carve-outs, and the two pipeline invariants that depend on where it runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sysmexcbctools.data.sysmexclean.discrete_channels import (
    mask_unmeasured,
    parse_discrete,
    unmeasured_by_discrete,
)
from sysmexcbctools.data.sysmexclean.processors import (
    encode_flags,
    mask_unmeasured_channels,
    remove_empty_columns,
)

# One column per channel of interest, plus the carve-out cases.  Values are the
# analyser's own fill for the channels that did not run.
_COLUMNS = {
    "Sample No.": "S1",
    "Date": "2024/01/02",
    "Time": "03:04:05",
    "Measurement Mode": "WB",
    "Discrete": "CBC+DIFF",
    "WBC(10^3/uL)": "6.5",  # CBC-WNR
    "HGB(g/dL)": "14.2",  # CBC-HGB
    "PLT-I(10^3/uL)": "250",  # CBC-RBC/PLT
    "PLT-I/M": "*",  # mark of a CBC-RBC/PLT parameter
    "NEUT#(10^3/uL)": "3.1",  # DIFF/WDF
    "RET%(%)": "0.00",  # RET
    "RET-He(pg)": "5.3",  # RET, artefact rather than zero
    "LFR(%)": "100.0",  # RET, artefact rather than zero
    "PLT-F(10^3/uL)": "0",  # PLT-F
    "IPF(%)": "0.0",  # PLT-F
    "WBC-P(10^3/uL)": "0.00",  # WPC
    "TNC(10^3/uL)": "0.00",  # unattributed in the dictionary
    "WBC(10^3/uL).1": "6.5",  # duplicated header as pandas mangles it
    "RET%(%)_1": "0.00",  # ...and as normalize_duplicate_columns spells it
}

_FULL = "CBC+DIFF+RET+PLT-F+WPC"

_RET_AND_LATER = {
    "RET%(%)",
    "RET-He(pg)",
    "LFR(%)",
    "PLT-F(10^3/uL)",
    "IPF(%)",
    "WBC-P(10^3/uL)",
}


def _frame(**overrides) -> pd.DataFrame:
    """One-row XN_SAMPLE-shaped frame, with column overrides applied."""
    row = dict(_COLUMNS)
    row.update(overrides)
    return pd.DataFrame([row])


def _masked(df: pd.DataFrame) -> set[str]:
    """Columns the mask blanked in the frame's single row."""
    out, _ = mask_unmeasured(df)
    return {c for c in out.columns if pd.isna(out.iloc[0][c])}


class TestMaskUnmeasured:
    """The core rule: a channel absent from Discrete is NaN, not zero."""

    def test_cbc_diff_masks_ret_pltf_and_wpc_columns(self):
        assert _RET_AND_LATER <= _masked(_frame())

    def test_cbc_diff_leaves_measured_channels_intact(self):
        masked = _masked(_frame())
        for column in (
            "WBC(10^3/uL)",
            "HGB(g/dL)",
            "PLT-I(10^3/uL)",
            "NEUT#(10^3/uL)",
            "Sample No.",
            "Date",
        ):
            assert column not in masked

    def test_full_order_masks_nothing(self):
        assert _masked(_frame(Discrete=_FULL)) == set()

    def test_mark_inherits_its_value_columns_attribution(self):
        # PLT-I/M is filed under PLT-F in the dictionary, but PLT-I is a
        # CBC-RBC/PLT parameter, so its marks are real under CBC+DIFF.
        assert "PLT-I/M" not in _masked(_frame())

    @pytest.mark.parametrize(
        ("duplicate", "base"),
        [("WBC(10^3/uL).1", "WBC(10^3/uL)"), ("RET%(%)_1", "RET%(%)")],
    )
    def test_duplicated_headers_follow_their_base_column(self, duplicate, base):
        masked = _masked(_frame())
        assert (duplicate in masked) == (base in masked)

    def test_unattributed_column_is_never_masked(self):
        assert "TNC(10^3/uL)" not in _masked(_frame())

    def test_compact_and_expanded_discrete_agree(self):
        assert _masked(_frame(Discrete="CDRP")) == _masked(
            _frame(Discrete="CBC+DIFF+RET+PLT-F")
        )

    def test_free_select_is_left_alone(self):
        assert parse_discrete("FREE SELECT") == []
        assert _masked(_frame(Discrete="FREE SELECT")) == set()

    def test_body_fluid_is_left_alone(self):
        assert _masked(_frame(**{"Measurement Mode": "BF"})) == set()

    def test_counts_only_cells_that_held_something(self):
        _, n_masked = mask_unmeasured(_frame())
        assert n_masked == len(_RET_AND_LATER) + 1  # + the RET%(%)_1 duplicate

    def test_missing_discrete_column_masks_nothing(self):
        df = _frame().drop(columns=["Discrete"])
        out, n_masked = mask_unmeasured(df)
        assert n_masked == 0
        assert not out.isna().to_numpy().any()

    def test_plan_is_reproducible_from_the_cleaned_frame(self):
        out, _ = mask_unmeasured(_frame())
        plan = unmeasured_by_discrete(out)
        assert _RET_AND_LATER <= set(plan["CBC+DIFF"])


class TestPipelinePlacement:
    """Properties that depend on the mask running after ``encode_flags``."""

    @staticmethod
    def _flag_frame(discrete: str) -> pd.DataFrame:
        return pd.DataFrame([{
            "Sample No.": "S1",
            "Measurement Mode": "WB",
            "Discrete": discrete,
            "RET%(%)": "0.00",
            # A RET-channel IP flag, blank as the analyser exports it.
            "IP ABN(RBC)Reticulocytosis": np.nan,
        }])

    def test_flag_of_an_unrun_channel_survives_the_zero_fill_as_nan(self, logger):
        df = self._flag_frame("CBC+DIFF")
        df = encode_flags(df, logger)
        assert df["IP ABN(RBC)Reticulocytosis"].iloc[0] == 0  # fill happened
        df = mask_unmeasured_channels(df, logger)
        assert pd.isna(df["IP ABN(RBC)Reticulocytosis"].iloc[0])

    def test_blank_flag_of_a_measured_channel_still_becomes_zero(self, logger):
        df = self._flag_frame("CBC+DIFF+RET")
        df = encode_flags(df, logger)
        df = mask_unmeasured_channels(df, logger)
        assert df["IP ABN(RBC)Reticulocytosis"].iloc[0] == 0

    def test_masking_an_int_column_widens_rather_than_raising(self, logger):
        df = self._flag_frame("CBC+DIFF")
        df = encode_flags(df, logger)
        assert pd.api.types.is_integer_dtype(df["IP ABN(RBC)Reticulocytosis"])
        df = mask_unmeasured_channels(df, logger)
        assert pd.api.types.is_float_dtype(df["IP ABN(RBC)Reticulocytosis"])


class TestNaNsDoNotRemoveRows:
    """The output invariant: a NaN means "not measured", never "drop me"."""

    def test_masking_never_drops_a_row(self, logger):
        df = pd.concat([_frame(), _frame(Discrete=_FULL)], ignore_index=True)
        assert len(mask_unmeasured_channels(df, logger)) == 2

    def test_remove_empty_columns_drops_columns_not_rows(self, logger):
        df = pd.concat([_frame(), _frame()], ignore_index=True)
        df = mask_unmeasured_channels(df, logger)
        out = remove_empty_columns(df, logger)
        assert len(out) == 2
        # Every column masked on both rows is now empty and gone.
        assert _RET_AND_LATER.isdisjoint(out.columns)
        assert "WBC(10^3/uL)" in out.columns

    def test_a_partly_masked_column_is_kept(self, logger):
        df = pd.concat([_frame(), _frame(Discrete=_FULL)], ignore_index=True)
        df = mask_unmeasured_channels(df, logger)
        out = remove_empty_columns(df, logger)
        assert "RET%(%)" in out.columns
        assert out["RET%(%)"].isna().sum() == 1
