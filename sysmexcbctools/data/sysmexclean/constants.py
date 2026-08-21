# Identifier columns that must never be type-inferred as numeric.  Sample
# numbers are identifiers, not quantities: a purely numeric cohort would
# otherwise be read as int/float, breaking string operations (e.g.
# ``.str.startswith``) and the ``(Sample No., datetime)`` matching keys used
# to align OutputData.csv and SCT files (a float ``123456.0`` never matches
# the string ``123456``).  Always read these as ``str``.
ID_COLUMNS = ["Sample No."]

# List of standard FBC parameters in XN_SAMPLE notation
STANDARD_FBC_DECRYPT = [
    "WBC(10^3/uL)",
    "RBC(10^6/uL)",
    "HGB(g/dL)",
    "HCT(%)",
    "PLT(10^3/uL)",
]

# Standard FBC features for correlation analysis
STANDARD_FBC_FEATURES = [
    "PLT(10^3/uL)",
    "MPV(fL)",
    "PCT(%)",
    "RBC(10^6/uL)",
    "HGB(g/dL)",
    "HCT(%)",
    "MCV(fL)",
    "MCH(pg)",
    "MCHC(g/dL)",
    "RDW-SD(fL)",
    "WBC(10^3/uL)",
    "NEUT#(10^3/uL)",
    "LYMPH#(10^3/uL)",
    "MONO#(10^3/uL)",
    "EO#(10^3/uL)",
    "BASO#(10^3/uL)",
    "NEUT%(%)",
    "LYMPH%(%)",
    "MONO%(%)",
    "EO%(%)",
    "BASO%(%)",
]

# Sysmex technical samples prefixes to exclude
SYSMEX_TECHNICAL_SAMPLE_PREFIXES = [
    "QC-",
    "BACKGROUNDCHECK",
    "CAL-",
    "CAL",
    "xncal",
    "ERR",
    "PRE-CHK-",
    "OPT-AXIS",
]

# Columns to remove from the dataset.
#
# ``Discrete`` and ``Measurement Mode`` are deliberately NOT here.  They are
# what ``mask_unmeasured_channels`` reads to decide which measurements the
# analyser never took, and it runs after ``remove_unused_columns``.  Keeping
# them also makes the cleaned file self-explanatory: ``Discrete`` says which
# channels ran, ``Measurement Mode`` says why body-fluid rows were left with
# the analyser's own fill.  See ``discrete_channels.unmeasured_by_discrete``.
TRASH_COLUMNS = [
    "Nickname",
    "Rack",
    "Position",
    "Sample Inf.",
    "Order Type",
    "Patient ID",
    "Analysis Info.",
    "Judgment",
    "Order Info.",
    "WBC Info.",
    "PLT Info.",
    "Rule Result",
    "Validate",
    "Validator",
    "Action Message (Check)",
    "Action Message (Review)",
    "Action Message (Retest)",
    "Sample Comment",
    "Patient Name",
    "Birth",
    "Sex",
    "Patient Comment",
    "Ward Name",
    "Doctor Name",
    "Output",
    "Sequence No.",
    "Unclassified()",
]

# Redundant HGB columns (different units of HGB, sometimes repeated)
REDUNDANT_HGB_COLUMNS = [
    "HGB_NONSI(g/dL)",
    "HGB_SI(mmol/L)",
    "HGB_SI2(mmol/L)",
    "HGB_NONSI2(g/dL)",
    "HGB-S(g/dL)",
]

MARKS = {
    "----": "error",
    "++++": "oor",
    "*": "low_reliability",
    "@": "oor_linearity",
    "!": "oor_panic",
}
