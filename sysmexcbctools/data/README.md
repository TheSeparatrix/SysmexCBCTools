# Sysmex Data Cleaning Module

A Python toolkit for cleaning and preprocessing Sysmex XN haematology analyzer data. This module processes `XN_SAMPLE.csv` files exported from decrypted `.116` files using Sysmex HAAS software.

Maintained by Daniel Kreuter. Parts written by Daniel Kreuter, Simon Deltadahl, Julian Gilbey, and Allerdien Visser.

## Overview

The data cleaning module provides a scikit-learn-style API for processing Sysmex XN_SAMPLE files. It handles many common challenges including:

- Multiple measurements of the same sample
- Technical and quality control samples
- Non-numeric values and special codes
- Flags and indicators encoding
- Samples with potential clots or other quality issues
- Multi-file consolidation from multiple decryptions

## Installation

```bash
# Install with data module dependencies
pip install -e ".[data]"

# Or install all dependencies
pip install -e ".[all]"
```

### Requirements

- Python 3.7+
- Required packages: pandas, numpy, pyyaml, tqdm, psutil
- Optional (for parquet input/output): pyarrow

## Quick Start

### Python API (Recommended)

```python
from sysmexcbctools.data import XNSampleProcessor

# Create processor with default settings
processor = XNSampleProcessor()

# Process single file
df = processor.process_files("XN_SAMPLE.csv", save_output=False)

# Process multiple files
df = processor.process_files(
    ["part1.csv", "part2.csv", "part3.csv"],
    save_output=False
)

# Customize processing parameters
processor = XNSampleProcessor(
    remove_clotintube=True,
    remove_multimeasurementsamples=True,
    std_threshold=1.0,
    log_to_file=False,  # No log files by default
    verbose=1
)

df = processor.process_files("XN_SAMPLE.csv")
```

### Command-Line Interface

```bash
# Process files directly
python -m sysmexcbctools.data.process_XN_SAMPLE --files data1.csv data2.csv --output-dir ./results

# Process using config file
python -m sysmexcbctools.data.process_XN_SAMPLE --config config.yaml

# Process specific dataset from config
python -m sysmexcbctools.data.process_XN_SAMPLE --config config.yaml --dataset INTERVAL
```

## Usage Examples

### Basic Processing

```python
from sysmexcbctools.data import XNSampleProcessor

processor = XNSampleProcessor()
df_clean = processor.process_files("XN_SAMPLE.csv", save_output=False)

print(f"Processed {len(df_clean)} samples")
print(f"Columns: {df_clean.shape[1]}")
```

### Multi-Dataset Consolidation

```python
# Process multiple files at once
files = [
    "batch1/XN_SAMPLE.csv",
    "batch2/XN_SAMPLE.csv",
    "batch3/XN_SAMPLE.csv"
]

processor = XNSampleProcessor()
df = processor.process_files(files, dataset_name="consolidated", save_output=False)
```

### Custom Parameters

```python
# Strict processing
processor = XNSampleProcessor(
    remove_clotintube=True,  # Remove clotted samples
    remove_multimeasurementsamples=True,  # Handle multiple measurements
    std_threshold=0.5,  # Strict threshold
    remove_correlated=False,  # Keep all features
    verbose=2
)

df = processor.process_files("XN_SAMPLE.csv", save_output=False)
```

### Config-Based Processing

Create `config.yaml`:

```yaml
output:
  directory: "./output"
  filename_prefix: "XN_SAMPLE_processed"

input:
  datasets:
    - name: "INTERVAL"
      files:
        - "/path/to/INTERVAL/XN_SAMPLE.csv"
    - name: "STRIDES"
      files:
        - "/path/to/STRIDES/batch1/XN_SAMPLE.csv"
        - "/path/to/STRIDES/batch2/XN_SAMPLE.csv"

processing:
  remove_clotintube: true
  remove_multimeasurementsamples: true
  std_threshold: 1.0
  remove_correlated: false
```

Then process:

```python
from sysmexcbctools.data import XNSampleProcessor

# Load config and process specific dataset
processor = XNSampleProcessor(config_path="config.yaml")
df = processor.process("INTERVAL")
```

### Save Output

```python
processor = XNSampleProcessor(
    output_dir="./results",
    output_prefix="cleaned_data"
)

# Save to timestamped file
df = processor.process_files(
    "XN_SAMPLE.csv",
    dataset_name="my_study",
    save_output=True
)
# Saves to: ./results/cleaned_data_my_study_YYYYMMDD_HHMMSS.csv
```

### Enable Logging and Diagnostics

```python
processor = XNSampleProcessor(
    log_to_file=True,  # Create log files
    output_dir="./output"
)

df = processor.process_files("XN_SAMPLE.csv", save_output=True)
# Creates: output/XN_SAMPLE_YYYYMMDD_HHMMSS.log
# Creates: output/*_diagnostic_*.csv (if issues found)
```

## API Reference

### XNSampleProcessor

Main class for processing Sysmex XN_SAMPLE data.

**Parameters:**

- `config_path` (str, optional): Path to YAML configuration file
- `remove_clotintube` (bool, default=True): Remove samples with clot flags
- `remove_multimeasurementsamples` (bool, default=True): Handle multiple measurements
- `std_threshold` (float, default=1.0): Threshold for comparing multiple measurements
- `remove_correlated` (bool, default=False): Remove highly correlated features
- `keep_drop_rows` (bool, default=False): Mark rows for removal without dropping
- `make_dummy_marks` (bool, default=False): Create dummy variables for marks
- `drop_empty_columns` (bool, default=True): Remove columns that are NaN for every row
- `use_memory_optimized` (bool, default=True): Use memory-efficient processing
- `enable_memory_monitoring` (bool, default=True): Log memory usage
- `output_dir` (str, default="./output"): Output directory
- `output_prefix` (str, default="XN_SAMPLE_processed"): Output filename prefix
- `log_to_file` (bool, default=False): Create log and diagnostic files
- `verbose` (int, default=1): Verbosity level (0=silent, 1=info, 2=debug)

**Methods:**

- `process_files(input_files, dataset_name=None, save_output=False)`: Process one or more CSV files
- `process(dataset_name)`: Process a dataset from config file

## Processing Steps

The processor performs the following steps in order:

1. **Loading and concatenation** of the input CSV/Parquet files
2. **Removal of duplicate rows**
3. **Removal of technical samples** (QC, calibration, background checks, error records)
4. **Normalisation of duplicated column labels** -- `XN_SAMPLE.csv` repeats around 30
   header names; the repeats are suffixed `_1`, `_2`, ... so that every step below sees a
   single column per name
5. **Removal of unused columns** (patient identifiers, free-text fields, `(Reserved)`
   and `Unnamed` blocks)
6. **Encoding of flags and indicators** as binary values, and of `Q-Flag` columns into a
   numeric value plus `_err` / `_disc` companions
7. **NULL-filling of unmeasured channels** -- every measurement the `Discrete` order
   never asked for is set to NaN. See [NaN policy](#nan-policy) below
8. **Encoding of data marks** (`/M` columns), optionally as dummies
9. **Removal of duplicate columns**
10. **Removal of clotted samples** (optional, on by default)
11. **Handling of multiple measurements** of the same sample (optional, on by default)
12. **Cleaning of non-numeric values** and special codes (`----`, blanks)
13. **Removal of redundant columns** (repeated HGB unit variants)
14. **Removal of columns that are empty for every row** (optional, on by default)
15. **Conversion to numeric**
16. **Analysis of correlations** with standard FBC features, and their removal (optional,
    off by default)

With `keep_drop_rows=True` no row is actually removed; each removal rule instead writes a
`drop_*` column, and a final `drop` column ORs them together.

## NaN policy

**Every NaN in the cleaned output is intentional, and no row is ever removed for having
one.**

### Why a zero from the analyser is not a measurement

A Sysmex XN runs only the channels the operator ordered, and the `Discrete` column records
that order. Fields belonging to a channel that never ran are *not* left blank in the
export -- they are zero-filled. A `CBC+DIFF` sample is exported with `RET%(%) = 0.00`,
`PLT-F(10^3/uL) = 0`, `IPF(%) = 0.0`, all indistinguishable downstream from a genuine
count of zero. Worse, a few fields are not zero at all but a constant artefact of
computing on zeros -- `LFR(%) = 100.0` (it is `100 - MFR - HFR`), `RET-He(pg)` and
`RBC-He(pg) = 5.3` (a floor) -- and those survive a naive "drop the zeros" filter as
perfectly plausible values.

The processor therefore replaces them with NaN. The rule composes two tables from the
Sysmex data dictionary, carried in `sysmexclean/discrete_channels.py` (the rules) and its
generated sibling `sysmexclean/_channel_table.py` (the data):

| `Discrete` token | Mechanical channels |
|---|---|
| `CBC` | CBC-RBC/PLT, CBC-HGB, CBC-WNR |
| `DIFF` | CBC-WNR, DIFF/WDF |
| `RET` | CBC-RBC/PLT, RET |
| `PLT-F` | CBC-RBC/PLT, PLT-F |
| `WPC` | CBC-WNR, DIFF/WDF, WPC |

plus `COLUMN_CHANNELS`, which attributes each of 441 `XN_SAMPLE` columns to the channels
that populate it. A column is kept when it is always-on or when any channel populating it
is active; otherwise it is set to NaN. Both the expanded (`CBC+DIFF+RET+PLT-F`) and the
compact (`CDRP`) spellings of `Discrete` are understood.

Four carve-outs, all load-bearing:

- a mark column `X/M` takes its **value** column's attribution, which also corrects the
  dictionary's one error (`PLT-I/M` is filed under PLT-F, but `PLT-I` is a CBC-RBC/PLT
  parameter whose marks are real under `CBC+DIFF`);
- the duplicated-header suffixes are stripped before lookup (`.1` as pandas mangles them,
  `_1` as step 4 above spells them), so a repeat is masked exactly like its original;
- only whole-blood rows (`Measurement Mode = WB`) are masked -- body fluids have their own
  parameter set and a different channel story;
- a column the dictionary does not attribute at all is **never** masked. Only a positive
  attribution justifies discarding a value.

Rows whose `Discrete` reads `FREE SELECT`, or names nothing recognised, are left exactly
as the analyser exported them.

### Reading the NaNs back

`Discrete` and `Measurement Mode` are deliberately **kept** in the output. They are what
makes a NaN explainable after the fact: given a cleaned file, `unmeasured_by_discrete`
reproduces exactly which cells were blanked and why.

```python
from sysmexcbctools.data.sysmexclean.discrete_channels import unmeasured_by_discrete

plan = unmeasured_by_discrete(df)   # {Discrete value: [columns not measured]}
plan["CBC+DIFF"][:3]                # ['RET%(%)', 'RET-He(pg)', 'LFR(%)']
```

A NaN in one of those columns, on a row carrying that `Discrete` value, means **"this
channel was never run"** -- not "this record is incomplete". Downstream code must keep the
row.

The other two sources of NaN are equally deliberate: the analyser's own error codes
(`----`, blanks), turned into NaN by step 12; and a data mark the `MARKS` table does not
recognise, blanked by step 8.

### What this means for filtering

- **No pipeline step removes a row because a cell is NaN.** Rows are removed only by
  explicit rules -- exact duplicates, technical-sample prefixes, clot flags, and
  duplicate-measurement consolidation. The similarity check in that last one is written
  as `~(difference > threshold)` specifically so that a NaN comparison keeps the sample.
- **The one NaN-based filter is column-wise.** Step 14 drops columns that are empty for
  every row, on the grounds that they carry no signal. Which columns those are depends on
  the channel mix of the cohort, so two runs over different cohorts can produce different
  column sets -- pass `drop_empty_columns=False` when the results have to line up.
- **Downstream `dropna()` needs care.** A plain `df.dropna()` on the cleaned file will
  discard every sample that did not order every channel. Filter on the columns the
  analysis actually needs, or subset by `Discrete` first.

### Where the masking sits, and why it cannot move

Step 7 runs immediately after the flag encoding in step 6 and before everything else.
Both halves matter. Later, and step 6's null-to-zero fill would re-fabricate "flag not
raised" for a channel that never ran. Earlier, and every step below it -- clot removal,
the missingness report, the correlation analysis -- would compute on the analyser's zero
fill; a RET column that is exactly `0.00` on every `CBC+DIFF` row correlates spuriously
with anything separating those rows.

`OutputData.csv` is **not** masked: its ~3,780 columns use a naming convention the Sysmex
dictionary does not cover.

## Output Files

The processor can generate:

1. **Processed CSV file**: Cleaned and consolidated data
2. **Log files**: Detailed processing information (if `log_to_file=True`)
3. **Diagnostic files**: Samples with multiple inconsistent measurements for clinical review
4. **Correlation analysis**: Relationships between columns and core FBC features

## Example Notebooks

See the `examples/notebooks/` directory for Jupyter notebook tutorials:

- `04_data_basic_cleaning.ipynb` - Basic usage and single-file processing
- `05_data_multi_dataset.ipynb` - Multi-file consolidation and batch processing
- `06_data_advanced_config.ipynb` - Advanced parameters and optimization

## File Structure

```
sysmexcbctools/data/
├── __init__.py                  # Module exports
├── process_XN_SAMPLE.py         # CLI script
├── config.yaml                  # Example configuration
├── README.md                    # This file
└── sysmexclean/                 # Core package
    ├── __init__.py
    ├── processor_api.py         # XNSampleProcessor class
    ├── processors.py            # Data transformation functions
    ├── utils.py                 # I/O and logging utilities
    ├── constants.py             # FBC parameters and column lists
    ├── memory_optimized.py      # Memory-efficient processing
    ├── ancillary.py             # OutputData.csv / SCT matching and copying
    ├── discrete_channels.py     # Which channel populates which column (rules)
    └── _channel_table.py        # ...and the generated data behind it
```

## Testing

Run the test suite:

```bash
# All data-module tests
pytest tests/data/ -v

# Just the Discrete / NaN-policy rules
pytest tests/data/test_discrete_channels.py -v
```

## Notes

- The default configuration is conservative and preserves most data
- Review log files to understand what changes were made
- Use `log_to_file=False` (default) for Jupyter notebooks to avoid file clutter
- For large datasets (>100k rows), memory-optimized mode is automatically used
- The `std_threshold` parameter controls how strict multiple measurement comparison is:
  - Lower values (e.g., 0.5) = stricter, fewer measurements considered "matching"
  - Higher values (e.g., 2.0) = more lenient, more measurements considered "matching"
