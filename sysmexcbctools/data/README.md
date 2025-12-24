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
- Optional (for large datasets): dask[dataframe], pyarrow

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
- `use_memory_optimized` (bool, default=True): Use memory-efficient processing
- `enable_memory_monitoring` (bool, default=True): Log memory usage
- `correlation_sample_size` (int, default=50000): Max rows for correlation analysis
- `chunk_size` (int, default=1000): Chunk size for memory-optimized processing
- `force_dask` (bool, default=False): Force Dask usage for testing
- `output_dir` (str, default="./output"): Output directory
- `output_prefix` (str, default="XN_SAMPLE_processed"): Output filename prefix
- `log_to_file` (bool, default=False): Create log and diagnostic files
- `verbose` (int, default=1): Verbosity level (0=silent, 1=info, 2=debug)

**Methods:**

- `process_files(input_files, dataset_name=None, save_output=False)`: Process one or more CSV files
- `process(dataset_name)`: Process a dataset from config file

## Processing Steps

The processor performs the following steps in order:

1. **Loading and concatenation** of multiple CSV files
2. **Removal of duplicate rows**
3. **Removal of technical samples** (QC samples, calibration, etc.)
4. **Processing of discrete columns** to identify which Sysmex channels were measured
5. **Encoding of flags and indicators** as binary values
6. **Handling of multiple measurements** of the same sample
7. **Cleaning of non-numeric values** and special codes
8. **Analysis of correlations** with standard FBC features (optional)
9. **Output of processed data** with optional logging

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
    ├── constants.py             # FBC parameters and thresholds
    ├── memory_optimized.py      # Memory-efficient processing
    └── sysmex_channels_*.py     # Channel definitions
```

## Testing

Run the test suite:

```bash
# Integration tests with real data
pytest tests/data/test_integration.py -v

# All tests
pytest tests/data/ -v
```

## Notes

- The default configuration is conservative and preserves most data
- Review log files to understand what changes were made
- Use `log_to_file=False` (default) for Jupyter notebooks to avoid file clutter
- For large datasets (>100k rows), memory-optimized mode is automatically used
- The `std_threshold` parameter controls how strict multiple measurement comparison is:
  - Lower values (e.g., 0.5) = stricter, fewer measurements considered "matching"
  - Higher values (e.g., 2.0) = more lenient, more measurements considered "matching"
