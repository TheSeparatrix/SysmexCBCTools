# Sysmex Transfer Module

Tools for aligning raw Sysmex haematology analyzer data between different machines using various transformation methods including Gaussian Mixture Model with Optimal Transport (GMM-OT) and MAD/median-based alignment.

## Overview

This module provides cross-analyzer alignment for three types of Sysmex data:

1. **Flow Cytometry Channels** (RET, WDF, WNR, PLTF) - GMM-OT transformation
2. **Impedance Channels** (RBC, PLT) - GMM-OT for histogram data
3. **XN_SAMPLE Tabular Data** - MAD/median-based alignment for 23 FBC parameters

## Installation

```bash
# Install with transfer module dependencies
pip install -e ".[transfer]"

# Or install all dependencies
pip install -e ".[all]"
```

## Quick Start

### Python API (Recommended for Jupyter Notebooks)

```python
from sysmexcbctools import FlowTransformer, ImpedanceTransformer, XNSampleTransformer

# Flow cytometry alignment
transformer = FlowTransformer(channel='RET')
transformer.fit(source_files, target_files)
transformed_files = transformer.transform(source_files, output_dir='output/')
transformer.save('models/ret_transformer.pkl')

# Impedance alignment
imp_transformer = ImpedanceTransformer()
imp_transformer.fit(source_df, target_df)
transformed_df = imp_transformer.transform(source_df)

# XN_SAMPLE alignment
xn_transformer = XNSampleTransformer(columns=['HGB', 'RBC', 'WBC'])
xn_transformer.fit(source_df, target_df)
transformed_df = xn_transformer.transform(source_df)
```

### Command Line Interface (CLI)

Edit `config/data_paths.yaml` to point to your dataset locations, then:

**Flow cytometry transformation** (GMM + Optimal Transport):
```bash
# Compute transformation models
python compute_transformation.py \
  --source-dataset strides_merged \
  --target-dataset interval_36 \
  --channel RET \
  --source-samples strides \
  --target-samples interval_baseline_36

# Apply transformations
python transform.py \
  --source-dataset strides_merged \
  --output-dataset strides_transformed_to_interval36 \
  --channel RET \
  --transformation strides_to_interval36 \
  --samples strides
```

**Impedance data transformation** (MAD/median-based):
```bash
python transform_impedance.py \
  --source-dataset strides \
  --target-dataset interval_36 \
  --output-dataset strides_impedance_transformed \
  --source-samples strides \
  --target-samples interval_baseline_36 \
  --n_jobs 30
```

**XN_SAMPLE tabular data transformation**:
```bash
python transform_xnsample.py \
  --source-dataset strides_xn_sample \
  --target-dataset interval_36_xn_sample \
  --output-dataset strides_xn_sample_transformed \
  --source-samples strides \
  --target-samples interval_baseline_36
```

## API Documentation

### FlowTransformer

Transform flow cytometry data (RET, WDF, WNR, PLTF) using GMM-OT.

**Parameters:**
- `channel` (str): Flow cytometry channel ('RET', 'WDF', 'WNR', 'PLTF')
- `n_components` (int, default=10): Number of Gaussian components for GMM
- `transport_method` (str, default='rand'): Transport method ('weight', 'rand', 'max')
- `max_samples` (int, default=1000000): Max samples for GMM fitting
- `preserve_rare` (bool, default=True): Preserve rare cell populations
- `use_gate_init` (bool, default=True): Use gate-informed GMM initialisation
- `use_cascade_init` (bool, default=False): Initialise target GMM from fitted source GMM parameters
- `n_jobs` (int, default=-1): Number of parallel jobs
- `random_state` (int, optional): Random seed for reproducibility

**Methods:**
- `fit(source_files, target_files)` or `fit(source_data, target_data)`: Fit GMM and compute transport map
- `transform(source_files, output_dir)`: Transform files and save
- `transform_array(source_data)`: Transform numpy array directly
- `save(filepath)`: Save fitted transformer
- `load(filepath)`: Load saved transformer (class method)

**Attributes:**
- `source_gmm_`: Fitted GMM for source distribution
- `target_gmm_`: Fitted GMM for target distribution
- `transport_dict_`: Transport maps and related information
- `is_fitted_`: Whether transformer has been fitted

**Example:**
```python
from sysmexcbctools import FlowTransformer

# Fit and transform
transformer = FlowTransformer(channel='RET', n_components=10)
transformer.fit(source_files=['path/to/RET*.116.csv'],
                target_files=['path/to/RET*.116.csv'])
transformer.transform(source_files=['path/to/RET*.116.csv'],
                     output_dir='output/')

# Save and reuse
transformer.save('models/ret_transformer.pkl')
loaded = FlowTransformer.load('models/ret_transformer.pkl')
```

See `examples/notebooks/07_transfer_flow_cytometry.ipynb` for comprehensive examples.

### ImpedanceTransformer

Transform impedance histogram data (RBC_RAW_000-127, PLT_RAW_000-127) using GMM-OT.

**Key Features:**
- Handles both RBC (0-250 fL, ~1.95 fL bin width) and PLT (0-40 fL, ~0.31 fL bin width) histograms
- Probabilistic rounding preserves distribution shape
- Increased gmm_sample_size (50,000) prevents bin loss

**Parameters:**
- `n_components_rbc` (int, default=6): GMM components for RBC
- `n_components_plt` (int, default=6): GMM components for PLT
- `gmm_sample_size` (int, default=50000): Samples for GMM fitting
- `random_state` (int, default=42): Random seed
- `n_jobs` (int, default=-1): Number of parallel jobs

**Methods:**
- `fit(source_df, target_df)`: Fit GMMs and compute transport maps
- `transform(source_df)`: Transform impedance DataFrame
- `save(filepath)`: Save fitted transformer
- `load(filepath)`: Load saved transformer (class method)

**Example:**
```python
from sysmexcbctools import ImpedanceTransformer
import pandas as pd

# Load impedance data
source_df = pd.read_csv('source_OutputData.csv')
target_df = pd.read_csv('target_OutputData.csv')

# Fit and transform
transformer = ImpedanceTransformer(gmm_sample_size=50000)
transformer.fit(source_df, target_df)
transformed_df = transformer.transform(source_df)
```

See `examples/notebooks/08_transfer_impedance.ipynb` for detailed examples and quality validation.

### XNSampleTransformer

Transform XN_SAMPLE tabular data using MAD/median-based alignment.

**Parameters:**
- `columns` (list, optional): Columns to transform (default: all 23 FBC parameters)
- `skip_columns` (list, optional): Columns to exclude from transformation

**Methods:**
- `fit(source_df, target_df)`: Learn transformation parameters (MAD, median)
- `transform(source_df)`: Transform DataFrame
- `save(filepath)`: Save fitted transformer
- `load(filepath)`: Load saved transformer (class method)

**Example:**
```python
from sysmexcbctools import XNSampleTransformer

# Transform specific columns
transformer = XNSampleTransformer(columns=['HGB', 'RBC', 'WBC', 'PLT'])
transformer.fit(source_df, target_df)
transformed_df = transformer.transform(source_df)

# Transform all FBC parameters (default)
transformer = XNSampleTransformer()
transformer.fit(source_df, target_df)
transformed_df = transformer.transform(source_df)
```

See `examples/notebooks/09_transfer_xnsample.ipynb` for comprehensive examples.

## Example Notebooks

Complete Jupyter notebook examples are available in `examples/notebooks/`:

1. **`07_transfer_flow_cytometry.ipynb`**: Flow cytometry alignment (RET, WDF, WNR, PLTF)
   - GMM-OT transformation with gate-informed and cascade initialisation
   - Comprehensive quality metrics and visualisations
   - Tested on HPC with real multi-analyser data

2. **`08_transfer_impedance.ipynb`**: Impedance histogram alignment (RBC, PLT)
   - GMM-OT for histogram data with quality enhancements
   - Wasserstein distance, JS divergence, correlation metrics
   - GMM fit visualization for debugging

3. **`09_transfer_xnsample.ipynb`**: XN_SAMPLE tabular data alignment
   - MAD/median-based transformation for 23 FBC parameters
   - Distribution histograms, Q-Q plots, correlation matrices
   - Save/load functionality demonstration

4. **`10_transfer_config_workflow.ipynb`**: Configuration-based workflow
   - End-to-end pipeline for all three data types
   - Batch processing multiple channels
   - Production workflow best practices

## Data Types

### Flow Cytometry Channels
- **Format**: `.116.csv` files
- **Channels**: RET, WDF, WNR, PLTF
- **Dimensions**: 2D scatter plots (Side Fluorescence Light, Forward Scatter)
- **Filename format**: `{CHANNEL}_{instrument}{sample_id}{timestamp}{barcode}.116.csv`

### Impedance Channels
- **Format**: `OutputData.csv` files
- **Channels**:
  - `RBC_RAW_000` to `RBC_RAW_127`: 128 bins, 0-250 fL range (~1.95 fL bin width)
  - `PLT_RAW_000` to `PLT_RAW_127`: 128 bins, 0-40 fL range (~0.31 fL bin width, higher resolution)
- **Note**: Bin indices (000-127) represent different femtoliter ranges for RBC vs PLT

### XN_SAMPLE Tabular Data
- **Format**: `XN_SAMPLE.csv` files
- **Columns**: 23 full blood count (FBC) parameters
  - Red blood cells: RBC, HGB, HCT, MCV, MCH, MCHC
  - White blood cells: WBC, NEUT#, LYMPH#, MONO#, EO#, BASO#
  - Platelets: PLT, MPV, PCT
  - And additional derived parameters

## Testing & Validation

Run the complete test suite:
```bash
# Run all tests (config-based + legacy interfaces)
python tests/run_all_tests.py

# Prepare test data first (requires RDS access)
python tests/run_all_tests.py --prepare-data

# Run only config-based tests
python tests/run_all_tests.py --config-only
```

Test suite validates:
- Flow cytometry transformation (GMM-OT)
- Impedance transformation (GMM-OT)
- XN_SAMPLE transformation (MAD/median)
- Both config-based and legacy interfaces
- Model save/load functionality

See `tests/transfer/README.md` for detailed testing information.

## Recent Improvements (2025)

### ImpedanceTransformer Quality Enhancements

**Problem**: Original implementation had severe quality issues due to aggressive downsampling:
- With `gmm_sample_size=1000`, **44.5% of RBC bins and 23.4% of PLT bins were completely lost**
- Naive `int()` rounding caused bins with low counts to disappear

**Solutions**:
1. **Increased gmm_sample_size** from 1,000 → 50,000 (default)
   - Provides ~390 samples per bin for excellent statistical representation
   - Prevents catastrophic bin loss while remaining computationally feasible

2. **Probabilistic Rounding** in `sample_impedance_array()`
   - Replaces naive `int()` with probabilistic rounding
   - Preserves expected values and distribution shape
   - Uses fixed random seed for reproducibility

3. **API Fix**: Fit-once-transform-many pattern
   - GMMs fitted once in `fit()`, not every `transform()`
   - Enables saving/loading transformers for reuse

**Impact**: Dramatically improved transformation quality. See notebook 08 for GMM fit visualization and quality metrics.

**Known Limitations**:
- PLT distributions have asymmetric shape (Gaussian left, sharp cutoff right)
- GMMs assume Gaussian components, cannot perfectly capture asymmetry
- Future work: Consider truncated Gaussians or alternative distributions

### Overflow File Handling

**Problem**: Files ending in `.116(1).csv`, `.116(2).csv` caused parsing errors (overflow files when base file reaches 65,535 row limit).

**Solution**:
- Updated `parse_sysmex_raw_filename()` to handle overflow pattern
- Added `get_overflow_files()` to find and merge overflow files
- Added `merge_csv_with_overflows()` to merge base + overflow before preprocessing
- API automatically filters overflow files in `transform()` to prevent duplicates

## Configuration System

### Configuration-Based Interface (Recommended)

Uses `config/data_paths.yaml` for centralized path management:

**Advantages:**
- Simple commands: `--source-dataset strides` instead of `/rds/project/.../STRIDES/...`
- Centralized path management: Change RDS structure in one config file
- Environment flexibility: Easy switching between development/production data
- Error prevention: Validated paths and datasets before processing
- Professional tool interface: Clean, intuitive command structure

**Configuration File Structure:**
```yaml
base_paths:
  rds_base: "/rds/project/rds-ccRCdkyWMsY/ref"

datasets:
  raw:
    strides_merged: "${base_paths.rds_base}/STRIDES/flow_cytometry"
    interval_36: "${base_paths.rds_base}/INTERVAL/36/flow_cytometry"

  processed:
    strides: "${base_paths.rds_base}/STRIDES/impedance"

files:
  centile_samples:
    strides: "${base_paths.rds_base}/samples/strides_samples.npy"
    interval_baseline_36: "${base_paths.rds_base}/samples/interval_36_samples.npy"
```

### Legacy Interface (Backward Compatibility)

Original path-based commands are still supported:
```bash
python compute_transformation.py <source_dir> <target_dir> <output_dir> \
  "<source_name>" "<target_name>" "<channel>" \
  --sample_nos_source=<source.npy> --sample_nos_target=<target.npy>
```

## Architecture

```
sysmexcbctools/transfer/
├── sysmexalign/               # Main package
│   ├── __init__.py            # Public API exports
│   ├── api.py                 # FlowTransformer, ImpedanceTransformer, XNSampleTransformer
│   ├── gmm_ot.py              # GMM-OT core implementation
│   ├── alignment_1d.py        # MAD/median alignment for 1D data
│   ├── load_and_preprocess.py # Data loading and preprocessing
│   ├── gate_utils.py          # Flow cytometry gating utilities
│   └── flow_gating_pipeline.py # Gating system implementation
├── config/                    # Configuration management
│   ├── config_loader.py       # YAML loader with variable substitution
│   └── data_paths.yaml        # Dataset path configurations
├── utils/                     # Utility functions
│   └── data_loader.py         # Config-aware data loading
├── tests/                     # Test suite
│   ├── run_all_tests.py       # Master test runner
│   └── scripts/               # Individual test scripts
├── transformations/           # Pre-computed transformation models
└── README.md                  # This file
```

## Future Enhancements

- **FCS file support**: Add native FCS format support for flow cytometry (currently uses CSV only)
- **Advanced distributions**: Investigate truncated Gaussians or alternative distributions for PLT asymmetry
- **Automated gate generation**: Replace manual gates with Sysmex official gates or expert-validated gates

## Contributing

This module follows scientific Python best practices:
- NumPy-style docstrings for all public functions
- Scikit-learn API conventions (fit/transform pattern)
- Comprehensive test coverage
- Reproducible results with fixed random seeds
- Fail-loudly error handling with descriptive messages

## Related Modules

- **`sysmexcbctools.data`**: XN_SAMPLE data cleaning and preprocessing
- **`sysmexcbctools.correction`**: GAM-based covariate correction
- **`sysmexcbctools.disae2`**: Domain-invariant feature learning
