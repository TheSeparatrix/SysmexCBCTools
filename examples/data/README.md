# Test Dataset: data_B.csv

## Overview

`data_B.csv` is a synthetic test dataset designed for demonstrating and testing the SysmexCBCTools modules, particularly the Dis-AE 2 (Domain-Invariant Autoencoder) module.

## Dataset Structure

- **Total samples**: 50,000
- **Total columns**: 38

### Feature Columns (32 columns)

Columns `0` through `31`: Numeric feature values representing CBC-like measurements.

- **Example ranges**:
  - Column `0`: [9.77, 40.68]
  - Column `1`: [9.16, 39.28]
  - Column `2`: [10.31, 39.72]
  - Column `3`: [10.07, 23.54]
  - Column `4`: [9.98, 23.99]

### Task Label (1 column)

- **`ClassCategory_0`**: Binary classification task (0 or 1)
  - Class 0: 25,000 samples
  - Class 1: 25,000 samples
  - Perfectly balanced dataset

### Domain Factors (3 columns)

These columns represent different domain factors that can introduce spurious correlations or domain shift:

1. **`Machine`**: Instrument/analyzer identifier (5 machines)
   - Values: 0.0, 1.0, 2.0, 3.0, 4.0
   - Distribution: ~10,000 samples per machine (roughly balanced)
   - Machine 0: 9,895 samples
   - Machine 1: 10,058 samples
   - Machine 2: 10,069 samples
   - Machine 3: 9,978 samples
   - Machine 4: 10,000 samples

2. **`vendelay_binned`**: Venepuncture delay binned (10 bins)
   - Values: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
   - Distribution: ~5,000 samples per bin (roughly balanced)
   - Represents pre-analytical variation due to time between blood draw and analysis

3. **`studytime_binned`**: Time into study binned (10 bins)
   - Values: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
   - Distribution: ~5,000 samples per bin (roughly balanced)
   - Represents temporal drift or batch effects over the study duration

### Continuous Covariates (2 columns)

These are the unbinned continuous versions of some domain factors:

- **`TimeIntoStudy`**: Continuous time into study
  - Range: [0.02, 499.99]
  - Unit: likely days or hours

- **`VenepunctureDelay`**: Continuous venepuncture delay
  - Range: [0.00, 36.00]
  - Unit: likely hours

## Use Cases

### 1. Dis-AE 2 Module (Domain-Invariant Learning)

Train a domain-invariant autoencoder to learn features that are:
- Predictive of `ClassCategory_0` (task-relevant)
- Invariant to `Machine`, `vendelay_binned`, and `studytime_binned` (domain-invariant)

```python
from sysmexcbctools.disae2 import DisAE

# Features are columns 0-31
X = df.iloc[:, 0:32].values

# Task label (binary classification)
y_task = df['ClassCategory_0'].values

# Domain factors
y_domains = df[['Machine', 'vendelay_binned', 'studytime_binned']].values

model = DisAE(
    input_dim=32,
    num_tasks=[2],  # Binary task
    num_domains=[5, 10, 10],  # 5 machines, 10 vendelay bins, 10 studytime bins
)

model.fit(X, [y_task], y_domains)
```

### 2. Correction Module (GAM-Based Covariate Correction)

Correct for spurious covariates using GAM models:

```python
from sysmexcbctools.correction import GAMCorrector

# Use continuous covariates for correction
corrector = GAMCorrector(
    covariates=['TimeIntoStudy', 'VenepunctureDelay']
)

feature_cols = [str(i) for i in range(32)]
covariate_cols = ['TimeIntoStudy', 'VenepunctureDelay']

corrector.fit(df, feature_cols, covariate_cols)
df_corrected = corrector.transform(df)
```

### 3. Transfer Module (Cross-Analyzer Alignment)

Simulate cross-analyzer transformation using different machines:

```python
from sysmexcbctools.transfer import XNSampleTransformer

# Split by machine
source_data = df[df['Machine'] == 0.0]
target_data = df[df['Machine'] == 1.0]

transformer = XNSampleTransformer()
transformer.fit(source_data.iloc[:, 0:32], target_data.iloc[:, 0:32])
transformed = transformer.transform(source_data.iloc[:, 0:32])
```

## Data Characteristics

- **Balanced**: All classes and domains are roughly balanced
- **Synthetic**: Generated data, no patient information
- **Safe for version control**: Explicitly allowed in .gitignore
- **Multi-domain**: Contains multiple domain factors for comprehensive testing
- **Continuous + Discrete**: Both continuous covariates and binned versions available

## Notes

- This dataset is synthetic and does not contain any patient or sensitive information
- It is safe to commit to version control (exception added to .gitignore)
- The dataset is designed to test domain-invariant learning, covariate correction, and cross-domain transfer
- Feature columns (0-31) are numeric values simulating CBC measurements
