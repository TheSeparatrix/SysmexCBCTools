# Correction Module - GAM-based Covariate Correction

**Purpose**: Remove spurious covariate effects from Sysmex CBC data using Generalized Additive Models (GAMs).

This module provides tools to correct for domain shift caused by technical covariates such as:
- Sample age (time from venepuncture to analysis)
- Time-of-day effects (circadian patterns)
- Day-of-week effects (weekday vs. weekend)
- Seasonal patterns (time into study)
- Batch effects (different analyzers/machines)

## Key Features

- **Scikit-learn-style API**: Simple `fit()`/`transform()` interface
- **Multiple covariate correction**: Simultaneously correct for multiple sources of variation
- **Group-specific models**: Fit separate GAMs for different subpopulations (e.g., per machine)
- **Group normalization**: Remove systematic offsets between groups after correction
- **Advanced GAM terms**: Support for tensor products (interactions), factor terms (categorical), and cyclic splines
- **Automatic transformations**: Auto-detect and handle percentages, log-scale features
- **Model persistence**: Save/load fitted models for reuse
- **Parallel processing**: Speed up fitting across many features
- **Comprehensive diagnostics**: Partial dependence plots, before/after comparisons

## Installation

```bash
# Install with correction module dependencies
pip install -e ".[correction]"

# Or install all dependencies
pip install -e ".[all]"
```

**Dependencies**: numpy, pandas, scipy, pygam, scikit-learn, matplotlib, seaborn, joblib, tqdm

## Quick Start

```python
from sysmexcbctools import GAMCorrector
import pandas as pd

# Load your data
df = pd.read_csv('xn_sample_data.csv')

# Initialize corrector
corrector = GAMCorrector(
    covariates=['VenepunctureDelay', 'TimeIntoStudy'],
    feature_columns=['WBC', 'RBC', 'HGB', 'PLT'],  # Or None for all numeric columns
    n_splines=25
)

# Fit and transform
df_corrected = corrector.fit_transform(df)

# Save model for later use
corrector.save('gam_corrector.pkl')

# Load and apply to new data
corrector = GAMCorrector.load('gam_corrector.pkl')
df_new_corrected = corrector.transform(df_new)
```

## Core Concepts

### What is Covariate Correction?

Technical covariates (like sample age or time-of-day) can create **spurious correlations** that:
- Reduce model generalization across domains
- Confound biological signals with technical artifacts
- Bias downstream predictions

**Solution**: Fit GAMs to model the covariate effects, then "middle them out" by subtracting the learned effects and adding back reference means.

### How GAMs Work

Generalized Additive Models extend linear regression with **smooth non-linear functions**:

```
y = β₀ + f₁(x₁) + f₂(x₂) + ... + ε
```

Where each `fᵢ(xᵢ)` is a smooth spline function. This allows capturing:
- Non-linear relationships (e.g., exponential decay of sample age effects)
- Interactions via tensor products
- Categorical effects via factor terms

### Correction Workflow

1. **Fit GAM**: Learn smooth covariate effect `f(covariate)` for each feature
2. **Calculate reference**: Determine reference mean (e.g., at sample age = 0 hours)
3. **Transform**: `corrected = original - f(covariate) + reference_mean`
4. **Result**: Data normalized to reference condition, covariate effects removed

## API Reference

### GAMCorrector Class

```python
GAMCorrector(
    covariates,                      # List of covariate column names
    feature_columns=None,            # Features to correct (None = all numeric)
    group_column=None,               # Column for group-specific GAMs
    normalize_groups=False,          # Remove group offsets after correction
    reference_group=None,            # Reference group for normalization
    transformation='none',           # 'log', 'logit', or 'none'
    auto_detect_percentages=True,    # Auto-apply logit to percentages
    n_splines=25,                    # Number of splines (int or dict)
    term_spec=None,                  # Advanced: custom term specifications
    centralise_threshold=None,       # MAD threshold for outlier filtering
    reference_condition=None,        # Dict of conditions for reference mean
    parallel=False,                  # Parallel processing
    n_jobs=-1,                       # Number of parallel jobs
    verbose=True                     # Print progress
)
```

### Methods

#### `fit(df)`
Fit GAMs to the provided DataFrame.

**Parameters:**
- `df` (pd.DataFrame): Training data with covariates and features

**Returns:** self

---

#### `transform(df)`
Apply fitted GAMs to correct covariate effects.

**Parameters:**
- `df` (pd.DataFrame): Data to correct

**Returns:** pd.DataFrame with corrected features

---

#### `fit_transform(df)`
Fit GAMs and transform in one step.

**Parameters:**
- `df` (pd.DataFrame): Data to fit and correct

**Returns:** pd.DataFrame with corrected features

---

#### `save(path)`
Save fitted model to disk using pickle.

**Parameters:**
- `path` (str): Output path (e.g., 'gam_model.pkl')

**Note**: Lambda functions in `reference_condition` cannot be pickled and will be stored as `None`.

---

#### `load(path)` [classmethod]
Load fitted model from disk.

**Parameters:**
- `path` (str): Path to saved model

**Returns:** GAMCorrector instance

---

#### `partial_dependence(df, covariate, feature, n_points=100, group=None)`
Calculate partial dependence of a feature on a covariate.

**Parameters:**
- `df` (pd.DataFrame): Data with covariates
- `covariate` (str): Covariate name
- `feature` (str): Feature name
- `n_points` (int): Number of evaluation points
- `group` (str, optional): Group identifier

**Returns:** Tuple of (covariate_values, predictions)

## Examples

Four comprehensive Jupyter notebooks demonstrate all functionality:

### 1. Single Covariate Correction
**Notebook**: `examples/notebooks/11_correction_single_covariate.ipynb`

Learn the basics:
- Correct for a single covariate (VenepunctureDelay)
- Visualize partial dependence plots
- Compare before/after distributions and correlations
- Evaluate correction quality

```python
corrector = GAMCorrector(
    covariates=['VenepunctureDelay'],
    n_splines=25
)
df_corrected = corrector.fit_transform(df)
```

### 2. Multi-Covariate and Grouped Correction
**Notebook**: `examples/notebooks/12_correction_multi_covariate.ipynb`

Compare five correction strategies:
- Single covariate (baseline)
- Multiple independent covariates
- Group-specific GAMs (per machine)
- Grouped + normalization (remove machine offsets)
- Downstream task evaluation with logistic regression

```python
# Strategy 4: Group-specific with normalization
corrector = GAMCorrector(
    covariates=['VenepunctureDelay', 'TimeIntoStudy'],
    group_column='Machine',
    normalize_groups=True,
    n_splines=25
)
```

### 3. Advanced GAM Usage
**Notebook**: `examples/notebooks/13_correction_advanced.ipynb`

Master advanced features:
- Custom spline specifications per covariate
- Reference conditions (correct to specific sub-populations)
- Outlier filtering with `centralise_threshold`
- Automatic transformations (log, logit)
- Model diagnostics and validation
- Best practices summary

```python
# Custom splines and reference condition
corrector = GAMCorrector(
    covariates=['VenepunctureDelay', 'TimeIntoStudy'],
    n_splines={'VenepunctureDelay': 50, 'TimeIntoStudy': 30},
    reference_condition={'VenepunctureDelay': lambda x: x <= 2},
    centralise_threshold=3.0
)
```

### 4. Tensor Products and Factor Terms
**Notebook**: `examples/notebooks/14_correction_tensor_factor_terms.ipynb`

Use advanced GAM term types:
- **Tensor products** (`'te'`): Model interactions between continuous covariates
- **Factor terms** (`'f'`): Handle categorical variables
- **Cyclic splines** (`basis='cp'`): For periodic covariates (time-of-day)

```python
# Interaction between delay and time-of-day + categorical weekday
corrector = GAMCorrector(
    covariates=['VenepunctureDelay', 'TimeOfDay', 'Weekday'],
    term_spec={
        ('VenepunctureDelay', 'TimeOfDay'): {'type': 'te', 'n_splines': 30},
        'TimeOfDay': {'type': 's', 'n_splines': 25, 'basis': 'cp'},
        'Weekday': {'type': 'f'}
    }
)
```

## Advanced Features

### Group Normalization

When correcting group-specific effects (e.g., different machines), you may want to:
1. Remove within-group covariate effects (group-specific GAMs)
2. Normalize groups to each other (remove systematic group offsets)

```python
corrector = GAMCorrector(
    covariates=['VenepunctureDelay'],
    group_column='Machine',
    normalize_groups=True,        # Two-step correction
    reference_group='Machine_A'   # Optional: normalize to specific group
)
```

**How it works:**
1. Fit separate GAMs per group
2. Apply group-specific corrections
3. Calculate mean difference between groups
4. Shift all groups to reference (or global mean)

### Custom Term Specifications

The `term_spec` parameter provides full control over GAM structure:

```python
term_spec = {
    # Smooth term with custom splines and basis
    'time': {'type': 's', 'n_splines': 50, 'basis': 'ps'},

    # Tensor product for interactions
    ('time', 'age'): {'type': 'te', 'n_splines': 30},

    # Factor term for categorical variables
    'weekday': {'type': 'f'},

    # Cyclic spline for periodic variables
    'time_of_day': {'type': 's', 'n_splines': 24, 'basis': 'cp'}
}
```

**When to use:**
- **Tensor products**: When covariate effects depend on each other (e.g., age effects vary by time-of-day)
- **Factor terms**: For categorical covariates (weekday, site, batch)
- **Cyclic splines**: For periodic covariates (hour of day, month of year)

### Reference Conditions

Correct to a specific sub-population rather than overall mean:

```python
# Correct all samples to "fresh blood" condition (≤2 hours old)
corrector = GAMCorrector(
    covariates=['VenepunctureDelay'],
    reference_condition={'VenepunctureDelay': lambda x: x <= 2}
)
```

Useful when:
- One condition is "ground truth" (e.g., fresh samples)
- You want to normalize to a specific population
- Covariate range varies across datasets

### Automatic Transformations

Features with non-linear distributions benefit from transformations:

```python
corrector = GAMCorrector(
    covariates=['VenepunctureDelay'],
    transformation='log',              # Apply log to all features
    auto_detect_percentages=True       # Auto-apply logit to PCT columns
)
```

**Guidelines:**
- **Log**: For positive continuous features (counts, volumes)
- **Logit**: For proportions/percentages in [0, 1]
- **None**: For features already on reasonable scale

### Parallel Processing

Speed up fitting for many features:

```python
corrector = GAMCorrector(
    covariates=['VenepunctureDelay'],
    parallel=True,
    n_jobs=-1  # Use all available CPUs
)
```

Especially useful when:
- Correcting 100+ features
- Using group-specific models (multiplies # of GAMs)
- Working with large datasets

## Testing

Comprehensive test suite with 90 tests covering all functionality:

```bash
# Run all tests
cd tests/correction
python run_tests.py

# Run specific test modules
pytest test_utils.py -v                  # Utility functions (18 tests)
pytest test_gam_correction.py -v         # Core API (48 tests)
pytest test_advanced_features.py -v      # Advanced features (15 tests)
pytest test_scipy_compatibility.py -v    # Scipy >= 1.12 compatibility (9 tests)
```

**Test coverage:**
- ✅ Utility functions (mad, centralise, tqdm_joblib)
- ✅ GAMCorrector initialization and validation
- ✅ Fit and transform methods
- ✅ Transformations (log, logit, auto-detect)
- ✅ Group-specific and normalized correction
- ✅ Save/load functionality
- ✅ Partial dependence calculations
- ✅ Advanced term types (tensor, factor, cyclic)
- ✅ Reference conditions and outlier filtering
- ✅ Parallel processing
- ✅ Edge cases and error handling
- ✅ Scipy compatibility

## Best Practices

### 1. Start Simple
Begin with single covariate correction, visualize effects, then add complexity.

### 2. Check Partial Dependence
Always visualize partial dependence plots to verify GAM learns sensible relationships.

### 3. Validate with Downstream Tasks
Compare model performance (AUC, accuracy) before/after correction on held-out data.

### 4. Use Appropriate Transformations
- Log-transform count data (WBC, RBC, PLT)
- Logit-transform percentages (NEUT%, LYMPH%)
- Check distributions after transformation

### 5. Choose Spline Count Carefully
- Too few splines: Under-fit, miss non-linear effects
- Too many splines: Over-fit, remove biological signal
- Typical range: 15-50 splines depending on sample size

### 6. Group Normalization vs. Simple Grouping
- **Simple grouping**: Preserves differences between groups (e.g., machine types)
- **Normalization**: Removes group offsets (e.g., align machines to each other)
- Choose based on whether group differences are real or artifact

### 7. Reference Conditions
Use reference conditions when:
- One condition is "ground truth"
- Covariate distribution differs between train/test
- You want to simulate specific experimental conditions

### 8. Monitor Correction Strength
After correction, covariates should have minimal correlation with features. If strong correlations remain:
- Increase n_splines
- Check for outliers (use centralise_threshold)
- Consider tensor products for interactions

## Troubleshooting

### Issue: "Sparse matrix `.A` attribute not found"
**Solution**: This is automatically handled by the scipy compatibility monkey-patch in `gam_correction.py`. Ensure you're using scipy >= 1.12 with the latest version of this module.

### Issue: "Cannot pickle lambda function"
**Solution**: If using `reference_condition` with lambda functions, the lambda will be dropped during `save()`. Use named functions instead if persistence is needed.

### Issue: GAM fitting fails or produces strange results
**Causes:**
- Outliers: Use `centralise_threshold=3.0` to filter
- Too few samples: Need at least 100+ samples per GAM
- Inappropriate transformation: Check feature distributions
- Covariate range issues: Ensure covariates vary sufficiently

### Issue: Correction removes too much signal
**Diagnosis:** Over-correction can happen if:
- n_splines too high (overfitting)
- Covariates are confounded with biological signal
- Groups overlap too much with true classes

**Solution:**
- Reduce n_splines
- Validate on held-out data
- Compare multiple correction strategies (see Notebook 12)

## Related Modules

This module is part of the **SysmexCBCTools** suite:
- **Data cleaning** (`sysmexcbctools.data`): Preprocess XN_SAMPLE.csv files
- **Transfer learning** (`sysmexcbctools.transfer`): Align data across analyzers
- **Domain-invariant learning** (`sysmexcbctools.disae2`): Learn features robust to domain shift

Typical workflow:
1. Clean data → **Data module**
2. Align analyzers → **Transfer module** (if multi-site)
3. Correct covariates → **Correction module** ← *You are here*
4. Learn representations → **Dis-AE 2 module**

## Future Development

Planned improvements:
- [ ] Automatic spline selection (cross-validation)
- [ ] Support for non-Gaussian likelihoods (Poisson, Binomial)
- [ ] Confidence intervals for partial dependence
- [ ] Integration with transfer module for joint correction + alignment
