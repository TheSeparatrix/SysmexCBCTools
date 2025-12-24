# Dis-AE 2: Domain Separation Network Multi-Domain Adversarial Autoencoder

A scikit-learn style library for multi-task, multi-domain learning with domain generalization capabilities through adversarial training and feature separation.

## Overview

Dis-AE 2 implements a **Domain Separation Network (DSN)** combined with **Multi-Domain Adversarial Training** to learn robust representations that generalize across domains while maintaining strong task performance. The model addresses the encoder conflict problem by separating features into:

- **Shared features**: Domain-invariant representations used for task prediction
- **Private features**: Domain-specific representations used for reconstruction

This separation allows the model to:
1. Remove domain-specific information from task-relevant features (improving generalization)
2. Preserve sufficient information for accurate data reconstruction
3. Handle multiple independent domain factors simultaneously

## Key Features

- **Scikit-learn compatible API**: Simple `fit()`, `predict()`, `embed()` interface
- **Multi-task learning**: Support for multiple classification tasks with different cardinalities
- **Multi-domain modeling**: Handle multiple independent domain factors (e.g., machine, time, location)
- **Domain generalization**: Learn domain-invariant features through adversarial training
- **Autoencoder capability**: Reconstruct original inputs from learned representations
- **Feature separation**: Explicit split between shared and private feature spaces
- **Model persistence**: Save and load trained models

## Installation

```bash
# Install with disae2 module dependencies
pip install -e ".[disae2]"

# Or install all dependencies
pip install -e ".[all]"
```

### Dependencies

- Python >= 3.7
- PyTorch >= 1.8.0
- NumPy >= 1.19.0
- scikit-learn >= 0.24.0

### Optional Dependencies

For running examples and notebooks:
```bash
pip install -e ".[examples]"  # Adds pandas, matplotlib
```

## Quick Start

### Option 1: Interactive Notebooks (Recommended for Learning)

```bash
# From repository root
cd examples/notebooks
jupyter notebook 01_disae2_basic_usage.ipynb
```

Start with the basic usage notebook for a guided introduction with visualizations.

### Option 2: Python API (For Production Use)

```python
from sysmexcbctools import DisAE
import numpy as np

# Generate data
X = np.random.randn(1000, 32)  # 1000 samples, 32 features
y_tasks = np.random.randint(0, 2, size=(1000, 2))  # 2 binary tasks
y_domains = np.random.randint(0, 5, size=(1000, 3))  # 3 domain factors

# Initialize model
model = DisAE(
    input_dim=32,
    num_tasks=[2, 2],  # Two binary tasks
    num_domains=[5, 5, 5],  # Three domain factors with 5 classes each
)

# Train model
model.fit(X, y_tasks, y_domains, max_epochs=100)

# Make predictions
y_pred = model.predict_tasks(X)  # List of predictions for each task
shared, private = model.embed(X)  # Get feature embeddings
```

## API Reference

### DisAE Class

The main class for the Dis-AE 2 model.

#### Initialization

```python
model = DisAE(
    input_dim,                    # Input feature dimensionality
    latent_dim=16,                # Total latent dimension (must be even)
    shared_dim=None,              # Shared feature dimension (default: latent_dim // 2)
                                  # NOTE: Must equal private_dim for orthogonality loss
    private_dim=None,             # Private feature dimension (default: latent_dim // 2)
                                  # NOTE: Must equal shared_dim for orthogonality loss
    num_tasks=[2],                # List of class counts for each task
    num_domains=[2],              # List of cardinalities for each domain factor
    hidden_dims=[64, 32],         # Encoder/decoder hidden layer sizes
    reconstruction_weight=0.1,    # Reconstruction loss weight
    adversarial_weight=1.0,       # Adversarial loss weight
    orthogonality_weight=0.1,     # Orthogonality loss weight
    learning_rate=0.01,           # Learning rate
    batch_size=128,               # Training batch size
    device='cuda',                # 'cuda' or 'cpu'
    random_state=None,            # Random seed
)
```

#### Training

```python
model.fit(
    X,                              # Training data (n_samples, n_features)
    y_tasks,                        # Task labels (n_samples, n_tasks)
    y_domains,                      # Domain labels (n_samples, n_domains)
    X_val=None,                     # Optional validation data
    y_tasks_val=None,               # Optional validation task labels
    y_domains_val=None,             # Optional validation domain labels
    max_epochs=100,                 # Maximum training epochs
    early_stopping_patience=16,     # Early stopping patience
    verbose=True,                   # Print training progress
)
```

**Note**: All labels must be pre-encoded as integers (0, 1, 2, ...).

#### Prediction Methods

```python
# Task predictions
y_pred = model.predict_tasks(X)           # Returns list of label arrays
y_proba = model.predict_tasks_proba(X)    # Returns list of probability arrays

# Domain predictions (should be poor for good generalization)
d_pred = model.predict_domains(X)         # Returns list of domain label arrays
d_proba = model.predict_domains_proba(X)  # Returns list of domain probability arrays

# Embeddings
shared, private = model.embed(X, private=True)   # Returns (shared, private) features
shared_only = model.embed(X, private=False)      # Returns only shared features

# Reconstruction
X_reconstructed = model.reconstruct(X)    # Reconstruct input data
```

#### Model Persistence

```python
# Save model
model.save('model.pkl')

# Load model
loaded_model = DisAE.load('model.pkl')
```

## Architecture Details

### Domain Separation Network (DSN)

The DSN architecture splits the feature encoder into two pathways:

```
Input (X)
    ↓
Base Encoder
    ↓
    ├─→ Shared Projection → Shared Features → Task Classifiers
    │                                      └→ Domain Discriminators (adversarial)
    └─→ Private Projection → Private Features ─┐
                                               ↓
                                    [Shared + Private] → Decoder → Reconstructed Input
```

### Loss Function

The model optimizes a multi-objective loss:

```
Total Loss = Task Loss + λ_adv × Adversarial Loss + λ_rec × Reconstruction Loss + λ_orth × Orthogonality Loss
```

Where:
- **Task Loss**: Sum of cross-entropy losses for all tasks
- **Adversarial Loss**: Negative discriminator loss (encourages domain confusion)
- **Reconstruction Loss**: MSE between original and reconstructed inputs
- **Orthogonality Loss**: Penalizes correlation between shared and private features

### Training Dynamics

The model alternates between two update steps:

1. **Discriminator Step**: Update domain discriminators to better classify domains from shared features
2. **Generator Step**: Update encoder, task classifiers, and decoder to:
   - Improve task accuracy
   - Fool domain discriminators (remove domain information)
   - Improve reconstruction quality
   - Maintain feature orthogonality

## Examples

### Jupyter Notebooks (Recommended)

Interactive tutorials with visualizations:

1. **[01_disae2_basic_usage.ipynb](../../examples/notebooks/01_disae2_basic_usage.ipynb)** - Introduction with synthetic data
   - Basic API usage
   - Feature visualization
   - Domain invariance verification

2. **[02_disae2_data_B.ipynb](../../examples/notebooks/02_disae2_data_B.ipynb)** - Real-world example with data_B.csv
   - Complete workflow on CBC-like data
   - Comparison with baseline methods
   - Per-domain performance analysis

3. **[03_disae2_advanced.ipynb](../../examples/notebooks/03_disae2_advanced.ipynb)** - Advanced topics
   - Hyperparameter tuning
   - Model persistence and reuse
   - Using embeddings for downstream tasks

### Python Scripts

For command-line execution:

- [`examples/basic_usage.py`](../../examples/basic_usage.py) - Synthetic data example
- [`examples/data_B_example.py`](../../examples/data_B_example.py) - data_B.csv example

```bash
# From repository root
python examples/basic_usage.py
python examples/data_B_example.py
```

## Use Cases

### 1. Domain Generalization

Train on source domains and generalize to unseen target domains:

```python
# Train on domains 0, 1
train_mask = (y_domains[:, 0] == 0) | (y_domains[:, 0] == 1)
model.fit(X[train_mask], y_tasks[train_mask], y_domains[train_mask])

# Test on domain 2
test_mask = y_domains[:, 0] == 2
y_pred = model.predict_tasks(X[test_mask])
```

### 2. Multi-Task Learning

Learn multiple related tasks simultaneously:

```python
# Two tasks: sentiment (3 classes) and topic (10 classes)
model = DisAE(
    input_dim=768,  # e.g., BERT embeddings
    num_tasks=[3, 10],
    num_domains=[5],  # e.g., 5 different data sources
)
```

### 3. Domain-Invariant Representations

Extract features for downstream tasks:

```python
# Get domain-invariant features
shared_features = model.embed(X, private=False)

# Use for downstream classifier
from sklearn.linear_model import LogisticRegression
downstream_model = LogisticRegression()
downstream_model.fit(shared_features, downstream_labels)
```

### 4. Data Augmentation via Reconstruction

Generate reconstructed samples with domain-specific variations:

```python
# Original data
X_reconstructed = model.reconstruct(X)

# Can manipulate private features to generate variations
shared, private = model.embed(X, private=True)
private_modified = private + np.random.randn(*private.shape) * 0.1
# (Note: would need to expose decoder input for full flexibility)
```

## Model Selection and Hyperparameters

### Key Hyperparameters

| Parameter | Description | Typical Range | Default |
|-----------|-------------|---------------|---------|
| `latent_dim` | Total feature dimension | 8-64 | 16 |
| `hidden_dims` | Encoder/decoder architecture | [32,16] to [256,128] | [64,32] |
| `reconstruction_weight` | Reconstruction loss importance | 0.01-1.0 | 0.1 |
| `adversarial_weight` | Adversarial loss importance | 0.5-2.0 | 1.0 |
| `orthogonality_weight` | Feature separation importance | 0.01-1.0 | 0.1 |
| `learning_rate` | Optimizer learning rate | 0.001-0.1 | 0.01 |
| `d_steps_per_g_step` | Discriminator updates per generator update | 1-5 | 1 |

### Training Tips

1. **Early Stopping**: Always use validation data for early stopping to prevent overfitting
2. **Learning Rate**: Start with 0.01, reduce if training is unstable
3. **Balance Loss Weights**: If one objective dominates, adjust the corresponding weight
4. **Batch Size**: Larger batches (128-256) typically work better for adversarial training
5. **Feature Dimensions**: Shared dim should be large enough for task information, private dim for domain information

### Evaluating Domain Generalization

Good domain generalization is indicated by:
- **High task accuracy** on validation/test domains
- **Low domain classification accuracy** from shared features (close to random chance)
- **Smooth probability distributions** for domain predictions (uniform-like)

```python
# Check domain invariance
d_proba = model.predict_domains_proba(X_test)
for i, proba in enumerate(d_proba):
    entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1).mean()
    max_entropy = np.log(proba.shape[1])  # Uniform distribution
    print(f"Domain {i} entropy: {entropy:.3f} / {max_entropy:.3f} (higher is better)")
```

## Comparison with Other Methods

| Method | Multi-Task | Multi-Domain | Feature Separation | Reconstruction |
|--------|-----------|--------------|-------------------|----------------|
| **Dis-AE 2** | ✓ | ✓ | ✓ | ✓ |
| DANN | ✗ | ✗ | ✗ | ✗ |
| MDANN | ✓ | ✓ | ✗ | ✗ |
| VAE | ✗ | ✗ | ✗ | ✓ |
| Domain Confusion | ✗ | ✗ | ✗ | ✗ |

## Limitations

- **Equal Dimension Requirement**: Shared and private dimensions must be equal due to the orthogonality loss implementation (uses element-wise dot product)
- **Computational Cost**: Adversarial training is slower than standard supervised learning
- **Hyperparameter Sensitivity**: Requires tuning of loss weights for optimal performance
- **Tabular Data Focus**: Current implementation uses MLP architecture (image support can be added)
- **Pre-encoded Labels Required**: Does not handle categorical encoding internally

## References

1. Bousmalis et al. "Domain Separation Networks" (NeurIPS 2016)
2. Ganin & Lempitsky "Domain-Adversarial Training of Neural Networks" (JMLR 2016)
3. Gulrajani & Lopez-Paz "In Search of Lost Domain Generalization" (ICLR 2021)
