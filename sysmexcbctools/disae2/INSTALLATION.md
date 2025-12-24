# Installation and Usage Guide

## Quick Start

### Installation

You can install Dis-AE 2 in several ways:

#### Option 1: Install from source (recommended for development)

```bash
cd /home/dk659/MDANN/DomainBed/disae2
pip install -e .
```

This installs the package in "editable" mode, allowing you to modify the code and see changes immediately.

#### Option 2: Install dependencies only

```bash
cd /home/dk659/MDANN/DomainBed/disae2
pip install -r requirements.txt
```

Then use the package by adding the path to your Python scripts:

```python
import sys
sys.path.insert(0, '/home/dk659/MDANN/DomainBed/disae2')
from disae2 import DisAE
```

### Verify Installation

Run the test suite to verify everything is working:

```bash
cd /home/dk659/MDANN/DomainBed/disae2
pytest tests/test_model.py -v
```

All 12 tests should pass.

### Run Examples

#### Basic synthetic data example:

```bash
python examples/basic_usage.py
```

This demonstrates the core API with synthetic multi-task, multi-domain data.

#### Data_B dataset example:

```bash
python examples/data_B_example.py
```

This shows how to use Dis-AE 2 on real domain generalization data.

## Usage

### Minimal Example

```python
from disae2 import DisAE
import numpy as np

# Generate data
X = np.random.randn(1000, 32)
y_tasks = np.random.randint(0, 2, size=(1000, 1))  # Single binary task
y_domains = np.random.randint(0, 5, size=(1000, 1))  # Single domain factor

# Train model
model = DisAE(input_dim=32, num_tasks=[2], num_domains=[5])
model.fit(X, y_tasks, y_domains, max_epochs=100)

# Make predictions
y_pred = model.predict_tasks(X)
shared_features = model.embed(X, private=False)
```

## Package Structure

```
disae2/
├── disae2/              # Main package
│   ├── __init__.py      # Package initialization
│   ├── model.py         # DisAE main class
│   ├── networks.py      # Neural network components
│   ├── training.py      # Training utilities
│   └── utils.py         # Helper functions
├── examples/            # Usage examples
│   ├── basic_usage.py   # Basic synthetic example
│   └── data_B_example.py # Real dataset example
├── tests/               # Unit tests
│   └── test_model.py    # Model tests
├── setup.py             # Package setup
├── requirements.txt     # Dependencies
└── README.md            # Documentation
```

## Dependencies

- Python >= 3.7
- PyTorch >= 1.8.0
- NumPy >= 1.19.0
- scikit-learn >= 0.24.0

Optional (for examples):
- pandas >= 1.2.0

Optional (for testing):
- pytest >= 6.0

## Troubleshooting

### ImportError: No module named 'disae2'

If running examples gives this error, either:
1. Install the package with `pip install -e .`
2. Or the examples already include the path fix, so run them from the `disae2/` directory

### CUDA out of memory

If you get CUDA memory errors:
1. Reduce `batch_size` (default 128, try 64 or 32)
2. Or use CPU: `model = DisAE(..., device='cpu')`

### Training is slow

This is expected for adversarial training. To speed up:
1. Use GPU: `device='cuda'`
2. Reduce `max_epochs`
3. Use early stopping with validation data
4. Reduce model size with smaller `hidden_dims` or `latent_dim`

## Next Steps

- Read the [README.md](README.md) for detailed API documentation
- Check the [examples/](examples/) directory for more usage patterns
- Modify hyperparameters in the examples to see their effects
- Try Dis-AE 2 on your own multi-domain datasets
