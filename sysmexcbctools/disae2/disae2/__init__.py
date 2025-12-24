"""
Dis-AE 2: Domain Separation Network Multi-Domain Adversarial Autoencoder

A multi-task, multi-domain learning library with domain generalization capabilities.
"""

__version__ = "0.1.0"

from .model import DisAE
from .networks import Decoder, DSNFeaturizer, MultiTaskClassifier
from .training import EarlyStopping
from .utils import denormalize_data, normalize_data

__all__ = [
    'DisAE',
    'DSNFeaturizer',
    'Decoder',
    'MultiTaskClassifier',
    'EarlyStopping',
    'normalize_data',
    'denormalize_data',
]
