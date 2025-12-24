"""
Utility functions for Dis-AE 2
"""

import numpy as np


def normalize_data(X: np.ndarray) -> tuple:
    """
    Normalize data to zero mean and unit variance.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input data

    Returns
    -------
    X_normalized : np.ndarray, shape (n_samples, n_features)
        Normalized data
    mean : np.ndarray, shape (n_features,)
        Feature means
    std : np.ndarray, shape (n_features,)
        Feature standard deviations
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1  # Avoid division by zero
    X_normalized = (X - mean) / std
    return X_normalized, mean, std


def denormalize_data(X_normalized: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Denormalize data using saved mean and std.

    Parameters
    ----------
    X_normalized : np.ndarray, shape (n_samples, n_features)
        Normalized data
    mean : np.ndarray, shape (n_features,)
        Feature means
    std : np.ndarray, shape (n_features,)
        Feature standard deviations

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
        Original scale data
    """
    return X_normalized * std + mean
