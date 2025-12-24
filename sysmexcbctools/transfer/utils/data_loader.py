"""
Data loading utilities with configuration-based path management.
Replace your existing file loading functions with these.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Import the configuration loader
from ..config.config_loader import get_config_loader


class DataLoader:
    """Data loader with configuration-based path resolution."""

    def __init__(self, config_file: str = "config/data_paths.yaml",
                 environment: str = "production"):
        """
        Initialise the data loader.
        
        Args:
            config_file: Path to configuration file
            environment: Environment to use
        """
        self.config = get_config_loader(config_file, environment)

    def load_numpy_file(self, file_category: str, file_name: str,
                       allow_pickle: bool = True) -> np.ndarray:
        """
        Load a NumPy file using configuration.
        
        Args:
            file_category: File category from config (e.g., 'centile_samples')
            file_name: File name from config (e.g., 'strides')
            allow_pickle: Whether to allow loading pickled objects (default: True)
        
        Returns:
            Loaded numpy array
        
        Example:
            # Instead of: np.load("./data/raw/strides_centile_samples.npy")
            data = loader.load_numpy_file('centile_samples', 'strides')
        """
        file_path = self.config.get_file_path(file_category, file_name)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            return np.load(file_path, allow_pickle=allow_pickle)
        except ValueError as e:
            if "allow_pickle" in str(e):
                print(f"Warning: File {file_path} contains Python objects.")
                print("Loading with allow_pickle=True for compatibility.")
                return np.load(file_path, allow_pickle=True)
            else:
                raise

    def load_dataset_files(self, category: str, dataset: str,
                          pattern: str = "*", recursive: bool = True) -> list[Path]:
        """
        Get list of files in a dataset directory.
        
        Args:
            category: Dataset category ('processed', 'raw')
            dataset: Dataset name ('interval_36', 'strides', etc.)
            pattern: File pattern to match (e.g., '*.csv', '*.npy')
            recursive: Whether to search recursively
        
        Returns:
            List of matching file paths
        
        Example:
            # Instead of: list(Path("./data/processed/STRIDES").glob("*.csv"))
            files = loader.load_dataset_files('processed', 'strides', '*.csv')
        """
        dataset_path = self.config.get_dataset_path(category, dataset)

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

        if recursive:
            return list(dataset_path.rglob(pattern))
        else:
            return list(dataset_path.glob(pattern))

    def load_csv_from_dataset(self, category: str, dataset: str,
                             filename: str, **kwargs) -> pd.DataFrame:
        """
        Load a CSV file from a dataset directory.
        
        Args:
            category: Dataset category
            dataset: Dataset name
            filename: CSV filename
            **kwargs: Additional arguments for pd.read_csv
        
        Returns:
            Loaded DataFrame
        
        Example:
            # Instead of: pd.read_csv("./data/processed/STRIDES/data.csv")
            df = loader.load_csv_from_dataset('processed', 'strides', 'data.csv')
        """
        dataset_path = self.config.get_dataset_path(category, dataset)
        file_path = dataset_path / filename

        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        return pd.read_csv(file_path, **kwargs)

    def load_pickle_from_dataset(self, category: str, dataset: str,
                                filename: str) -> Any:
        """
        Load a pickle file from a dataset directory.
        
        Args:
            category: Dataset category
            dataset: Dataset name  
            filename: Pickle filename
        
        Returns:
            Loaded object
        """
        dataset_path = self.config.get_dataset_path(category, dataset)
        file_path = dataset_path / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {file_path}")

        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def get_dataset_path(self, category: str, dataset: str) -> Path:
        """
        Get the full path to a dataset directory.
        
        Args:
            category: Dataset category
            dataset: Dataset name
        
        Returns:
            Path to dataset
        
        Example:
            # Instead of: Path("./data/processed/STRIDES")
            path = loader.get_dataset_path('processed', 'strides')
        """
        return self.config.get_dataset_path(category, dataset)

    def get_file_path(self, file_category: str, file_name: str) -> Path:
        """
        Get the full path to a specific configured file.
        
        Args:
            file_category: File category
            file_name: File name
        
        Returns:
            Path to file
        
        Example:
            # Instead of: Path("./data/raw/strides_centile_samples.npy")
            path = loader.get_file_path('centile_samples', 'strides')
        """
        return self.config.get_file_path(file_category, file_name)


# Convenience functions for quick usage
def load_centile_samples(dataset: str, config_file: str = "config/data_paths.yaml",
                        allow_pickle: bool = True) -> np.ndarray:
    """
    Quick function to load centile samples for a dataset.
    
    Args:
        dataset: Dataset name ('strides', 'interval_baseline', 'interval_baseline_36', 'interval_baseline_41')
        config_file: Path to config file
        allow_pickle: Whether to allow loading pickled objects (default: True)
    
    Returns:
        Loaded numpy array
    
    Example:
        # Instead of: np.load("./data/raw/strides_centile_samples.npy")
        data = load_centile_samples('strides')
    """
    loader = DataLoader(config_file)
    return loader.load_numpy_file('centile_samples', dataset, allow_pickle=allow_pickle)


def get_dataset_files(category: str, dataset: str, pattern: str = "*",
                     config_file: str = "config/data_paths.yaml") -> list[Path]:
    """
    Quick function to get files from a dataset directory.
    
    Args:
        category: Dataset category
        dataset: Dataset name
        pattern: File pattern
        config_file: Path to config file
    
    Returns:
        List of matching file paths
    
    Example:
        # Instead of: list(Path("./data/processed/STRIDES").glob("*.csv"))
        files = get_dataset_files('processed', 'strides', '*.csv')
    """
    loader = DataLoader(config_file)
    return loader.load_dataset_files(category, dataset, pattern)


def load_dataset_csv(category: str, dataset: str, filename: str,
                    config_file: str = "config/data_paths.yaml", **kwargs) -> pd.DataFrame:
    """
    Quick function to load a CSV from a dataset.
    
    Args:
        category: Dataset category
        dataset: Dataset name
        filename: CSV filename
        config_file: Path to config file
        **kwargs: Additional arguments for pd.read_csv
    
    Returns:
        Loaded DataFrame
    
    Example:
        # Instead of: pd.read_csv("./data/processed/STRIDES/data.csv")
        df = load_dataset_csv('processed', 'strides', 'data.csv')
    """
    loader = DataLoader(config_file)
    return loader.load_csv_from_dataset(category, dataset, filename, **kwargs)
