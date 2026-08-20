"""
Configuration loader for data paths.
Handles loading and resolving paths from YAML configuration files.
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigLoader:
    """Load and manage configuration for data paths."""

    def __init__(self, config_file: str = "config/data_paths.yaml"):
        """
        Initialise the configuration loader.

        Args:
            config_file: Path to the YAML configuration file
        """
        self.config_file = Path(config_file)
        self._config = None
        self._load_config()

    def _load_config(self) -> None:
        """Load the YAML configuration file."""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")

        with open(self.config_file) as f:
            self._config = yaml.safe_load(f)

        # Resolve variable substitutions
        self._resolve_variables()

    def _resolve_variables(self) -> None:
        """Resolve variable substitutions in the configuration."""
        def resolve_string(s: str, context: Dict[str, Any]) -> str:
            """Resolve ${variable.path} references in a string."""
            if not isinstance(s, str):
                return s

            # Find all ${...} patterns
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, s)

            for match in matches:
                # Navigate through nested dictionary structure
                keys = match.split('.')
                value = context
                try:
                    for key in keys:
                        value = value[key]
                    s = s.replace(f"${{{match}}}", str(value))
                except KeyError:
                    raise ValueError(f"Could not resolve variable: {match}")

            return s

        def resolve_dict(d: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively resolve variables in a dictionary."""
            resolved = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    resolved[key] = resolve_dict(value, context)
                elif isinstance(value, str):
                    resolved[key] = resolve_string(value, context)
                else:
                    resolved[key] = value
            return resolved

        self._config = resolve_dict(self._config, self._config)

    def get_dataset_path(self, category: str, dataset: str) -> Path:
        """
        Get the path for a dataset.

        Args:
            category: Category (e.g., 'processed', 'raw')
            dataset: Dataset name (e.g., 'interval_36', 'strides')

        Returns:
            Path to the dataset
        """
        try:
            path_str = self._config['datasets'][category][dataset]
            return Path(path_str)
        except KeyError:
            raise ValueError(f"Dataset not found: {category}.{dataset}")

    def get_file_path(self, file_category: str, file_name: str) -> Path:
        """
        Get the path for a specific file.

        Args:
            file_category: File category (e.g., 'centile_samples')
            file_name: File name (e.g., 'strides', 'interval_baseline')

        Returns:
            Path to the file
        """
        try:
            path_str = self._config['files'][file_category][file_name]
            return Path(path_str)
        except KeyError:
            raise ValueError(f"File not found: {file_category}.{file_name}")

    def list_datasets(self, category: Optional[str] = None) -> Dict[str, Any]:
        """
        List available datasets.

        Args:
            category: Optional category to filter by

        Returns:
            Dictionary of available datasets
        """
        datasets = self._config['datasets']
        if category:
            return {category: datasets.get(category, {})}
        return datasets


def load_config(config_file: str = "config/data_paths.yaml") -> Dict[str, Any]:
    """
    Load configuration file and return as a dictionary.

    Loads the YAML config and resolves all variable substitutions, returning
    the complete configuration as a dict.

    Args:
        config_file: Path to the YAML configuration file

    Returns:
        Dictionary containing the complete configuration with resolved variables

    Examples:
        >>> config = load_config('config/data_paths.yaml')
        >>> source_dir = config['datasets']['raw']['strides_merged']
        >>> samples_file = config['files']['centile_samples']['strides']
    """
    return ConfigLoader(config_file)._config
