"""
Configuration Management Module
================================

Provides YAML-based configuration loading with variable substitution
for managing dataset paths and processing parameters.
"""

from .config_loader import (
    ConfigLoader,
    get_config_loader,
    get_dataset_path,
    get_file_path,
    load_config,
)

__all__ = [
    'get_config_loader',
    'load_config',
    'get_dataset_path',
    'get_file_path',
    'ConfigLoader',
]
