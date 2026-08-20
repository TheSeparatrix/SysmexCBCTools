"""
Configuration Management Module
================================

Provides YAML-based configuration loading with variable substitution
for managing dataset paths and processing parameters.
"""

from .config_loader import ConfigLoader, load_config

__all__ = [
    'ConfigLoader',
    'load_config',
]
