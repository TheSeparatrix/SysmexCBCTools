"""
SysmexCBCTools: A toolkit for processing and analysing Sysmex CBC data

This package provides tools for:
- Data cleaning and preprocessing (data module)
- Cross-analyser alignment (transfer module)
- Covariate correction (correction module)
- Domain-invariant feature learning (disae2 module)
"""

from importlib.metadata import version

__version__ = version("sysmexcbctools")

# Import submodules for convenient access
from sysmexcbctools import correction, data, disae2, transfer
from sysmexcbctools.correction.sysmexcorrect import GAMCorrector

# Tier 1: Top-level exports (main API classes)
from sysmexcbctools.data.sysmexclean import XNSampleProcessor
from sysmexcbctools.transfer.sysmexalign import (
    FlowTransformer,
    ImpedanceTransformer,
    XNSampleTransformer,
)

# DisAE requires torch, which is optional
try:
    from sysmexcbctools.disae2.disae2 import DisAE
    _has_disae2 = True
except ImportError:
    _has_disae2 = False
    DisAE = None

__all__ = [
    # Version
    '__version__',
    # Submodules
    'data',
    'transfer',
    'correction',
    'disae2',
    # Tier 1 API classes
    'XNSampleProcessor',
    'FlowTransformer',
    'ImpedanceTransformer',
    'XNSampleTransformer',
    'GAMCorrector',
    'DisAE',
]
