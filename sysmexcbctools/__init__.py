"""
SysmexCBCTools: A toolkit for processing and analysing Sysmex CBC data.

This package provides tools for:
- Data cleaning and preprocessing (data module)
- Cross-analyser alignment (transfer module)
- Covariate correction (correction module)
- Domain-invariant feature learning (disae2 module)

Optional modules are imported lazily. If a module's dependencies are not
installed, its classes will be ``None`` in this namespace. Install extras
to enable them, e.g. ``pip install 'sysmexcbctools[transfer]'``.
"""

from importlib.metadata import version

__version__ = version("sysmexcbctools")

# Tier 1: Top-level exports (main API classes)
# Data module -- always available
from sysmexcbctools.data.sysmexclean import XNSampleProcessor

# GAMCorrector requires pygam (optional: pip install 'sysmexcbctools[correction]')
try:
    from sysmexcbctools.correction.sysmexcorrect import GAMCorrector
    _has_correction = True
except ImportError:
    _has_correction = False
    GAMCorrector = None

# Transfer classes require scipy, sklearn, etc.
# (optional: pip install 'sysmexcbctools[transfer]')
try:
    from sysmexcbctools.transfer.sysmexalign import (
        FlowTransformer,
        ImpedanceTransformer,
        XNSampleTransformer,
    )
    _has_transfer = True
except ImportError:
    _has_transfer = False
    FlowTransformer = None
    ImpedanceTransformer = None
    XNSampleTransformer = None

# DisAE requires torch (optional: pip install 'sysmexcbctools[disae2]')
try:
    from sysmexcbctools.disae2.disae2 import DisAE
    _has_disae2 = True
except ImportError:
    _has_disae2 = False
    DisAE = None

__all__ = [
    # Version
    '__version__',
    # Tier 1 API classes
    'XNSampleProcessor',
    'FlowTransformer',
    'ImpedanceTransformer',
    'XNSampleTransformer',
    'GAMCorrector',
    'DisAE',
]
