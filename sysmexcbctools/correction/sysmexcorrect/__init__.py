"""
SysmexCorrect: GAM-based covariate correction for Sysmex CBC data

This package provides tools for correcting spurious covariate effects in
complete blood count data using Generalized Additive Models (GAMs).
"""

from .gam_correction import GAMCorrector
from .utils import centralise, mad

__all__ = ["GAMCorrector", "mad", "centralise"]
__version__ = "0.1.0"
