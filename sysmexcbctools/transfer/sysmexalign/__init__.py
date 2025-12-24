"""
Sysmex Alignment Package
========================

This package provides tools for aligning raw Sysmex haematology analyzer data
between different machines using various transformation methods including
Optimal Transport (GMM-OT) and MAD/median-based alignment.

Main API Classes
----------------
FlowTransformer : Transformer for flow cytometry channels (RET, WDF, WNR, PLTF)
ImpedanceTransformer : Transformer for impedance channels (RBC, PLT)
XNSampleTransformer : Transformer for XN_SAMPLE tabular data

Core Functions
--------------
From gmm_ot:
    compute_gmm_transport_map
    transform_points_gmm
    validate_transformation

From alignment_1d:
    transform_nonnormal
    mad
    transform_impedance_data

From load_and_preprocess:
    SysmexRawData
    concatenate_sysmex_data
    parse_sysmex_raw_filename
    load_sample_nos
    load_sample_nos_from_config
    is_overflow_file
    filter_overflow_files
    get_overflow_files
    merge_csv_with_overflows
"""

from .alignment_1d import (
    fit_gmm_plt,
    fit_gmm_rbc,
    mad,
    transform_impedance_data,
    transform_nonnormal,
)

# Import API classes
from .api import FlowTransformer, ImpedanceTransformer, XNSampleTransformer
from .flow_gating_pipeline import (
    FlowGate,
    FlowGatingSystem,
)
from .gate_utils import (
    classify_points_by_gate,
    find_default_gate_file,
    initialize_gmm_means_from_gates,
    load_gates,
)
from .gmm_ot import (
    compute_gmm_transport_map,
    transform_points_gmm,
    validate_transformation,
)
from .load_and_preprocess import (
    SysmexRawData,
    concatenate_sysmex_data,
    filter_overflow_files,
    get_overflow_files,
    is_overflow_file,
    load_sample_nos,
    load_sample_nos_from_config,
    merge_csv_with_overflows,
    parse_sysmex_raw_filename,
)

__all__ = [
    # API classes
    'FlowTransformer',
    'ImpedanceTransformer',
    'XNSampleTransformer',
    # GMM-OT functions
    'compute_gmm_transport_map',
    'transform_points_gmm',
    'validate_transformation',
    # Alignment functions
    'transform_nonnormal',
    'mad',
    'transform_impedance_data',
    'fit_gmm_rbc',
    'fit_gmm_plt',
    # Data loading functions
    'SysmexRawData',
    'concatenate_sysmex_data',
    'parse_sysmex_raw_filename',
    'load_sample_nos',
    'load_sample_nos_from_config',
    # Overflow file handling
    'is_overflow_file',
    'filter_overflow_files',
    'get_overflow_files',
    'merge_csv_with_overflows',
    # Gate utilities
    'load_gates',
    'find_default_gate_file',
    'initialize_gmm_means_from_gates',
    'classify_points_by_gate',
    # Flow gating pipeline
    'FlowGate',
    'FlowGatingSystem',
]

__version__ = '0.1.0'
