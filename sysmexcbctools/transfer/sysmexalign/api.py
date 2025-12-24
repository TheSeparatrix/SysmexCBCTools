"""
Sysmex Alignment API
====================

Scikit-learn style API for aligning Sysmex haematology analyzer data between different machines.

This module provides three main transformer classes:
- FlowTransformer: For flow cytometry channels (RET, WDF, WNR, PLTF)
- ImpedanceTransformer: For impedance channels (RBC, PLT)
- XNSampleTransformer: For XN_SAMPLE tabular data

Example usage:
    # Flow cytometry alignment
    transformer = FlowTransformer(channel='RET')
    transformer.fit(source_files, target_files)
    transformed_files = transformer.transform(source_files, output_dir='output/')

    # Impedance alignment
    imp_transformer = ImpedanceTransformer()
    imp_transformer.fit(source_df, target_df)
    transformed_df = imp_transformer.transform(source_df)

    # XN_SAMPLE alignment
    xn_transformer = XNSampleTransformer(columns=['HGB', 'RBC', 'WBC'])
    xn_transformer.fit(source_df, target_df)
    transformed_df = xn_transformer.transform(source_df)
"""

import copy
import os
import pickle
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from .alignment_1d import (
    mad,
    transform_nonnormal,
)
from .gate_utils import (
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
    parse_sysmex_raw_filename,
)


class FlowTransformer:
    """
    Transform flow cytometry data between different Sysmex analyzers using GMM-OT.

    This transformer uses Gaussian Mixture Models (GMM) combined with Optimal Transport (OT)
    to align flow cytometry channels (RET, WDF, WNR, PLTF) between different analyzers.

    Parameters
    ----------
    channel : str
        Flow cytometry channel to transform. Must be one of: 'RET', 'WDF', 'WNR', 'PLTF'.
    n_components : int, default=10
        Number of Gaussian components for GMM fitting.
    covariance_type : str, default='full'
        Type of covariance parameters. Must be one of: 'full', 'tied', 'diag', 'spherical'.
    transport_method : str, default='weight'
        Transport method for transformation. Options: 'weight', 'rand', 'max'.
    max_samples : int, default=1000000
        Maximum number of samples to use for GMM fitting (downsamples if exceeded).
    preserve_rare : bool, default=True
        Whether to preserve rare cell populations during transformation.
    rare_threshold : float or None, default=None
        Probability threshold for identifying rare populations (channel-specific if None).
    omega_threshold : float or None, default=None
        Threshold for transport plan weights (channel-specific if None).
    n_jobs : int, default=-1
        Number of parallel jobs for file processing (-1 uses all cores).
    random_state : int or None, default=None
        Random seed for reproducibility.
    save_fitted_data : bool, default=False
        Whether to save the downsampled source and target data used for fitting.
        If True, data is stored in source_data_ and target_data_ attributes.
        Set to False by default to avoid storing potentially sensitive patient data
        and to save memory.
    use_gate_init : bool, default=True
        Whether to use gate-informed GMM initialization to prevent component collapse.
        If True and gate_file is found, GMM component means are initialized by
        distributing them across gated populations (e.g., n_components/n_populations
        per gate region). This ensures components start spread out across all cell
        types before fitting on natural data distribution.

        **Note**: Current gates are manually derived and approximate. Official
        Sysmex gates or expert-validated gates would be preferable for production use.
    gate_file : str or None, default=None
        Path to gate file (.pkl or .json) containing gate definitions. If None,
        will search for default gate file in standard locations (flow_gates/{channel}_gates.pkl).
    gate_init_method : str, default='equal'
        Method for distributing GMM components across populations during initialization:
        - 'equal': Allocate components equally across populations (default)
        - 'sqrt': Allocate components proportional to sqrt of population size
        - 'proportional': Allocate proportionally to population size

        **Note**: All these methods are somewhat arbitrary. Ideally, components should be
        allocated based on how "non-Gaussian" each population is (distribution complexity).
        If results are poor, this allocation strategy should be revisited.
    use_cascade_init : bool, default=False
        Whether to use cascading initialisation where target GMM is initialised from
        the fitted source GMM parameters (means and covariances). When True, the source
        GMM is fitted first (using gate-informed initialisation if available), then its
        fitted parameters are used to initialise the target GMM, ensuring structural
        correspondence between source and target components.

        **When to use:**
        - Source and target distributions have similar structure but different scales
        - You want guaranteed component correspondence (component i maps to component i)
        - Debugging transformations (easier to interpret component relationships)

        **Compatibility:**
        - Works with use_gate_init=True (source uses gates, target uses source parameters)
        - Works with use_gate_init=False (source uses k-means++, target uses source parameters)
        - Overrides gate initialisation for target GMM when enabled

        **Note:** If source and target distributions are fundamentally different, this may
        produce worse results than independent initialisation. Default is False for backward
        compatibility and conservative behaviour.

    Attributes
    ----------
    source_gmm_ : GaussianMixture
        Fitted GMM for source distribution.
    target_gmm_ : GaussianMixture
        Fitted GMM for target distribution.
    transport_dict_ : dict
        Dictionary containing transport maps and related information.
    source_data_ : ndarray or None
        Source data used for fitting (after filtering and downsampling), shape (n_samples, 2).
        Only populated if save_fitted_data=True, otherwise None.
    target_data_ : ndarray or None
        Target data used for fitting (after filtering and downsampling), shape (n_samples, 2).
        Only populated if save_fitted_data=True, otherwise None.
    is_fitted_ : bool
        Whether the transformer has been fitted.

    Examples
    --------
    >>> # Basic usage with file paths
    >>> transformer = FlowTransformer(channel='RET')
    >>> transformer.fit(
    ...     source_files=['/path/to/RET*.116.csv'],
    ...     target_files=['/path/to/RET*.116.csv']
    ... )
    >>> transformer.transform(
    ...     source_files=['/path/to/RET*.116.csv'],
    ...     output_dir='output/'
    ... )

    >>> # Using data arrays directly
    >>> transformer = FlowTransformer(channel='RET')
    >>> transformer.fit(source_data=source_array, target_data=target_array)
    >>> transformed = transformer.transform_array(source_array)

    >>> # Save and load
    >>> transformer.save('models/ret_transformer.pkl')
    >>> loaded_transformer = FlowTransformer.load('models/ret_transformer.pkl')
    """

    def __init__(
        self,
        channel: str,
        n_components: int = 10,
        covariance_type: str = "full",
        transport_method: str = "rand",
        max_samples: int = 1_000_000,
        preserve_rare: bool = True,
        rare_threshold: Optional[float] = None,
        omega_threshold: Optional[float] = None,
        n_jobs: int = -1,
        random_state: Optional[int] = None,
        save_fitted_data: bool = False,
        use_gate_init: bool = True,
        gate_file: Optional[str] = None,
        gate_init_method: str = "equal",
        use_cascade_init: bool = False,
    ):
        # Validate channel
        valid_channels = ["RET", "WDF", "WNR", "PLTF"]
        if channel not in valid_channels:
            raise ValueError(
                f"channel must be one of {valid_channels}, got '{channel}'"
            )

        self.channel = channel
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.transport_method = transport_method
        self.max_samples = max_samples
        self.preserve_rare = preserve_rare
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.save_fitted_data = save_fitted_data
        self.use_gate_init = use_gate_init
        self.gate_file = gate_file
        self.gate_init_method = gate_init_method
        self.use_cascade_init = use_cascade_init

        # Set channel-specific thresholds if not provided
        if rare_threshold is None:
            rare_threshold_map = {
                "WDF": 0.005,
                "WNR": 0.01,
                "PLTF": 0.003,
                "RET": 0.01,
            }
            self.rare_threshold = rare_threshold_map[channel]
        else:
            self.rare_threshold = rare_threshold

        if omega_threshold is None:
            omega_threshold_map = {
                "WDF": 0.03,
                "WNR": 0.05,
                "PLTF": 0.02,
                "RET": 0.05,
            }
            self.omega_threshold = omega_threshold_map[channel]
        else:
            self.omega_threshold = omega_threshold

        # Attributes set during fitting
        self.source_gmm_ = None
        self.target_gmm_ = None
        self.transport_dict_ = None
        self.source_data_ = None
        self.target_data_ = None
        self.is_fitted_ = False

    def fit(
        self,
        source_files: Optional[List[str]] = None,
        target_files: Optional[List[str]] = None,
        source_data: Optional[np.ndarray] = None,
        target_data: Optional[np.ndarray] = None,
        source_sample_nos: Optional[List[str]] = None,
        target_sample_nos: Optional[List[str]] = None,
    ):
        """
        Fit the GMM models and compute the transport map.

        Parameters
        ----------
        source_files : list of str, optional
            List of source .116.csv file paths. Provide either files or data arrays.
        target_files : list of str, optional
            List of target .116.csv file paths. Provide either files or data arrays.
        source_data : ndarray, optional
            Source data array of shape (n_samples, n_features). Alternative to source_files.
        target_data : ndarray, optional
            Target data array of shape (n_samples, n_features). Alternative to target_files.
        source_sample_nos : list of str, optional
            Sample numbers to filter source files (if using files).
        target_sample_nos : list of str, optional
            Sample numbers to filter target files (if using files).

        Returns
        -------
        self : FlowTransformer
            Fitted transformer.
        """
        # Load data
        if source_data is None:
            if source_files is None:
                raise ValueError("Must provide either source_files or source_data")
            source_data = self._load_files(source_files, source_sample_nos)

        if target_data is None:
            if target_files is None:
                raise ValueError("Must provide either target_files or target_data")
            target_data = self._load_files(target_files, target_sample_nos)

        # Remove saturated measurements (0s and 255s)
        source_data = source_data[
            (source_data != 0).all(axis=1) & (source_data != 255).all(axis=1)
        ]
        target_data = target_data[
            (target_data != 0).all(axis=1) & (target_data != 255).all(axis=1)
        ]

        print(f"Fitting GMM models for channel {self.channel}...")
        print(f"Source data: {source_data.shape[0]:,} samples")
        print(f"Target data: {target_data.shape[0]:,} samples")

        # Downsample to max_samples using natural data distribution
        rng = np.random.default_rng(seed=self.random_state)
        if source_data.shape[0] > self.max_samples:
            print(f"Downsampling source to {self.max_samples:,} samples...")
            indices = rng.choice(source_data.shape[0], self.max_samples, replace=False)
            source_data = source_data[indices]

        if target_data.shape[0] > self.max_samples:
            print(f"Downsampling target to {self.max_samples:,} samples...")
            indices = rng.choice(target_data.shape[0], self.max_samples, replace=False)
            target_data = target_data[indices]

        # Try to load gates if gate initialization is enabled
        gates = None
        if self.use_gate_init:
            gate_file = self.gate_file
            if gate_file is None:
                # Try to find default gate file
                gate_file = find_default_gate_file(self.channel)

            if gate_file is not None:
                try:
                    gates = load_gates(gate_file, self.channel)
                    print(f"✓ Loaded gates from: {gate_file}")
                    print(f"  Gate populations: {list(gates.keys())}")
                    print(
                        "  ⚠️  Note: Gates are manually derived and approximate. "
                        "Official Sysmex gates recommended for production."
                    )
                except Exception as e:
                    warnings.warn(
                        f"Could not load gates from {gate_file}: {e}. "
                        f"Proceeding without gate-informed initialization."
                    )
                    gates = None
            else:
                print(
                    f"\n⚠ No gate file found for channel {self.channel}. "
                    f"Proceeding with standard GMM initialization."
                )
                print(
                    f"  To use gate-informed initialization, provide gate_file parameter or place "
                    f"{self.channel}_gates.pkl in flow_gates/ directory."
                )

        # Fit GMMs with gate-informed initialization if gates available
        if gates is not None and self.use_gate_init:
            print(
                f"\nFitting source GMM ({self.n_components} components) with gate-informed initialization..."
            )
            # Initialize means using gate information
            source_init_means = initialize_gmm_means_from_gates(
                source_data,
                gates,
                self.n_components,
                method=self.gate_init_method,
                random_state=self.random_state,
            )
            # Fit with custom initialization
            self.source_gmm_ = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                means_init=source_init_means,
                n_init=1,  # Only 1 init since we're providing means
                random_state=self.random_state,
            ).fit(source_data)

            # Fit target GMM with cascade, gate-informed, or standard initialization
            if self.use_cascade_init:
                print(
                    f"\nFitting target GMM ({self.n_components} components) with cascade initialization..."
                )
                print("  Initializing target from fitted source GMM parameters")
                # Deep copy source GMM and refit on target data
                self.target_gmm_ = copy.deepcopy(self.source_gmm_)
                self.target_gmm_.warm_start = True
                self.target_gmm_.fit(target_data)
            else:
                print(
                    f"\nFitting target GMM ({self.n_components} components) with gate-informed initialization..."
                )
                # Initialize means using gate information
                target_init_means = initialize_gmm_means_from_gates(
                    target_data,
                    gates,
                    self.n_components,
                    method=self.gate_init_method,
                    random_state=self.random_state,
                )
                # Fit with custom initialization
                self.target_gmm_ = GaussianMixture(
                    n_components=self.n_components,
                    covariance_type=self.covariance_type,
                    means_init=target_init_means,
                    n_init=1,  # Only 1 init since we're providing means
                    random_state=self.random_state,
                ).fit(target_data)
        else:
            # Standard GMM fitting with default initialization (k-means++)
            print(
                f"\nFitting source GMM ({self.n_components} components) with standard initialization..."
            )
            self.source_gmm_ = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                n_init=10,
                random_state=self.random_state,
            ).fit(source_data)

            # Fit target GMM with cascade or standard initialization
            if self.use_cascade_init:
                print(
                    f"\nFitting target GMM ({self.n_components} components) with cascade initialization..."
                )
                print("  Initializing target from fitted source GMM parameters")
                # Deep copy source GMM and refit on target data
                self.target_gmm_ = copy.deepcopy(self.source_gmm_)
                self.target_gmm_.warm_start = True
                self.target_gmm_.fit(target_data)
            else:
                print(
                    f"Fitting target GMM ({self.n_components} components) with standard initialization..."
                )
                self.target_gmm_ = GaussianMixture(
                    n_components=self.n_components,
                    covariance_type=self.covariance_type,
                    n_init=10,
                    random_state=self.random_state,
                ).fit(target_data)

        # Compute transport map
        print("Computing optimal transport map...")
        self.transport_dict_ = compute_gmm_transport_map(
            self.source_gmm_, self.target_gmm_
        )

        # Optionally save the downsampled data used for fitting (for later inspection/visualization)
        if self.save_fitted_data:
            self.source_data_ = source_data
            self.target_data_ = target_data
        else:
            self.source_data_ = None
            self.target_data_ = None

        self.is_fitted_ = True
        print("Fitting complete!")

        return self

    def transform(
        self,
        source_files: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        source_sample_nos: Optional[List[str]] = None,
    ):
        """
        Transform source files and save to output directory.

        Parameters
        ----------
        source_files : list of str
            List of source .116.csv file paths to transform.
        output_dir : str
            Directory to save transformed files.
        source_sample_nos : list of str, optional
            Sample numbers to filter files.

        Returns
        -------
        output_files : list of str
            Paths to transformed output files.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "Transformer must be fitted before transform(). Call fit() first."
            )

        if source_files is None:
            raise ValueError("Must provide source_files")

        if output_dir is None:
            raise ValueError("Must provide output_dir")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Filter out overflow files (they will be merged automatically during loading)
        source_files = filter_overflow_files(source_files)

        # Filter files by sample numbers if provided
        if source_sample_nos is not None:
            relevant_files = []
            for f in source_files:
                try:
                    parsed = parse_sysmex_raw_filename(os.path.basename(f))
                    if parsed["sample_number"] in source_sample_nos:
                        relevant_files.append(f)
                except Exception as e:
                    warnings.warn(f"Could not parse filename {f}: {e}")
            source_files = relevant_files

        print(f"Transforming {len(source_files)} files...")

        def transform_file(filepath):
            try:
                data = SysmexRawData(filepath)
                if len(data.data) == 0:
                    return None

                data.data = transform_points_gmm(
                    data.data,
                    self.source_gmm_,
                    self.transport_dict_,
                    transport_method=self.transport_method,
                    threshold=self.omega_threshold,
                    preserve_rare=self.preserve_rare,
                    rare_probability_threshold=self.rare_threshold,
                    random_state=self.random_state,
                )

                # Clip to valid range
                data.data = np.clip(data.data, 0, 255)

                # Save
                output_file = os.path.join(
                    output_dir, os.path.basename(filepath)[:-4] + "_transformed.csv"
                )
                data.to_csv(output_file)
                return output_file
            except Exception as e:
                warnings.warn(f"Error transforming {filepath}: {e}")
                return None

        output_files = Parallel(n_jobs=self.n_jobs)(
            delayed(transform_file)(f)
            for f in tqdm(source_files, desc="Transforming files")
        )

        output_files = [f for f in output_files if f is not None]
        print(f"Successfully transformed {len(output_files)} files")

        return output_files

    def transform_array(self, X: np.ndarray) -> np.ndarray:
        """
        Transform a data array directly.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "Transformer must be fitted before transform_array(). Call fit() first."
            )

        X_transformed = transform_points_gmm(
            X,
            self.source_gmm_,
            self.transport_dict_,
            transport_method=self.transport_method,
            threshold=self.omega_threshold,
            preserve_rare=self.preserve_rare,
            rare_probability_threshold=self.rare_threshold,
            random_state=self.random_state,
        )

        return np.clip(X_transformed, 0, 255)

    def validate(
        self,
        source_data: np.ndarray,
        target_data: np.ndarray,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Validate transformation quality by comparing distributions.

        Parameters
        ----------
        source_data : ndarray
            Original source data.
        target_data : ndarray
            Target distribution data.
        output_path : str, optional
            Path to save validation plot.

        Returns
        -------
        metrics : dict
            Validation metrics including Wasserstein distances and likelihoods.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "Transformer must be fitted before validate(). Call fit() first."
            )

        transformed_data = self.transform_array(source_data)

        return validate_transformation(
            source_data,
            target_data,
            transformed_data,
            self.target_gmm_,
            output_path=output_path,
        )

    def save(self, filepath: str):
        """
        Save the fitted transformer to a file.

        Parameters
        ----------
        filepath : str
            Path to save the transformer.
        """
        if not self.is_fitted_:
            raise RuntimeError("Cannot save unfitted transformer. Call fit() first.")

        state = {
            "channel": self.channel,
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "transport_method": self.transport_method,
            "max_samples": self.max_samples,
            "preserve_rare": self.preserve_rare,
            "rare_threshold": self.rare_threshold,
            "omega_threshold": self.omega_threshold,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "use_gate_init": self.use_gate_init,
            "gate_file": self.gate_file,
            "gate_init_method": self.gate_init_method,
            "use_cascade_init": self.use_cascade_init,
            "source_gmm": self.source_gmm_,
            "target_gmm": self.target_gmm_,
            "transport_dict": self.transport_dict_,
        }

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        print(f"Saved transformer to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """
        Load a fitted transformer from a file.

        Parameters
        ----------
        filepath : str
            Path to the saved transformer.

        Returns
        -------
        transformer : FlowTransformer
            Loaded transformer.
        """
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        transformer = cls(
            channel=state["channel"],
            n_components=state["n_components"],
            covariance_type=state["covariance_type"],
            transport_method=state["transport_method"],
            max_samples=state["max_samples"],
            preserve_rare=state["preserve_rare"],
            rare_threshold=state["rare_threshold"],
            omega_threshold=state["omega_threshold"],
            n_jobs=state["n_jobs"],
            random_state=state["random_state"],
            use_gate_init=state.get("use_gate_init", True),
            gate_file=state.get("gate_file", None),
            gate_init_method=state.get("gate_init_method", "equal"),
            use_cascade_init=state.get("use_cascade_init", False),
        )

        transformer.source_gmm_ = state["source_gmm"]
        transformer.target_gmm_ = state["target_gmm"]
        transformer.transport_dict_ = state["transport_dict"]
        transformer.is_fitted_ = True

        print(f"Loaded transformer from {filepath}")
        return transformer

    def _load_files(
        self, files: List[str], sample_nos: Optional[List[str]] = None
    ) -> np.ndarray:
        """Load and concatenate data from files."""
        # Filter out overflow files (they will be merged automatically during loading)
        files = filter_overflow_files(files)

        # Filter by sample numbers
        if sample_nos is not None:
            filtered_files = []
            for f in files:
                try:
                    parsed = parse_sysmex_raw_filename(os.path.basename(f))
                    if parsed["sample_number"] in sample_nos:
                        filtered_files.append(f)
                except Exception as e:
                    warnings.warn(f"Could not parse filename {f}: {e}")
            files = filtered_files

        print(f"Loading {len(files)} files...")
        data_objects = Parallel(n_jobs=self.n_jobs)(
            delayed(SysmexRawData)(f) for f in tqdm(files, desc="Loading files")
        )

        concatenated = concatenate_sysmex_data(data_objects)
        return concatenated.data


class ImpedanceTransformer:
    """
    Transform impedance data between different Sysmex analyzers.

    This transformer aligns RBC and PLT impedance histograms using GMM-based optimal transport.

    Parameters
    ----------
    gmm_sample_size : int, default=10000
        Number of samples to use for GMM fitting.
    n_jobs : int, default=-1
        Number of parallel jobs for processing (-1 uses all cores).

    Attributes
    ----------
    rbc_source_gmm_ : GaussianMixture
        Fitted GMM for source RBC distribution.
    rbc_target_gmm_ : GaussianMixture
        Fitted GMM for target RBC distribution.
    plt_source_gmm_ : GaussianMixture
        Fitted GMM for source PLT distribution.
    plt_target_gmm_ : GaussianMixture
        Fitted GMM for target PLT distribution.
    rbc_transport_dict_ : dict
        Optimal transport map for RBC.
    plt_transport_dict_ : dict
        Optimal transport map for PLT.
    is_fitted_ : bool
        Whether the transformer has been fitted.

    Examples
    --------
    >>> transformer = ImpedanceTransformer()
    >>> transformer.fit(source_df, target_df)
    >>> transformed_df = transformer.transform(source_df)
    >>> transformer.save('impedance_transformer.pkl')
    """

    def __init__(
        self,
        gmm_sample_size: int = 10000,
        n_jobs: int = -1,
        random_state: Optional[int] = None,
    ):
        self.gmm_sample_size = gmm_sample_size
        self.n_jobs = n_jobs
        self.random_state = random_state

        # GMMs will be fitted during fit()
        self.rbc_source_gmm_ = None
        self.rbc_target_gmm_ = None
        self.plt_source_gmm_ = None
        self.plt_target_gmm_ = None

        # Transport maps will be computed during fit()
        self.rbc_transport_dict_ = None
        self.plt_transport_dict_ = None

        self.is_fitted_ = False

    def fit(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        source_sample_nos: Optional[List[str]] = None,
        target_sample_nos: Optional[List[str]] = None,
    ):
        """
        Fit GMM models and compute transport maps for RBC and PLT channels.

        Parameters
        ----------
        source_df : DataFrame
            Source impedance data (OutputData.csv format).
        target_df : DataFrame
            Target impedance data (OutputData.csv format).
        source_sample_nos : list of str, optional
            Sample numbers to use from source for fitting.
        target_sample_nos : list of str, optional
            Sample numbers to use from target for fitting.

        Returns
        -------
        self : ImpedanceTransformer
            Fitted transformer.
        """
        from .alignment_1d import fit_gmm_plt, fit_gmm_rbc, sample_impedance_array
        from .gmm_ot import compute_gmm_transport_map

        # Mark standard samples
        source_df = source_df.copy()
        target_df = target_df.copy()

        source_df["IsStandard"] = 1
        target_df["IsStandard"] = 1

        if source_sample_nos is not None:
            source_df["IsStandard"] = (
                source_df["Sample No."].isin(source_sample_nos).astype(int)
            )

        if target_sample_nos is not None:
            target_df["IsStandard"] = (
                target_df["Sample No."].isin(target_sample_nos).astype(int)
            )

        # Get standard samples for GMM fitting
        source_standards = source_df[source_df["IsStandard"] == 1]
        target_standards = target_df[target_df["IsStandard"] == 1]

        print(
            f"Fitting GMMs on {len(source_standards)} source samples and {len(target_standards)} target samples..."
        )

        # Create args object for sample_impedance_array
        class Args:
            gmm_sample_size = self.gmm_sample_size

        args = Args()

        # ========== Fit RBC GMMs ==========
        print("  Fitting RBC GMMs...")
        source_rbc_data = source_standards.filter(like="RBC_RAW_").sum(axis=0)
        target_rbc_data = target_standards.filter(like="RBC_RAW_").sum(axis=0)

        X_source_rbc = sample_impedance_array(
            args, source_rbc_data, random_state=self.random_state
        )
        X_target_rbc = sample_impedance_array(
            args, target_rbc_data, random_state=self.random_state
        )

        self.rbc_source_gmm_ = fit_gmm_rbc(X_source_rbc)
        self.rbc_target_gmm_ = fit_gmm_rbc(X_target_rbc)
        self.rbc_transport_dict_ = compute_gmm_transport_map(
            self.rbc_source_gmm_, self.rbc_target_gmm_
        )

        print(
            f"    Source RBC GMM: {len(X_source_rbc):,} samples, {self.rbc_source_gmm_.n_components} components"
        )
        print(
            f"    Target RBC GMM: {len(X_target_rbc):,} samples, {self.rbc_target_gmm_.n_components} components"
        )

        # ========== Fit PLT GMMs ==========
        print("  Fitting PLT GMMs...")
        source_plt_data = source_standards.filter(like="PLT_RAW_").sum(axis=0)
        target_plt_data = target_standards.filter(like="PLT_RAW_").sum(axis=0)

        X_source_plt = sample_impedance_array(
            args, source_plt_data, random_state=self.random_state
        )
        X_target_plt = sample_impedance_array(
            args, target_plt_data, random_state=self.random_state
        )

        # PLT uses log transform
        X_source_plt = np.log(X_source_plt + 1)
        X_target_plt = np.log(X_target_plt + 1)

        self.plt_source_gmm_ = fit_gmm_plt(X_source_plt)
        self.plt_target_gmm_ = fit_gmm_plt(X_target_plt)
        self.plt_transport_dict_ = compute_gmm_transport_map(
            self.plt_source_gmm_, self.plt_target_gmm_
        )

        print(
            f"    Source PLT GMM: {len(X_source_plt):,} samples, {self.plt_source_gmm_.n_components} components"
        )
        print(
            f"    Target PLT GMM: {len(X_target_plt):,} samples, {self.plt_target_gmm_.n_components} components"
        )

        self.is_fitted_ = True
        print("\n✓ Impedance transformer fitted successfully!")
        print("  GMMs and transport maps computed for RBC and PLT channels")

        return self

    def transform(self, source_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform impedance data using pre-computed GMMs and transport maps.

        Parameters
        ----------
        source_df : DataFrame
            Source impedance data to transform.

        Returns
        -------
        transformed_df : DataFrame
            Transformed impedance data.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "Transformer must be fitted before transform(). Call fit() first."
            )

        from .alignment_1d import sample_impedance_array
        from .gmm_ot import transform_points_gmm

        source_df = source_df.copy()

        # Create args object for sample_impedance_array
        class Args:
            gmm_sample_size = self.gmm_sample_size

        args = Args()

        # Helper function to transform a single row's histogram
        def process_row(row, cols, gmm_source, transport_dict, col_prefix):
            sample = sample_impedance_array(args, row[cols])
            if len(sample) == 0:
                return np.zeros(128)

            # PLT uses log transform
            if col_prefix == "PLT":
                sample = np.log(sample + 1)

            # Transform using pre-computed GMMs and transport map
            sample_transformed = transform_points_gmm(
                sample,
                gmm_source,
                transport_dict,
                transport_method="rand",
                random_state=self.random_state,
            )

            # Inverse log transform for PLT
            if col_prefix == "PLT":
                sample_transformed = np.exp(sample_transformed) - 1

            # Re-bin into histogram
            transformed_hist, _ = np.histogram(
                sample_transformed,
                bins=np.arange(
                    -0.5, 128.5
                ),  # 128 bins with bin centers at integers 0-127
            )
            return transformed_hist

        # ========== Transform RBC ==========
        print(f"Transforming RBC impedance for {len(source_df)} samples...")
        rbc_cols = source_df.filter(like="RBC_RAW_").columns

        rbc_results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_row)(
                source_df.iloc[i],
                rbc_cols,
                self.rbc_source_gmm_,
                self.rbc_transport_dict_,
                "RBC",
            )
            for i in tqdm(range(len(source_df)), desc="Transforming RBC")
        )
        source_df.loc[:, rbc_cols] = np.array(rbc_results)

        # ========== Transform PLT ==========
        print(f"Transforming PLT impedance for {len(source_df)} samples...")
        plt_cols = source_df.filter(like="PLT_RAW_").columns

        plt_results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_row)(
                source_df.iloc[i],
                plt_cols,
                self.plt_source_gmm_,
                self.plt_transport_dict_,
                "PLT",
            )
            for i in tqdm(range(len(source_df)), desc="Transforming PLT")
        )
        source_df.loc[:, plt_cols] = np.array(plt_results)

        print("✓ Transformation complete!")
        return source_df

    def save(self, filepath: str):
        """Save the fitted transformer with GMMs and transport maps."""
        if not self.is_fitted_:
            raise RuntimeError("Cannot save unfitted transformer. Call fit() first.")

        state = {
            "gmm_sample_size": self.gmm_sample_size,
            "n_jobs": self.n_jobs,
            "rbc_source_gmm": self.rbc_source_gmm_,
            "rbc_target_gmm": self.rbc_target_gmm_,
            "plt_source_gmm": self.plt_source_gmm_,
            "plt_target_gmm": self.plt_target_gmm_,
            "rbc_transport_dict": self.rbc_transport_dict_,
            "plt_transport_dict": self.plt_transport_dict_,
        }

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        print(f"Saved transformer to {filepath}")
        print("  Saved: 4 GMMs (RBC source/target, PLT source/target)")
        print("  Saved: 2 transport maps (RBC, PLT)")

    @classmethod
    def load(cls, filepath: str):
        """Load a fitted transformer with GMMs and transport maps."""
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        transformer = cls(
            gmm_sample_size=state["gmm_sample_size"],
            n_jobs=state["n_jobs"],
        )

        transformer.rbc_source_gmm_ = state["rbc_source_gmm"]
        transformer.rbc_target_gmm_ = state["rbc_target_gmm"]
        transformer.plt_source_gmm_ = state["plt_source_gmm"]
        transformer.plt_target_gmm_ = state["plt_target_gmm"]
        transformer.rbc_transport_dict_ = state["rbc_transport_dict"]
        transformer.plt_transport_dict_ = state["plt_transport_dict"]
        transformer.is_fitted_ = True

        print(f"Loaded transformer from {filepath}")
        print("  Loaded: 4 GMMs (RBC source/target, PLT source/target)")
        print("  Loaded: 2 transport maps (RBC, PLT)")
        return transformer


class XNSampleTransformer:
    """
    Transform XN_SAMPLE tabular data using MAD/median-based alignment.

    This transformer aligns tabular blood count parameters between analyzers using
    robust statistics (Median Absolute Deviation and median).

    Parameters
    ----------
    columns : list of str or None, default=None
        List of column names to transform. If None, must be provided during fit().

    Attributes
    ----------
    source_params_ : dict
        Median and MAD parameters for source distribution.
    target_params_ : dict
        Median and MAD parameters for target distribution.
    is_fitted_ : bool
        Whether the transformer has been fitted.

    Examples
    --------
    >>> columns = ['HGB', 'RBC', 'WBC', 'PLT']
    >>> transformer = XNSampleTransformer(columns=columns)
    >>> transformer.fit(source_df, target_df)
    >>> transformed_df = transformer.transform(source_df)
    """

    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns

        self.source_params_ = None
        self.target_params_ = None
        self.is_fitted_ = False

    def fit(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        source_sample_nos: Optional[List[str]] = None,
        target_sample_nos: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
    ):
        """
        Fit the transformation parameters.

        Parameters
        ----------
        source_df : DataFrame
            Source XN_SAMPLE data.
        target_df : DataFrame
            Target XN_SAMPLE data.
        source_sample_nos : list of str, optional
            Sample numbers to use from source for fitting.
        target_sample_nos : list of str, optional
            Sample numbers to use from target for fitting.
        columns : list of str, optional
            Columns to transform (overrides constructor argument).

        Returns
        -------
        self : XNSampleTransformer
            Fitted transformer.
        """
        # Update columns if provided
        if columns is not None:
            self.columns = columns

        if self.columns is None:
            raise ValueError("Must provide columns either in constructor or fit()")

        # Validate columns exist
        missing_source = set(self.columns) - set(source_df.columns)
        missing_target = set(self.columns) - set(target_df.columns)

        if missing_source:
            raise ValueError(f"Columns not found in source_df: {missing_source}")
        if missing_target:
            raise ValueError(f"Columns not found in target_df: {missing_target}")

        # Mark standard samples
        source_df = source_df.copy()
        target_df = target_df.copy()

        source_df["IsStandard"] = 1
        target_df["IsStandard"] = 1

        if source_sample_nos is not None:
            source_df["IsStandard"] = (
                source_df["Sample No."].isin(source_sample_nos).astype(int)
            )

        if target_sample_nos is not None:
            target_df["IsStandard"] = (
                target_df["Sample No."].isin(target_sample_nos).astype(int)
            )

        # Compute parameters
        self.source_params_ = {}
        self.target_params_ = {}

        print("\nFitting transformation parameters:")
        for col in self.columns:
            source_standard_mask = source_df["IsStandard"] == 1
            target_standard_mask = target_df["IsStandard"] == 1

            masked_source = pd.to_numeric(
                source_df.loc[source_standard_mask, col], errors="coerce"
            ).dropna()
            masked_target = pd.to_numeric(
                target_df.loc[target_standard_mask, col], errors="coerce"
            ).dropna()

            self.source_params_[col] = {
                "median": masked_source.median(),
                "mad": mad(masked_source),
            }
            self.target_params_[col] = {
                "median": masked_target.median(),
                "mad": mad(masked_target),
            }

            print(
                f"  {col}: Source (median={self.source_params_[col]['median']:.4f}, "
                f"MAD={self.source_params_[col]['mad']:.4f}) → "
                f"Target (median={self.target_params_[col]['median']:.4f}, "
                f"MAD={self.target_params_[col]['mad']:.4f})"
            )

        self.is_fitted_ = True
        print("XN_SAMPLE transformer fitted!")

        return self

    def transform(self, source_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform XN_SAMPLE data from source to target distribution.

        Parameters
        ----------
        source_df : DataFrame
            Source data to transform.

        Returns
        -------
        transformed_df : DataFrame
            Transformed data.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "Transformer must be fitted before transform(). Call fit() first."
            )

        source_df = source_df.copy()

        print("Transforming columns...")
        for col in tqdm(self.columns):
            # Create mask for numeric values
            numeric_mask = pd.to_numeric(source_df[col], errors="coerce").notna()

            # Store original values
            original_values = source_df[col].copy()

            # Convert and transform numeric values
            source_df.loc[numeric_mask, col] = pd.to_numeric(
                source_df.loc[numeric_mask, col]
            )

            source_df.loc[numeric_mask, col] = transform_nonnormal(
                X=source_df.loc[numeric_mask, col].values,
                median_source=self.source_params_[col]["median"],
                median_target=self.target_params_[col]["median"],
                mad_source=self.source_params_[col]["mad"],
                mad_target=self.target_params_[col]["mad"],
            )

            # Restore non-numeric values
            source_df.loc[~numeric_mask, col] = original_values[~numeric_mask]

        return source_df

    def save(self, filepath: str):
        """Save the fitted transformer."""
        if not self.is_fitted_:
            raise RuntimeError("Cannot save unfitted transformer. Call fit() first.")

        state = {
            "columns": self.columns,
            "source_params": self.source_params_,
            "target_params": self.target_params_,
        }

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        print(f"Saved transformer to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Load a fitted transformer."""
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        transformer = cls(columns=state["columns"])
        transformer.source_params_ = state["source_params"]
        transformer.target_params_ = state["target_params"]
        transformer.is_fitted_ = True

        print(f"Loaded transformer from {filepath}")
        return transformer


__all__ = ["FlowTransformer", "ImpedanceTransformer", "XNSampleTransformer"]
