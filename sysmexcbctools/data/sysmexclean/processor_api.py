"""
XNSampleProcessor - A user-friendly API for processing Sysmex XN_SAMPLE data.

This module provides a scikit-learn style API for cleaning and preprocessing
Sysmex XN_SAMPLE.csv files, making it easy to use in Jupyter notebooks and
Python scripts.
"""

import logging
import os
from datetime import datetime
from typing import List, Optional, Union

import pandas as pd

from .memory_optimized import handle_multiple_measurements_optimized
from .processors import (
    analyze_correlations,
    clean_non_numeric_values,
    encode_flags,
    handle_multiple_measurements,
    make_union_drop_column,
    process_discrete_columns,
    process_marks,
    remove_clot_in_tube_samples,
    remove_correlated_columns,
    remove_duplicate_columns,
    remove_duplicate_rows,
    remove_redundant_columns,
    remove_technical_samples,
    remove_unused_columns,
)
from .utils import (
    convert_to_numeric,
    load_config,
    load_dataframes,
    log_memory_usage,
    save_results,
)


class XNSampleProcessor:
    """
    Processor for cleaning and standardizing Sysmex XN_SAMPLE.csv files.

    This class provides a simple interface for processing raw Sysmex XN_SAMPLE
    data exported from decrypted .116 files. It handles consolidation of datasets
    from multiple decryptions and performs comprehensive data cleaning.

    Parameters
    ----------
    config_path : str, optional
        Path to YAML configuration file. If provided, all other parameters
        are loaded from this file.
    remove_clotintube : bool, default=True
        Remove samples with indicators of clot in tube (turbidity, agglutination,
        PLT clumps).
    remove_multimeasurementsamples : bool, default=True
        Handle multiple measurements per sample. Similar measurements are
        consolidated, keeping the earliest. Dissimilar measurements are saved
        for manual review.
    remove_correlated : bool, default=False
        Remove columns highly correlated (>= 0.8) with standard FBC features.
        NOT RECOMMENDED as it may remove useful data.
    std_threshold : float, default=1.0
        Standard deviation threshold for comparing multiple measurements.
        Measurements differing by more than this many SDs are flagged.
    keep_drop_rows : bool, default=False
        If True, rows are not actually removed. Instead, columns prefixed
        with 'drop_' indicate which rows would be dropped and why.
    make_dummy_marks : bool, default=False
        If True, data mark fields (ending in "/M") are one-hot encoded
        into multiple columns.
    use_memory_optimized : bool, default=True
        Use memory-optimized processing for large datasets (>100k rows).
    enable_memory_monitoring : bool, default=True
        Log memory usage throughout processing.
    correlation_sample_size : int, default=50000
        Maximum rows to use for correlation analysis (reduces memory usage).
    chunk_size : int, default=1000
        Number of sample IDs to process at once in chunked operations.
    force_dask : bool, default=False
        Force use of Dask for multiple measurements processing (for testing).
    output_dir : str, default="./output"
        Directory for output files.
    output_prefix : str, default="XN_SAMPLE_processed"
        Prefix for output filenames.
    log_to_file : bool, default=False
        If True, write logs to file and save diagnostic CSV files
        (odd measurements, correlation analysis, etc.) in output_dir/.
        If False, only log to console and skip diagnostic file creation.
    verbose : int, default=1
        Verbosity level. 0 = silent, 1 = info, 2 = debug.

    Attributes
    ----------
    logger : logging.Logger
        Logger for processing operations.
    last_processed_ : pd.DataFrame or None
        The most recently processed dataframe.
    diagnostic_files_ : dict
        Paths to diagnostic files generated during processing.

    Examples
    --------
    Basic usage with default settings:

    >>> from sysmexcbctools.data import XNSampleProcessor
    >>> processor = XNSampleProcessor()
    >>> df_clean = processor.process_files("path/to/XN_SAMPLE.csv")

    Using custom parameters:

    >>> processor = XNSampleProcessor(
    ...     remove_clotintube=True,
    ...     std_threshold=1.5,
    ...     output_dir="./results"
    ... )
    >>> df_clean = processor.process_files(
    ...     input_files=["file1.csv", "file2.csv"],
    ...     dataset_name="my_cohort"
    ... )

    Using a config file:

    >>> processor = XNSampleProcessor(config_path="config.yaml")
    >>> df_clean = processor.process(dataset_name="INTERVAL")

    Processing without saving to disk:

    >>> df_clean = processor.process_files(
    ...     input_files=["data.csv"],
    ...     save_output=False
    ... )
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        remove_clotintube: bool = True,
        remove_multimeasurementsamples: bool = True,
        remove_correlated: bool = False,
        std_threshold: float = 1.0,
        keep_drop_rows: bool = False,
        make_dummy_marks: bool = False,
        use_memory_optimized: bool = True,
        enable_memory_monitoring: bool = True,
        correlation_sample_size: int = 50000,
        chunk_size: int = 1000,
        force_dask: bool = False,
        output_dir: str = "./output",
        output_prefix: str = "XN_SAMPLE_processed",
        log_to_file: bool = False,
        verbose: int = 1,
    ):
        """Initialize the XNSampleProcessor with processing parameters."""

        # If config file provided, load it and override defaults
        if config_path is not None:
            self.config = load_config(config_path)
            self.config_path = config_path

            # Extract processing parameters from config
            proc = self.config.get("processing", {})
            self.remove_clotintube = proc.get("remove_clotintube", remove_clotintube)
            self.remove_multimeasurementsamples = proc.get(
                "remove_multimeasurementsamples", remove_multimeasurementsamples
            )
            self.remove_correlated = proc.get("remove_correlated", remove_correlated)
            self.std_threshold = proc.get("std_threshold", std_threshold)
            self.keep_drop_rows = proc.get("keep_drop_rows", keep_drop_rows)
            self.make_dummy_marks = proc.get("make_dummy_marks", make_dummy_marks)
            self.use_memory_optimized = proc.get(
                "use_memory_optimized", use_memory_optimized
            )
            self.enable_memory_monitoring = proc.get(
                "enable_memory_monitoring", enable_memory_monitoring
            )
            self.correlation_sample_size = proc.get(
                "correlation_sample_size", correlation_sample_size
            )
            self.chunk_size = proc.get("chunk_size", chunk_size)
            self.force_dask = proc.get("force_dask", force_dask)

            # Extract output parameters from config
            out = self.config.get("output", {})
            self.output_dir = out.get("directory", output_dir)
            self.output_prefix = out.get("filename_prefix", output_prefix)
        else:
            # Use provided parameters
            self.config = None
            self.config_path = None
            self.remove_clotintube = remove_clotintube
            self.remove_multimeasurementsamples = remove_multimeasurementsamples
            self.remove_correlated = remove_correlated
            self.std_threshold = std_threshold
            self.keep_drop_rows = keep_drop_rows
            self.make_dummy_marks = make_dummy_marks
            self.use_memory_optimized = use_memory_optimized
            self.enable_memory_monitoring = enable_memory_monitoring
            self.correlation_sample_size = correlation_sample_size
            self.chunk_size = chunk_size
            self.force_dask = force_dask
            self.output_dir = output_dir
            self.output_prefix = output_prefix

        self.log_to_file = log_to_file
        self.verbose = verbose

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up logging
        self.logger = self._setup_logger()

        # Attributes set during processing
        self.last_processed_ = None
        self.diagnostic_files_ = {}

    def _setup_logger(self) -> logging.Logger:
        """Set up logger for processing operations."""
        # Create logger
        logger = logging.getLogger(f"XNSampleProcessor_{id(self)}")
        logger.setLevel(logging.DEBUG if self.verbose >= 2 else logging.INFO)

        # Clear any existing handlers
        if logger.hasHandlers():
            logger.handlers.clear()

        # Create formatters
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")

        # Console handler
        if self.verbose > 0:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO if self.verbose == 1 else logging.DEBUG)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # File handler (optional)
        if self.log_to_file:
            # Create logs directory
            logdir = os.path.join(self.output_dir, "logs")
            os.makedirs(logdir, exist_ok=True)

            # Get timestamp for log filename
            now = datetime.now()
            dt_string = now.strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(logdir, f"XN_SAMPLE_processing_{dt_string}.log")

            file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

            logger.info(f"Logging initialized. Detailed logs: {log_file}")
        else:
            logger.info("Logging initialized (console only)")

        return logger

    def process_files(
        self,
        input_files: Union[str, List[str]],
        dataset_name: str = "dataset",
        save_output: bool = False,
    ) -> pd.DataFrame:
        """
        Process XN_SAMPLE CSV files.

        Parameters
        ----------
        input_files : str or list of str
            Path(s) to XN_SAMPLE.csv file(s) to process.
        dataset_name : str, default="dataset"
            Name for this dataset (used in output filenames).
        save_output : bool, default=False
            Whether to save the processed dataframe to disk.

        Returns
        -------
        df_processed : pd.DataFrame
            The processed and cleaned dataframe.

        Examples
        --------
        >>> processor = XNSampleProcessor()
        >>> df = processor.process_files("data/XN_SAMPLE.csv")
        >>> df = processor.process_files(
        ...     ["file1.csv", "file2.csv"],
        ...     dataset_name="combined",
        ...     save_output=True
        ... )
        """
        # Convert single file to list
        if isinstance(input_files, str):
            input_files = [input_files]

        self.logger.info(f"Processing dataset: {dataset_name}")

        # Load dataframes
        df = load_dataframes(input_files, self.logger)
        if self.enable_memory_monitoring:
            log_memory_usage(self.logger, f"After loading {dataset_name} dataset")

        # Run processing pipeline
        df = self._process_pipeline(df, dataset_name)

        # Save results if requested
        if save_output:
            now = datetime.now()
            dt_string = now.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_prefix}_{dataset_name}_{dt_string}.csv"
            output_path = os.path.join(self.output_dir, filename)
            save_results(df, output_path, self.logger)

        # Store for later access
        self.last_processed_ = df

        return df

    def process(self, dataset_name: str) -> pd.DataFrame:
        """
        Process a dataset defined in the configuration file.

        This method requires that the processor was initialized with a
        config_path parameter.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset defined in the config file.

        Returns
        -------
        df_processed : pd.DataFrame
            The processed and cleaned dataframe.

        Raises
        ------
        ValueError
            If no config file was provided at initialization.

        Examples
        --------
        >>> processor = XNSampleProcessor(config_path="config.yaml")
        >>> df = processor.process("INTERVAL")
        """
        if self.config is None:
            raise ValueError(
                "No config file provided. Use process_files() instead, "
                "or initialize with config_path parameter."
            )

        # Find dataset in config
        dataset_config = None
        for ds in self.config["input"]["datasets"]:
            if ds["name"] == dataset_name:
                dataset_config = ds
                break

        if dataset_config is None:
            raise ValueError(f"Dataset '{dataset_name}' not found in config file")

        # Process the dataset
        return self.process_files(
            input_files=dataset_config["files"],
            dataset_name=dataset_name,
            save_output=True,
        )

    def _process_pipeline(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Run the complete processing pipeline on a dataframe.

        This is an internal method that applies all processing steps
        in sequence.
        """
        # Remove duplicate rows
        df = remove_duplicate_rows(df, self.logger)

        # Remove technical samples
        df = remove_technical_samples(df, self.logger, self.keep_drop_rows)

        df = df.reset_index(drop=True)

        # Process discrete columns
        df = process_discrete_columns(df, self.logger)

        # Remove unused columns
        df = remove_unused_columns(df, self.logger)

        # Encode flags
        df = encode_flags(df, self.logger)

        # Encode data marks
        df = process_marks(df, self.logger, self.make_dummy_marks)

        # Remove duplicate columns
        df = remove_duplicate_columns(df, self.logger)

        # Remove clotted samples if requested
        if self.remove_clotintube:
            df = remove_clot_in_tube_samples(df, self.logger, self.keep_drop_rows)

        # Handle multiple measurements if requested
        if self.remove_multimeasurementsamples:
            if self.use_memory_optimized:
                df, odd_samples_df, one_different_df = handle_multiple_measurements_optimized(
                    df, self.logger, self.std_threshold,
                    self.keep_drop_rows, use_dask=self.force_dask
                )
            else:
                df, odd_samples_df, one_different_df = handle_multiple_measurements(
                    df, self.logger, self.std_threshold, self.keep_drop_rows
                )

            # Save diagnostic files (only if log_to_file is enabled)
            if self.log_to_file:
                if not odd_samples_df.empty:
                    odd_path = os.path.join(
                        self.output_dir, f"{dataset_name}_oddmultiplemeasurements.csv"
                    )
                    odd_samples_df.to_csv(odd_path, index=False)
                    self.diagnostic_files_["odd_measurements"] = odd_path
                    self.logger.info(f"Saved odd measurements to {odd_path}")

                if not one_different_df.empty:
                    one_diff_path = os.path.join(
                        self.output_dir, f"{dataset_name}_onlyonedifferentmeasurement.csv"
                    )
                    one_different_df.to_csv(one_diff_path, index=False)
                    self.diagnostic_files_["one_different"] = one_diff_path
                    self.logger.info(f"Saved one-different measurements to {one_diff_path}")
            else:
                # Still log the counts even if not saving files
                if not odd_samples_df.empty:
                    self.logger.info(f"Found {len(odd_samples_df)} samples with discrepant multiple measurements (not saved)")
                if not one_different_df.empty:
                    self.logger.info(f"Found {len(one_different_df)} samples with one different measurement (not saved)")

        # Clean non-numeric values
        df = clean_non_numeric_values(df, self.logger)

        # Remove redundant columns
        df = remove_redundant_columns(df, self.logger)

        # Convert to numeric
        df = convert_to_numeric(df, self.logger)

        # Analyze correlations
        correlated_columns = analyze_correlations(
            df, self.logger, self.output_dir, dataset_name, save_file=self.log_to_file
        )

        # Remove correlated columns if requested
        if self.remove_correlated:
            df = remove_correlated_columns(df, correlated_columns, self.logger)

        # Final reporting
        if self.keep_drop_rows:
            df = make_union_drop_column(df)
            self.logger.info(
                f"Final number of rows to be dropped: {df['drop'].sum()} / {len(df)}"
            )
        else:
            self.logger.info(f"Final dataframe shape: {df.shape}")
            self.logger.info(f"Final number of unique samples: {df['Sample No.'].nunique()}")

        return df

    def get_last_processed(self) -> Optional[pd.DataFrame]:
        """
        Get the most recently processed dataframe.

        Returns
        -------
        df : pd.DataFrame or None
            The last processed dataframe, or None if nothing has been processed yet.
        """
        return self.last_processed_

    def get_diagnostic_files(self) -> dict:
        """
        Get paths to diagnostic files generated during processing.

        Returns
        -------
        diagnostic_files : dict
            Dictionary mapping diagnostic file types to their paths.
        """
        return self.diagnostic_files_.copy()
