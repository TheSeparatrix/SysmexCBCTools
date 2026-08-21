"""
XNSampleProcessor - A user-friendly API for processing Sysmex XN_SAMPLE data.

This module provides a scikit-learn style API for cleaning and preprocessing
Sysmex XN_SAMPLE.csv files, making it easy to use in Jupyter notebooks and
Python scripts.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

import pandas as pd

from .memory_optimized import handle_multiple_measurements_optimized
from .processors import (
    analyze_correlations,
    clean_non_numeric_values,
    encode_flags,
    handle_multiple_measurements,
    make_union_drop_column,
    mask_unmeasured_channels,
    normalize_duplicate_columns,
    process_marks,
    remove_clot_in_tube_samples,
    remove_correlated_columns,
    remove_duplicate_columns,
    remove_duplicate_rows,
    remove_empty_columns,
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
    drop_empty_columns : bool, default=True
        Remove columns that are NaN for every row.  Which columns those are
        depends on the channel mix of the cohort, so set this False when the
        column sets of separate runs have to line up.
    use_memory_optimized : bool, default=True
        Use memory-optimized processing for large datasets (>100k rows).
    enable_memory_monitoring : bool, default=True
        Log memory usage throughout processing.
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
        config_path: str | None = None,
        remove_clotintube: bool = True,
        remove_multimeasurementsamples: bool = True,
        remove_correlated: bool = False,
        std_threshold: float = 1.0,
        keep_drop_rows: bool = False,
        make_dummy_marks: bool = False,
        drop_empty_columns: bool = True,
        use_memory_optimized: bool = True,
        enable_memory_monitoring: bool = True,
        output_dir: str = "./output",
        output_prefix: str = "XN_SAMPLE_processed",
        log_to_file: bool = False,
        verbose: int = 1,
    ):
        """Initialize the XNSampleProcessor with processing parameters."""

        processing_params = {
            "remove_clotintube": remove_clotintube,
            "remove_multimeasurementsamples": remove_multimeasurementsamples,
            "remove_correlated": remove_correlated,
            "std_threshold": std_threshold,
            "keep_drop_rows": keep_drop_rows,
            "make_dummy_marks": make_dummy_marks,
            "drop_empty_columns": drop_empty_columns,
            "use_memory_optimized": use_memory_optimized,
            "enable_memory_monitoring": enable_memory_monitoring,
        }

        # A config file, when given, overrides the keyword defaults.
        if config_path is not None:
            self.config = load_config(config_path)
            self.config_path = config_path
            proc = self.config.get("processing", {})
            out = self.config.get("output", {})
            self.output_dir = out.get("directory", output_dir)
            self.output_prefix = out.get("filename_prefix", output_prefix)
        else:
            self.config = None
            self.config_path = None
            proc = {}
            self.output_dir = output_dir
            self.output_prefix = output_prefix

        for name, default in processing_params.items():
            setattr(self, name, proc.get(name, default))

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
        input_files: pd.DataFrame | str | list[str],
        dataset_name: str = "dataset",
        save_output: bool = False,
        *,
        copy_output_data: bool = False,
        copy_sct: bool = False,
        output_data_columns: list[str] | None = None,
        sct_archive_files: str | list[str] | None = None,
        sample_nos: list[str] | None = None,
        output_format: str = "csv",
    ) -> pd.DataFrame:
        """
        Process XN_SAMPLE data from files or a pre-loaded DataFrame.

        Parameters
        ----------
        input_files : pd.DataFrame or str or list of str
            Data to process. Accepts:
            - A ``pd.DataFrame`` (bypasses file loading entirely).
            - A path to a ``.csv``, ``.parquet``, or ``.pq`` file.
            - A list of file paths (may mix formats).
        dataset_name : str, default="dataset"
            Name for this dataset (used in output filenames).
        save_output : bool, default=False
            Whether to save the processed dataframe to disk.
        copy_output_data : bool, default=False
            If True, filter and copy OutputData.csv from each source
            directory into the output directory, keeping only rows that
            match surviving samples.  Requires *input_files* to be file
            paths (not a DataFrame) and *save_output* to be True.
        copy_sct : bool, default=False
            If True, copy matching SCT files from each source directory
            into ``output_dir/SCT/``, consolidating overflow files.
            Requires *input_files* to be file paths (not a DataFrame) and
            *save_output* to be True (unless *sct_archive_files* is
            provided).
        output_data_columns : list of str, optional
            If given, only these columns are written to the filtered
            OutputData CSV.  Useful for reducing file size when only
            metadata and a few measurement columns are needed (e.g.
            ``["Sample No.", "AnalyzeDate", "AnalyzeTime", "RBC", "PLT"]``).
            Ignored when *copy_output_data* is False.
        sct_archive_files : str or list of str, optional
            Paths to consolidated SCT archive files (``.csv``,
            ``.parquet``, or ``.pq``).  When provided with
            ``copy_sct=True``, individual SCT files are reconstructed
            from these archives instead of being copied from source
            directories.  This allows ``input_files`` to be a DataFrame.
        output_format : {"csv", "parquet"}, default="csv"
            Format used when *save_output* is True.  ``"parquet"`` is
            strongly recommended for large datasets (hundreds of thousands
            of rows and above): the resulting file is much smaller and
            is written in a streaming fashion, avoiding the peak memory
            spike of CSV serialisation.
        sample_nos : list of str, optional
            If provided, only rows whose ``"Sample No."`` (after stripping
            leading/trailing whitespace) appears in this list are retained.
            Filtering occurs after the initial file load and merge but before
            any processing steps, reducing both computation and peak memory.
            Each entry is cast to ``str`` before comparison, so integer
            sample numbers (e.g. ``123456``) are handled transparently.
            Works with ``copy_output_data=True`` and ``copy_sct=True``.

        Returns
        -------
        df_processed : pd.DataFrame
            The processed and cleaned dataframe.

        Raises
        ------
        ValueError
            If *copy_output_data* or *copy_sct* (without
            *sct_archive_files*) is True but *input_files* is a
            DataFrame (source directories cannot be derived), or if
            *save_output* is False (no output directory to write to).

        Examples
        --------
        Process a single CSV file:

        >>> processor = XNSampleProcessor()
        >>> df = processor.process_files("data/XN_SAMPLE.csv")

        Process a parquet file:

        >>> df = processor.process_files("data/XN_SAMPLE.parquet")

        Process a pre-loaded DataFrame:

        >>> import pandas as pd
        >>> raw = pd.read_csv("data/XN_SAMPLE.csv")
        >>> df = processor.process_files(raw)

        Process and copy ancillary files:

        >>> df = processor.process_files(
        ...     ["file1.csv", "file2.csv"],
        ...     dataset_name="combined",
        ...     save_output=True,
        ...     copy_output_data=True,
        ...     copy_sct=True,
        ... )

        Reconstruct SCT files from archives:

        >>> df = processor.process_files(
        ...     raw_df,
        ...     save_output=True,
        ...     copy_sct=True,
        ...     sct_archive_files=["archive_WDF.csv", "archive_RET.parquet"],
        ... )
        """
        # Normalise sct_archive_files to list or None
        if isinstance(sct_archive_files, str):
            sct_archive_files = [sct_archive_files]

        # Validate ancillary copy options
        needs_source_dirs = copy_output_data or (
            copy_sct and sct_archive_files is None
        )
        if needs_source_dirs and isinstance(input_files, pd.DataFrame):
            raise ValueError(
                "copy_output_data (and copy_sct without sct_archive_files) "
                "require file paths as input_files, not a DataFrame."
            )
        if (copy_output_data or copy_sct) and not save_output:
            raise ValueError(
                "copy_output_data and copy_sct require save_output=True."
            )

        self.logger.info(f"Processing dataset: {dataset_name}")

        # Dispatch: DataFrame, single path, or list of paths
        if isinstance(input_files, pd.DataFrame):
            df = input_files
            input_file_paths: list[str] | None = None
        else:
            if isinstance(input_files, str):
                input_files = [input_files]
            input_file_paths = list(input_files)
            df = load_dataframes(input_file_paths, self.logger)
        if self.enable_memory_monitoring:
            log_memory_usage(self.logger, f"After loading {dataset_name} dataset")

        # Filter to requested sample numbers before any processing
        if sample_nos is not None:
            stripped_nos = {str(s).strip() for s in sample_nos}
            mask = df["Sample No."].str.strip().isin(stripped_nos)
            n_before = len(df)
            df = df.loc[mask].reset_index(drop=True)
            self.logger.info(
                f"Retained {df['Sample No.'].nunique()} unique sample numbers "
                f"({len(df)} / {n_before} rows) after sample_nos filter"
            )

        # Run processing pipeline
        df = self._process_pipeline(df, dataset_name)

        # Save results if requested
        if save_output:
            fmt = output_format.lower()
            ext_map = {"csv": "csv", "parquet": "parquet", "pq": "parquet"}
            if fmt not in ext_map:
                raise ValueError(
                    f"Unsupported output_format '{output_format}'. "
                    "Use 'csv' or 'parquet'."
                )
            ext = ext_map[fmt]
            now = datetime.now()
            dt_string = now.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_prefix}_{dataset_name}_{dt_string}.{ext}"
            output_path = os.path.join(self.output_dir, filename)
            save_results(df, output_path, self.logger)

        # Copy ancillary files if requested
        if copy_output_data:
            od_filename = f"OutputData_{dataset_name}_{dt_string}.{ext}"
            od_path = os.path.join(self.output_dir, od_filename)
            self.copy_output_data(
                df,
                input_files=input_file_paths,
                output_path=od_path,
                columns=output_data_columns,
            )

        if copy_sct:
            sct_dir = os.path.join(self.output_dir, "SCT")
            self.copy_sct(
                df,
                input_files=input_file_paths,
                sct_archive_files=sct_archive_files,
                output_dir=sct_dir,
            )

        # Store for later access
        self.last_processed_ = df

        return df

    def copy_output_data(
        self,
        keys_source: pd.DataFrame | str | list[str],
        input_files: str | list[str],
        *,
        output_path: str | None = None,
        columns: list[str] | None = None,
        dataset_name: str = "dataset",
    ) -> int:
        """
        Filter ``OutputData.csv`` rows matching a processed XN_SAMPLE dataset.

        The matching keys ``(Sample No., YYYYMMDD_HHMMSS)`` are derived
        from *keys_source* and used to select rows from the
        ``OutputData.csv`` file in each source directory.  The filtered
        rows are streamed to *output_path* (csv or parquet).

        Parameters
        ----------
        keys_source : pd.DataFrame or str or list of str
            Processed XN_SAMPLE-like data used to derive matching keys.
            Accepts a DataFrame, a path to a ``.csv``/``.parquet``/``.pq``
            file, or a list of such paths.  Must contain ``Sample No.``,
            ``Date``, and ``Time`` columns.
        input_files : str or list of str
            Paths to raw XN_SAMPLE files whose parent directories contain
            the source ``OutputData.csv`` files.  Their unique parent
            directories are scanned.
        output_path : str, optional
            Destination file.  If ``None``, a path is built inside
            ``self.output_dir`` using *dataset_name* and a timestamp.
            Extension determines output format (``.csv`` or ``.parquet``).
        columns : list of str, optional
            If given, only these columns are written to the output file.
        dataset_name : str, default="dataset"
            Used to construct the default *output_path* filename when
            *output_path* is not provided.

        Returns
        -------
        n_written : int
            Number of rows written.
        """
        from .ancillary import (
            build_matching_keys,
            derive_source_dirs,
            filter_output_data,
        )

        if isinstance(input_files, str):
            input_files = [input_files]
        input_files = list(input_files)

        keys_df = self._keys_source_to_dataframe(keys_source)
        keys = build_matching_keys(keys_df)
        source_dirs = derive_source_dirs(input_files)

        if output_path is None:
            dt_string = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.output_dir,
                f"OutputData_{dataset_name}_{dt_string}.csv",
            )
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        n_rows = filter_output_data(
            source_dirs, keys, output_path, self.logger, columns=columns,
        )
        self.diagnostic_files_["output_data"] = output_path
        self.logger.info(
            f"Saved {n_rows} filtered OutputData rows to {output_path}"
        )
        return n_rows

    def copy_sct(
        self,
        keys_source: pd.DataFrame | str | list[str],
        *,
        input_files: str | list[str] | None = None,
        sct_archive_files: str | list[str] | None = None,
        output_dir: str | None = None,
    ) -> int:
        """
        Copy or reconstruct SCT files matching a processed XN_SAMPLE dataset.

        Matching keys ``(Sample No., YYYYMMDD_HHMMSS)`` are derived from
        *keys_source*.  SCT files can be obtained either by copying from
        the ``SCT/`` subdirectory of each source directory (when
        *input_files* is given) or by reconstructing individual files from
        consolidated archives (when *sct_archive_files* is given).
        Exactly one of the two must be provided.

        Parameters
        ----------
        keys_source : pd.DataFrame or str or list of str
            Processed XN_SAMPLE-like data used to derive matching keys.
            Accepts a DataFrame, a path to a ``.csv``/``.parquet``/``.pq``
            file, or a list of such paths.  Must contain ``Sample No.``,
            ``Date``, and ``Time`` columns.
        input_files : str or list of str, optional
            Paths to raw XN_SAMPLE files whose parent directories contain
            an ``SCT/`` subdirectory.  Mutually exclusive with
            *sct_archive_files*.
        sct_archive_files : str or list of str, optional
            Paths to consolidated SCT archive files (``.csv``,
            ``.parquet``, or ``.pq``) from which individual SCT files are
            reconstructed.  Mutually exclusive with *input_files*.
        output_dir : str, optional
            Destination directory for SCT files.  Defaults to
            ``{self.output_dir}/SCT``.

        Returns
        -------
        n_written : int
            Number of SCT files written.

        Raises
        ------
        ValueError
            If neither or both of *input_files* and *sct_archive_files*
            are provided.
        """
        from .ancillary import (
            build_matching_keys,
            copy_matching_sct_files,
            derive_source_dirs,
            reconstruct_sct_from_archives,
        )

        if isinstance(sct_archive_files, str):
            sct_archive_files = [sct_archive_files]
        if isinstance(input_files, str):
            input_files = [input_files]

        if (input_files is None) == (sct_archive_files is None):
            raise ValueError(
                "Provide exactly one of input_files or sct_archive_files."
            )

        keys_df = self._keys_source_to_dataframe(keys_source)
        keys = build_matching_keys(keys_df)

        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "SCT")
        os.makedirs(output_dir, exist_ok=True)

        if sct_archive_files is not None:
            n_copied = reconstruct_sct_from_archives(
                list(sct_archive_files), keys, output_dir, self.logger,
            )
        else:
            source_dirs = derive_source_dirs(list(input_files))
            n_copied = copy_matching_sct_files(
                source_dirs, keys, output_dir, self.logger,
            )

        self.diagnostic_files_["sct_dir"] = output_dir
        self.logger.info(f"Copied {n_copied} SCT files to {output_dir}")
        return n_copied

    def _keys_source_to_dataframe(
        self, keys_source: pd.DataFrame | str | list[str],
    ) -> pd.DataFrame:
        """Load a keys source into a DataFrame with only Sample No./Date/Time.

        Only the three columns needed to build matching keys are read from
        disk, keeping peak memory bounded even for very large processed
        XN_SAMPLE files.
        """
        key_cols = ["Sample No.", "Date", "Time"]

        if isinstance(keys_source, pd.DataFrame):
            return keys_source

        if isinstance(keys_source, str):
            paths = [keys_source]
        else:
            paths = list(keys_source)

        frames: list[pd.DataFrame] = []
        for p in paths:
            suffix = os.path.splitext(p)[1].lower()
            if suffix == ".csv":
                frames.append(
                    pd.read_csv(
                        p,
                        usecols=key_cols,
                        encoding="ISO-8859-1",
                        dtype=str,
                    )
                )
            elif suffix in {".parquet", ".pq"}:
                try:
                    frames.append(pd.read_parquet(p, columns=key_cols))
                except ImportError:
                    raise ImportError(
                        "Reading parquet files requires pyarrow. "
                        "Install it with: pip install pyarrow"
                    )
            else:
                raise ValueError(
                    f"Unsupported file extension '{suffix}' for {p}. "
                    "Supported: .csv, .parquet, .pq"
                )
            self.logger.info(
                f"Loaded {len(frames[-1])} key rows from {p}"
            )
        return pd.concat(frames, axis=0, ignore_index=True)

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

        # Make duplicated column labels unique, so every column-by-column
        # step below sees a Series rather than a two-column frame.
        df = normalize_duplicate_columns(df, self.logger)

        # Remove unused columns
        df = remove_unused_columns(df, self.logger)

        # Encode flags
        df = encode_flags(df, self.logger)

        # Blank the channels the Discrete order never asked for.  Must run
        # after encode_flags (whose 0-fill would otherwise re-fabricate the
        # flags) and before everything below, which would otherwise compute on
        # the analyser's zero fill.
        df = mask_unmeasured_channels(df, self.logger)

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
                    df, self.logger, self.std_threshold, self.keep_drop_rows
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

        # Drop columns left empty for every row
        if self.drop_empty_columns:
            df = remove_empty_columns(df, self.logger)

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

    def get_last_processed(self) -> pd.DataFrame | None:
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
