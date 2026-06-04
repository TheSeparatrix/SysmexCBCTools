import gc
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import yaml
from tqdm import tqdm

from .constants import ID_COLUMNS


def setup_logging(config):
    """Set up logging configuration with both file and console outputs."""
    # Make logs directory if it doesn't exist
    output_dir = config["output"]["directory"]
    logdir = output_dir + "/logs"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Get string of time and date for log filename
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    log_file = f"{logdir}/XN_SAMPLE_processing_{dt_string}.log"

    # Get the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Capture all levels

    # Clear any existing handlers (prevents duplicate logs)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatters
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler (for standard output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Less verbose for console
    console_handler.setFormatter(console_formatter)

    # File handler (more detailed logging)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # More detailed for file
    file_handler.setFormatter(file_formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Logging initialized. Detailed logs will be saved to {log_file}")

    return logger


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path) as file:
        return yaml.safe_load(file)


_SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".pq"}


def _read_single_file(path, logger):
    """
    Read a single data file, dispatching on file extension.

    Parameters
    ----------
    path : str or Path
        Path to a ``.csv``, ``.parquet``, or ``.pq`` file.
    logger : logging.Logger
        Logger for status messages.

    Returns
    -------
    df : pd.DataFrame

    Raises
    ------
    ValueError
        If the file extension is not supported.
    ImportError
        If pyarrow is not installed and a parquet file is requested.
    """
    suffix = Path(path).suffix.lower()

    # Identifier columns are always read as strings so that purely numeric
    # cohorts are not silently inferred as int/float (see ID_COLUMNS).
    if suffix == ".csv":
        return pd.read_csv(
            path,
            encoding="ISO-8859-1",
            low_memory=False,
            dtype={col: str for col in ID_COLUMNS},
        )

    if suffix in {".parquet", ".pq"}:
        try:
            df = pd.read_parquet(path)
        except ImportError:
            raise ImportError(
                "Reading parquet files requires pyarrow. "
                "Install it with: pip install pyarrow"
            )
        for col in ID_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype(str)
        return df

    raise ValueError(
        f"Unsupported file extension '{suffix}' for {path}. "
        f"Supported formats: {sorted(_SUPPORTED_EXTENSIONS)}"
    )


def load_dataframes(file_paths, logger):
    """
    Load and concatenate dataframes from multiple data files.

    Supports ``.csv``, ``.parquet``, and ``.pq`` files (may be mixed).

    Parameters
    ----------
    file_paths : list of str
        Paths to data files.
    logger : logging.Logger
        Logger for status messages.

    Returns
    -------
    df : pd.DataFrame
        Concatenated dataframe from all files.

    Raises
    ------
    ValueError
        If no valid dataframes could be loaded, or if a file has an
        unsupported extension.
    ImportError
        If pyarrow is not installed and a parquet file is requested.
    """
    logger.info(f"Loading {len(file_paths)} file(s)")

    dfs = []
    for file in tqdm(file_paths):
        try:
            df = _read_single_file(file, logger)
            dfs.append(df)
            logger.info(f"Successfully loaded {file} with {df.shape[0]} rows")
        except (ImportError, ValueError):
            raise
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")

    if not dfs:
        logger.error("No valid dataframes loaded")
        raise ValueError("No valid dataframes loaded")

    df = pd.concat(dfs, axis=0).reset_index(drop=True)
    del dfs  # free individual DataFrames while concatenated copy exists
    logger.info(f"Combined dataframe shape: {df.shape}")

    return df


def save_results(df, output_path, logger):
    """Save the processed dataframe to disk.

    Output format is inferred from the file extension:

    - ``.csv`` -> pandas ``to_csv``
    - ``.parquet`` / ``.pq`` -> pandas ``to_parquet`` (requires pyarrow)

    Parquet is strongly recommended for large datasets: the resulting file
    is much smaller and is written/read in a streaming fashion, avoiding
    the multi-GB peak memory spike that CSV serialisation of millions of
    rows can cause.

    Parameters
    ----------
    df : pd.DataFrame
        Processed dataframe to save.
    output_path : str
        Destination path. The extension determines the output format.
    logger : logging.Logger
        Logger for status messages.

    Returns
    -------
    output_path : str
        The path that was written.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    ImportError
        If ``pyarrow`` is required but not installed.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    suffix = Path(output_path).suffix.lower()

    logger.info(f"Saving processed data to {output_path}")
    if suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif suffix in {".parquet", ".pq"}:
        try:
            df.to_parquet(output_path, index=False)
        except ImportError:
            raise ImportError(
                "Writing parquet files requires pyarrow. "
                "Install it with: pip install pyarrow"
            )
    else:
        raise ValueError(
            f"Unsupported output extension '{suffix}' for {output_path}. "
            f"Supported: .csv, .parquet, .pq"
        )
    logger.info(f"Saved {df.shape[0]} rows and {df.shape[1]} columns")

    return output_path


def convert_to_numeric(df, logger):
    """Convert all columns to numeric where possible."""
    logger.info("Converting all columns to numeric")

    for col in df.columns:
        # Never coerce identifier columns: a numeric sample number must stay
        # a string so it matches OutputData/SCT keys (see ID_COLUMNS).
        if col in ID_COLUMNS:
            continue
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            logger.info(f"Could not convert column {col} to numeric")

    return df

def get_drop_cols(df):
    return [col for col in df.columns if col.startswith('drop')]


def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)


_last_memory_checkpoint: float | None = None


def log_memory_usage(logger, step_name):
    """Log current memory usage and change since the last checkpoint.

    Calls ``gc.collect()`` first so that unreferenced objects are freed
    before measuring, then reports the delta since the previous call.

    The old implementation only measured how much ``gc.collect()`` freed
    at that instant, which was almost always 0 because pandas DataFrames
    are freed by CPython reference counting, not the cyclic GC.  It also
    could not capture any freeing at all when the *caller* still held a
    reference to the old DataFrame (which is the normal case for every
    ``df = some_step(df, ...)`` call in ``_process_pipeline``).
    """
    global _last_memory_checkpoint
    gc.collect()
    current = get_memory_usage()
    if _last_memory_checkpoint is not None:
        delta = current - _last_memory_checkpoint
        sign = "+" if delta >= 0 else ""
        logger.info(
            f"{step_name} - Memory: {current:.2f}GB "
            f"({sign}{delta:.2f}GB since last checkpoint)"
        )
    else:
        logger.info(f"{step_name} - Memory: {current:.2f}GB")
    _last_memory_checkpoint = current
    return current


def chunked_correlation(df, chunk_size=10000, logger=None):
    """
    Calculate correlation matrix using chunked processing to handle large datasets.
    This is memory-efficient for large dataframes.
    """
    if logger:
        logger.info(f"Computing correlations in chunks of {chunk_size} rows")

    # Get numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    n_rows, n_cols = numeric_df.shape

    if logger:
        logger.info(f"Computing correlation matrix for {n_cols} numeric columns")

    # For very large datasets, sample if needed to prevent memory issues
    if n_rows > 100000 and n_cols > 300:
        sample_size = min(50000, n_rows)
        if logger:
            logger.warning(f"Large dataset detected ({n_rows} rows, {n_cols} cols). "
                          f"Sampling {sample_size} rows for correlation analysis")
        numeric_df = numeric_df.sample(n=sample_size, random_state=42)

    # Calculate correlation matrix
    try:
        correlation_matrix = numeric_df.corr()
        if logger:
            logger.info(f"Successfully computed {correlation_matrix.shape[0]}x{correlation_matrix.shape[1]} correlation matrix")
        return correlation_matrix
    except MemoryError as e:
        if logger:
            logger.error(f"Memory error during correlation calculation: {e}")
            logger.info("Trying with smaller sample size")
        # Fallback to smaller sample
        sample_size = min(10000, n_rows)
        numeric_df = numeric_df.sample(n=sample_size, random_state=42)
        correlation_matrix = numeric_df.corr()
        if logger:
            logger.warning(f"Computed correlation matrix using reduced sample of {sample_size} rows")
        return correlation_matrix
