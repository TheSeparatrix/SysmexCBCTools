import gc
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import psutil
import yaml
from tqdm import tqdm


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


def load_dataframes(file_paths, logger):
    """Load and concatenate dataframes from multiple CSV files."""
    logger.info(f"Loading {len(file_paths)} CSV files")

    dfs = []
    for file in tqdm(file_paths):
        try:
            df = pd.read_csv(file, encoding="ISO-8859-1", low_memory=False)
            dfs.append(df)
            logger.info(f"Successfully loaded {file} with {df.shape[0]} rows")
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")

    if not dfs:
        logger.error("No valid dataframes loaded")
        raise ValueError("No valid dataframes loaded")

    df = pd.concat(dfs, axis=0).reset_index(drop=True)
    logger.info(f"Combined dataframe shape: {df.shape}")

    return df


def save_results(df, output_path, logger):
    """Save processed dataframe to CSV."""
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Saving processed data to {output_path}")
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {df.shape[0]} rows and {df.shape[1]} columns")

    return output_path


def convert_to_numeric(df, logger):
    """Convert all columns to numeric where possible."""
    logger.info("Converting all columns to numeric")

    for col in df.columns:
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


def log_memory_usage(logger, step_name):
    """Log current memory usage and trigger garbage collection."""
    memory_before = get_memory_usage()
    gc.collect()
    memory_after = get_memory_usage()
    logger.info(f"{step_name} - Memory usage: {memory_after:.2f}GB (freed {memory_before-memory_after:.2f}GB)")
    return memory_after


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
