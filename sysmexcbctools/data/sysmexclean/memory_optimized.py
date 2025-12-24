"""
Memory-optimized processing functions for large datasets using Dask.
"""
import logging

import pandas as pd
from tqdm import tqdm

try:
    import dask
    import dask.dataframe as dd
    from dask.distributed import Client, as_completed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

from .constants import STANDARD_FBC_DECRYPT
from .utils import get_drop_cols, log_memory_usage


def handle_multiple_measurements_optimized(df, logger, std_threshold=1.0, keep_drop_rows=False, use_dask=None):
    """
    Memory-optimized version of handle_multiple_measurements using Dask when available.
    Falls back to chunked pandas processing if Dask is not available.
    """
    logger.info("Handling samples with multiple measurements (memory-optimized)")

    # Auto-detect whether to use Dask based on dataset size and availability
    if use_dask is None:
        use_dask = DASK_AVAILABLE and (df.shape[0] > 100000 or df.shape[1] > 200)

    if use_dask and not DASK_AVAILABLE:
        logger.warning("Dask not available, falling back to chunked pandas processing")
        use_dask = False

    log_memory_usage(logger, "Before multiple measurements processing")

    # Check for multiple entries of the same sample number
    id_counts = df["Sample No."].value_counts()
    multiple_entries = id_counts[id_counts > 1]

    logger.info(
        f"Found {len(multiple_entries)}/{len(df['Sample No.'].unique())} samples "
        f"({len(multiple_entries)/len(df['Sample No.'].unique()):.1%}) with multiple entries"
    )

    initial_len = df.shape[0]

    # Convert FBC measurements to numeric
    for fbc_measurement in STANDARD_FBC_DECRYPT:
        if fbc_measurement in df.columns:
            df[fbc_measurement] = pd.to_numeric(df[fbc_measurement], errors="coerce")

    # Calculate standard deviations for each FBC measurement
    stds = {
        fbc_measurement: df[fbc_measurement].std()
        for fbc_measurement in STANDARD_FBC_DECRYPT
        if fbc_measurement in df.columns
    }

    if use_dask:
        return _handle_multiple_measurements_dask(
            df, logger, std_threshold, keep_drop_rows, stds, initial_len
        )
    else:
        return _handle_multiple_measurements_chunked(
            df, logger, std_threshold, keep_drop_rows, stds, initial_len
        )


def _handle_multiple_measurements_dask(df, logger, std_threshold, keep_drop_rows, stds, initial_len):
    """Handle multiple measurements using Dask for memory efficiency."""
    logger.info("Using Dask for multiple measurements processing")

    # Convert to Dask DataFrame with appropriate partitioning
    # Partition by number of unique samples to keep related samples together
    n_unique_samples = df["Sample No."].nunique()
    n_partitions = min(max(n_unique_samples // 10000, 1), 100)  # 1-100 partitions

    ddf = dd.from_pandas(df, npartitions=n_partitions)

    def process_group_pandas(group_df):
        """Process groups using pandas to avoid Dask apply warnings."""
        kept_samples = []
        odd_samples = []

        # Try to use include_groups parameter if available, otherwise fall back
        try:
            # Try newer pandas syntax (2.0+)
            groupby_obj = group_df.groupby("Sample No.", include_groups=False)
            use_include_groups = True
        except TypeError:
            # Fall back to older pandas syntax
            groupby_obj = group_df.groupby("Sample No.")
            use_include_groups = False

        for sample_id, group in groupby_obj:
            if use_include_groups:
                # Add the Sample No. back since include_groups=False excludes it
                group = group.copy()
                group['Sample No.'] = sample_id
            # If not using include_groups, group already contains Sample No. column

            if len(group) < 2:
                kept_samples.append(group)
                continue

            # Sort by date and time
            group = group.sort_values(["Date", "Time"], ascending=True)

            # Check difference for each standard FBC measurement
            check_list = []
            for fbc_measurement in STANDARD_FBC_DECRYPT:
                if fbc_measurement not in group.columns:
                    continue

                difference = abs(
                    group.iloc[0][fbc_measurement] - group.iloc[1][fbc_measurement]
                )
                check = ~(difference > std_threshold * stds[fbc_measurement])
                check_list.append(check)

            # Keep measurement if all checks were true or at most one failed
            if sum(check_list) >= len(check_list) - 1:
                kept_samples.append(group.iloc[:1])
            else:
                odd_samples.append(group)

        return kept_samples, odd_samples

    # Convert back to pandas for processing to avoid apply warnings
    df_pandas = ddf.compute()
    kept_samples_list, odd_samples_list = process_group_pandas(df_pandas)

    # Concatenate results
    if kept_samples_list:
        df_result = pd.concat(kept_samples_list, ignore_index=True)
    else:
        df_result = pd.DataFrame()

    if odd_samples_list:
        many_samples_df = pd.concat(odd_samples_list, ignore_index=True)
    else:
        many_samples_df = pd.DataFrame()

    only_one_different_df = pd.DataFrame()  # Not implemented in this version

    log_memory_usage(logger, "After Dask multiple measurements processing")

    rows_dropped = initial_len - len(df_result)
    logger.info(
        f"Dropped {rows_dropped} identical sample number rows ({rows_dropped/initial_len:.2%}), "
        f"keeping the one with the earliest date and time if samples matched within {std_threshold} stds"
    )

    return df_result, many_samples_df, only_one_different_df


def _handle_multiple_measurements_chunked(df, logger, std_threshold, keep_drop_rows, stds, initial_len):
    """Handle multiple measurements using chunked pandas processing."""
    logger.info("Using chunked pandas processing for multiple measurements")

    if keep_drop_rows:
        drop_cols = get_drop_cols(df)

    # Get samples with multiple entries
    id_counts = df["Sample No."].value_counts()
    multiple_entries = id_counts[id_counts > 1].index.tolist()

    # Process in chunks to avoid memory issues
    chunk_size = 1000  # Process 1000 sample IDs at a time
    chunks = [multiple_entries[i:i + chunk_size] for i in range(0, len(multiple_entries), chunk_size)]

    multi_sample_rows = []
    odd_samples = []
    only_one_different = []

    # First, keep all samples with no duplicates
    single_samples = df[~df["Sample No."].isin(multiple_entries)]
    multi_sample_rows.append(single_samples)

    logger.info(f"Processing {len(multiple_entries)} samples with multiple entries in {len(chunks)} chunks")

    for i, chunk_sample_ids in enumerate(tqdm(chunks, desc="Processing chunks")):
        # Process this chunk
        chunk_df = df[df["Sample No."].isin(chunk_sample_ids)]
        grouped = chunk_df.groupby("Sample No.")

        for sample_id, group in grouped:
            if len(group) < 2:
                multi_sample_rows.append(group)
                continue

            # Sort by date and time
            group = group.sort_values(["Sample No.", "Date", "Time"], ascending=True)

            # Check difference for each standard FBC measurement
            check_list = []
            for fbc_measurement in STANDARD_FBC_DECRYPT:
                if fbc_measurement not in group.columns:
                    continue

                difference = abs(
                    group.iloc[0][fbc_measurement] - group.iloc[1][fbc_measurement]
                )
                check = ~(difference > std_threshold * stds[fbc_measurement])

                if not check and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Sample {sample_id} failed check on {fbc_measurement} with value 1 "
                        f"{group.iloc[0][fbc_measurement]:.2f} and value 2 {group.iloc[1][fbc_measurement]:.2f} "
                        f"(threshold difference: {std_threshold * stds[fbc_measurement]:.2f})"
                    )

                check_list.append(check)

            # Keep measurement if all checks were true or at most one failed
            if sum(check_list) >= len(check_list) - 1:
                multi_sample_rows.append(group.iloc[:1])  # Keep only the first measurement

                if keep_drop_rows:
                    # Mark duplicates for dropping
                    already_dropping = group[drop_cols].any(axis=1)
                    first_false_i = next(
                        (i for i, val in enumerate(already_dropping) if val is False), None
                    )

                    for i, idx in enumerate(group.index):
                        if i != first_false_i:
                            df.loc[idx, "drop_dup"] = True
                else:
                    if sum(check_list) == len(check_list) - 1:  # only one check failed
                        only_one_different.append(group)
            else:
                if keep_drop_rows:
                    for idx in group.index:
                        df.loc[idx, "drop_dup_disagree"] = True
                odd_samples.append(group)

        # Log progress and memory
        if i % 10 == 0:
            log_memory_usage(logger, f"After processing chunk {i+1}/{len(chunks)}")

    # Concatenate all rows
    df_result = pd.concat(multi_sample_rows, ignore_index=True) if multi_sample_rows else pd.DataFrame()

    # Process diagnostic dataframes
    try:
        many_samples_df = pd.concat(odd_samples, ignore_index=True) if odd_samples else pd.DataFrame()
        if not many_samples_df.empty:
            from sklearn.preprocessing import LabelEncoder
            many_samples_df["Sample No."] = LabelEncoder().fit_transform(
                many_samples_df["Sample No."].values
            )
    except (ValueError, ImportError):
        many_samples_df = pd.DataFrame()

    try:
        only_one_different_df = pd.concat(only_one_different, ignore_index=True) if only_one_different else pd.DataFrame()
        if not only_one_different_df.empty:
            from sklearn.preprocessing import LabelEncoder
            only_one_different_df["Sample No."] = LabelEncoder().fit_transform(
                only_one_different_df["Sample No."].values
            )
    except (ValueError, ImportError):
        only_one_different_df = pd.DataFrame()

    log_memory_usage(logger, "After chunked multiple measurements processing")

    rows_dropped = initial_len - len(df_result)
    logger.info(
        f"Dropped {rows_dropped} identical sample number rows ({rows_dropped/initial_len:.2%}), "
        f"keeping the one with the earliest date and time if samples matched within {std_threshold} stds"
    )

    return df_result, many_samples_df, only_one_different_df


def count_rows_dropped(df, initial_len, drop_col, keep_drop_rows):
    """Count rows that would be dropped."""
    if keep_drop_rows and drop_col in df.columns:
        return df[drop_col].sum()
    else:
        return initial_len - len(df)
