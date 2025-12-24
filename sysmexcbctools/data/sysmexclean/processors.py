import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .constants import (
    MARKS,
    REDUNDANT_HGB_COLUMNS,
    STANDARD_FBC_DECRYPT,
    STANDARD_FBC_FEATURES,
    SYSMEX_TECHNICAL_SAMPLE_PREFIXES,
    TRASH_COLUMNS,
)
from .sysmex_channels_discrete_columns_Cambr import (
    CBC_measurements,
    DIFF_measurements,
    RET_measurements,
    WPC_measurements,
)
from .utils import chunked_correlation, get_drop_cols, log_memory_usage


def remove_duplicate_rows(df, logger):
    """Remove duplicate rows from dataframe."""
    shape_before = df.shape
    logger.info("Removing duplicate rows")
    log_memory_usage(logger, "Before removing duplicate rows")

    df = df.drop_duplicates()

    rows_dropped = shape_before[0] - df.shape[0]
    logger.info(
        f"Removed {rows_dropped} duplicate rows ({rows_dropped/shape_before[0]:.2%})"
    )

    log_memory_usage(logger, "After removing duplicate rows")
    return df


def remove_duplicate_columns(df, logger):
    """Remove duplicate columns from dataframe."""
    shape_before = df.shape
    logger.info("Removing duplicate columns, keeping the first occurrence")
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    rows_dropped = shape_before[1] - df.shape[1]
    logger.info(
        f"Removed {rows_dropped} duplicate columns ({rows_dropped/shape_before[1]:.2%})"
    )
    return df


def count_rows_dropped(df, length_before, filter_string, keep_drop_rows):
    if keep_drop_rows:
        rows_dropped = df.filter(like=filter_string).any(axis=1).sum()
    else:
        rows_dropped = length_before - df.shape[0]
    return rows_dropped


def remove_technical_samples(df, logger, keep_drop_rows=False):
    """Remove Sysmex sample numbers that are not associated with an individual."""
    shape_before = df.shape
    logger.info(
        f"Removing rows with technical sample numbers (starting with {SYSMEX_TECHNICAL_SAMPLE_PREFIXES})"
    )
    log_memory_usage(logger, "Before removing technical samples")

    for sample_start in SYSMEX_TECHNICAL_SAMPLE_PREFIXES:
        drop_indices = ~df["Sample No."].str.startswith(sample_start)
        if keep_drop_rows:
            # create new column for this particular sample prefix
            new_col_name = "drop_techsample_" + sample_start
            df[new_col_name] = False
            # flag columns which will be dropped
            df.loc[~drop_indices, new_col_name] = True
        else:
            df = df[~df["Sample No."].str.startswith(sample_start)]

    rows_dropped = count_rows_dropped(
        df, shape_before[0], "drop_techsample_", keep_drop_rows
    )

    logger.info(
        f"Removed {rows_dropped} technical sample rows ({rows_dropped/shape_before[0]:.2%})"
    )
    logger.info(f"Number of unique samples remaining: {df['Sample No.'].nunique()}")

    return df


def process_discrete_columns(df, logger):
    """Set measurements to NaN that are not indicated in the Discrete column."""
    logger.info(
        "Processing Discrete column to identify which measurements were performed"
    )

    # First, check for and handle duplicate column names
    if len(df.columns) != len(set(df.columns)):
        logger.warning("Duplicate column names detected in DataFrame")
        # Find the duplicates
        dupes = df.columns[df.columns.duplicated()].tolist()
        logger.warning(f"Duplicate columns: {dupes}")

        # Create a clean DataFrame with unique column names
        clean_columns = []
        seen = set()
        for col in df.columns:
            if col in seen:
                # Append a suffix to make it unique
                count = 1
                new_col = f"{col}_{count}"
                while new_col in seen:
                    count += 1
                    new_col = f"{col}_{count}"
                logger.warning(f"Renamed duplicate column '{col}' to '{new_col}'")
                clean_columns.append(new_col)
                seen.add(new_col)
            else:
                clean_columns.append(col)
                seen.add(col)

        # Assign unique column names
        df.columns = clean_columns

    # Create helper columns for each discrete entry possibility
    df["discrete_freeselect"] = df["Discrete"].str.contains("FREE SELECT", na=False)
    df["discrete_cbc"] = df["Discrete"].str.contains("CBC", na=False)
    df["discrete_diff"] = df["Discrete"].str.contains("DIFF", na=False)
    df["discrete_ret"] = df["Discrete"].str.contains("RET", na=False)
    df["discrete_pltf"] = df["Discrete"].str.contains("PLT-F", na=False)
    df["discrete_wpc"] = df["Discrete"].str.contains("WPC", na=False)

    # Filter the measurement lists to include only columns that exist in the DataFrame
    cbc_cols = [col for col in CBC_measurements if col in df.columns]
    diff_cols = [col for col in DIFF_measurements if col in df.columns]
    ret_cols = [col for col in RET_measurements if col in df.columns]
    wpc_cols = [col for col in WPC_measurements if col in df.columns]

    logger.debug(
        f"Setting values to NaN for {len(cbc_cols)} CBC measurements if not performed"
    )
    logger.debug(
        f"Setting values to NaN for {len(diff_cols)} DIFF measurements if not performed"
    )
    logger.debug(
        f"Setting values to NaN for {len(ret_cols)} RET measurements if not performed"
    )
    logger.debug(
        f"Setting values to NaN for {len(wpc_cols)} WPC measurements if not performed"
    )

    # Set columns to NaN if they are not in the discrete column
    if cbc_cols:
        df.loc[
            (df["discrete_freeselect"] == False) & (df["discrete_cbc"] == False),
            cbc_cols,
        ] = np.nan

    if diff_cols:
        df.loc[
            (df["discrete_freeselect"] == False) & (df["discrete_diff"] == False),
            diff_cols,
        ] = np.nan

    if ret_cols:
        df.loc[
            (df["discrete_freeselect"] == False) & (df["discrete_ret"] == False),
            ret_cols,
        ] = np.nan

    if wpc_cols:
        df.loc[
            (df["discrete_freeselect"] == False) & (df["discrete_wpc"] == False),
            wpc_cols,
        ] = np.nan

    # Remove discrete helper columns
    df = df.drop(columns=[col for col in df.columns if col.startswith("discrete_")])

    return df


def remove_unused_columns(df, logger):
    """Remove columns that are not needed or can't be interpreted."""
    # Identify columns that contain '(Reserved)' or 'Unnamed'
    extra_trash = [
        col for col in df.columns if ("(Reserved)" in col) or ("Unnamed" in col)
    ]

    # Combine with predefined columns to remove
    columns_to_remove = TRASH_COLUMNS + extra_trash

    logger.info(f"Removing {len(columns_to_remove)} unused columns")
    df = df.drop(columns=columns_to_remove, errors="ignore")

    return df


def encode_flags(df, logger):
    """Encode various flags and indicators as binary values."""
    logger.info("Encoding flags and IP messages")

    # Encode Positive/Error flags
    for col in df.columns:
        if col.startswith("Positive") or col.startswith("Error"):
            df.loc[df[col].isna(), col] = 0
            df.loc[~(df[col] == 0), col] = 1
            df[col] = df[col].astype(int)

    # Encode Abnormal or Suspect flags
    for col in df.columns:
        if col.endswith("Abnormal") or col.endswith("Suspect"):
            df.loc[df[col].isna(), col] = 0
            df[col] = df[col].astype(int)

    # Encode IP flags
    for col in df.columns:
        if col.startswith("IP "):
            df.loc[df[col].isna(), col] = 0
            df[col] = df[col].astype(int)

    # Process Q-Flag columns
    # Collect Q-Flag columns to avoid fragmentation from repeated column additions
    qflag_cols = [col for col in df.columns if col.startswith("Q-Flag")]

    if qflag_cols:
        # Collect all new columns in a dictionary to concatenate once
        new_columns = {}

        for col in qflag_cols:
            # Create new columns for ERROR and DISCRETE values
            new_columns[col + "_err"] = df[col].apply(lambda x: 1 if x == "ERROR" else 0)
            new_columns[col + "_disc"] = df[col].apply(lambda x: 1 if x == "DISCRETE" else 0)

            # Modify original column
            df.loc[df[col] == "ERROR", col] = np.nan
            df.loc[df[col] == "DISCRETE", col] = np.nan

            # Convert to numeric
            df.loc[:, col] = pd.to_numeric(df.loc[:, col], errors="coerce")
            df[col] = df[col].astype(float)

        # Add all new columns at once to avoid fragmentation
        if new_columns:
            df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    # Exclude columns that end in a .X (X = int)
    df = df.loc[:, ~df.columns.str.contains(r"\.\d+$")]

    return df


def process_marks(df, logger, make_dummy=False):
    # Process /M columns
    logger.info("Processing /M (data marks) columns")
    logger.info("Making dummies is set to " + str(make_dummy))

    # Collect /M columns
    mark_cols = [col for col in df.columns if col.endswith("/M")]

    # Collect new dummy columns if needed (to avoid fragmentation)
    new_dummy_columns = {}
    cols_to_drop = []

    for col in mark_cols:
        # rename column entries from Sysmex notation to explanation
        # use MARKS dictionary for renaming; anything not in there should be made NaN
        df[col] = df[col].replace(MARKS)
        # Use mask instead of replace to avoid downcasting warning
        unknown_values = [k for k in df[col].unique() if k not in MARKS.values()]
        df[col] = df[col].mask(df[col].isin(unknown_values))

        if make_dummy:
            # create dummy columns for each mark
            for mark in MARKS.values():
                new_col_name = col + "_" + mark
                new_dummy_columns[new_col_name] = (
                    df[col].apply(lambda x: 1 if x == mark else 0).astype(int)
                )
            # mark original column for dropping
            cols_to_drop.append(col)

    # Add all dummy columns at once to avoid fragmentation
    if new_dummy_columns:
        df = pd.concat([df, pd.DataFrame(new_dummy_columns, index=df.index)], axis=1)
        df = df.drop(columns=cols_to_drop)

    return df


def remove_clot_in_tube_samples(df, logger, keep_drop_rows=False):
    """Remove samples consistent with clot in tube."""
    logger.info("Removing samples consistent with clot in tube")
    len_before = df.shape[0]

    # Remove samples with indicators of clots or turbidity
    for indicator in [
        "IP SUS(RBC)Turbidity/HGB Interf?",
        "IP SUS(RBC)RBC Agglutination?",
        "IP SUS(PLT)PLT Clumps?",
    ]:
        keep_row_indices = ~(df[indicator] == 1)
        if keep_drop_rows:
            new_col_name = "drop_" + indicator
            df[new_col_name] = False
            # df[new_col_name][~keep_row_indices] = True
            df.loc[~keep_row_indices, new_col_name] = True
        else:
            df = df[keep_row_indices]

    rows_dropped = count_rows_dropped(df, len_before, "drop_IP SUS", keep_drop_rows)

    logger.info(
        f"Removed {rows_dropped} samples with indicators of clot in tube ({rows_dropped/len_before:.2%})"
    )

    return df


def handle_multiple_measurements(df, logger, std_threshold=1.0, keep_drop_rows=False):
    """
    Handle samples with multiple measurements by comparing on FBC features.
    If they're similar enough, keep the first measurement.
    """
    logger.info("Handling samples with multiple measurements")

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
        df[fbc_measurement] = pd.to_numeric(df[fbc_measurement], errors="coerce")

    # Calculate standard deviations for each FBC measurement
    stds = {
        fbc_measurement: df[fbc_measurement].std()
        for fbc_measurement in STANDARD_FBC_DECRYPT
    }

    if keep_drop_rows:
        drop_cols = get_drop_cols(df)

    # Process each group of samples with the same sample number
    grouped = df.groupby("Sample No.")
    multi_sample_rows = []
    odd_samples = []
    only_one_different = []

    logger.info("Checking for multiple measurements of the same sample number")
    for sample_id, group in tqdm(grouped):
        # Skip groups with only one row
        if len(group) < 2:
            multi_sample_rows.append(group)
            continue

        # Sort by date and time
        group = group.sort_values(["Sample No.", "Date", "Time"], ascending=True)

        # Check difference for each standard FBC measurement
        check_list = []
        for fbc_measurement in STANDARD_FBC_DECRYPT:
            difference = abs(
                group.iloc[0][fbc_measurement] - group.iloc[1][fbc_measurement]
            )
            check = ~(
                difference > std_threshold * stds[fbc_measurement]
            )  # written this way to keep NaNs

            if not check:
                logger.warning(
                    f"Sample {sample_id} failed check on {fbc_measurement} with value 1 "
                    f"{group.iloc[0][fbc_measurement]:.2f} and value 2 {group.iloc[1][fbc_measurement]:.2f} "
                    f"(threshold difference: {std_threshold * stds[fbc_measurement]:.2f})"
                )

            check_list.append(check)

        # Keep measurement if all checks were true or at most one failed
        if sum(check_list) >= len(check_list) - 1:
            multi_sample_rows.append(group.iloc[:1])  # Keep only the first measurement

            if keep_drop_rows:
                # if the first of the duplicates is already scheduled to be dropped
                # loop over all indices after the first
                already_dropping = group[drop_cols].any(axis=1)
                first_false_i = next(
                    (i for i, val in enumerate(already_dropping) if val is False), None
                )

                for i, idx in enumerate(group.index):
                    if i != first_false_i:
                        df.loc[idx, "drop_dup"] = True

                if first_false_i is None:
                    first_false_i = 0

                multi_sample_rows.append(
                    group.iloc[first_false_i : (first_false_i + 1)]
                )
            else:
                if sum(check_list) == len(check_list) - 1:  # only one check failed
                    only_one_different.append(group)
        else:
            if keep_drop_rows:
                for idx in group:
                    df.loc[idx, "drop_dup_disagree"] = True
            odd_samples.append(group)

    # Concatenate all rows
    df = pd.concat(multi_sample_rows)

    # Save samples that failed checks for manual review
    try:
        many_samples_df = pd.concat(odd_samples)
        from sklearn.preprocessing import LabelEncoder

        many_samples_df["Sample No."] = LabelEncoder().fit_transform(
            many_samples_df["Sample No."].values
        )
    except ValueError:
        many_samples_df = pd.DataFrame()

    try:
        only_one_different_df = pd.concat(only_one_different)
        from sklearn.preprocessing import LabelEncoder

        only_one_different_df["Sample No."] = LabelEncoder().fit_transform(
            only_one_different_df["Sample No."].values
        )
    except ValueError:
        only_one_different_df = pd.DataFrame()

    rows_dropped = count_rows_dropped(df, initial_len, "drop_dup", keep_drop_rows)
    logger.info(
        f"Dropped {rows_dropped} identical sample number rows ({rows_dropped/initial_len:.2%}), "
        f"keeping the one with the earliest date and time if samples matched within {std_threshold} stds"
    )

    return df, many_samples_df, only_one_different_df


def clean_non_numeric_values(df, logger):
    """Clean non-numeric values and handle dashes and spaces."""
    # Convert to numeric where possible
    # Note: errors='ignore' is deprecated, so we apply column-by-column
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                # Keep as object if conversion fails
                pass

    # Check for remaining non-numeric columns
    non_numeric_columns = df.columns[df.dtypes == "object"].tolist()
    logger.info(
        f"Found {len(non_numeric_columns)} columns with non-numeric values: {non_numeric_columns}"
    )

    # Replace dashes and spaces with NaN
    # Use mask instead of replace to avoid downcasting warning
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].mask(df[col].isin(["----", "      "]))

    # Log columns with NaN values
    nan_columns = df.columns[df.isna().any()].tolist()
    logger.info(f"Found {len(nan_columns)} columns with NaN values")

    if logger.isEnabledFor(logging.INFO):
        for col in nan_columns:
            nan_percentage = df[col].isna().sum() / len(df[col]) * 100
            logger.info(f"{col}: {nan_percentage:.2f}% NaN")

        # Total percentage of NaN values
        total_nan_percentage = (df.isna().sum().sum() / df.size) * 100
        logger.info(f"Total percentage of NaN values: {total_nan_percentage:.2f}%")

    return df


def remove_redundant_columns(df, logger):
    """Remove known redundant columns."""
    logger.info(f"Removing {len(REDUNDANT_HGB_COLUMNS)} redundant HGB columns")
    df.drop(columns=REDUNDANT_HGB_COLUMNS, errors="ignore", inplace=True)
    return df


def analyze_correlations(df, logger, output_dir, dataset_name: str = "", save_file: bool = True):
    """Analyse correlations with standard FBC features."""
    logger.info("Analysing correlations with standard FBC features")

    # Log memory usage before correlation analysis
    log_memory_usage(logger, "Before correlation analysis")

    # Use chunked correlation calculation for large datasets
    try:
        correlation_matrix = chunked_correlation(df, logger=logger)
    except Exception as e:
        logger.error(f"Error in correlation calculation: {e}")
        logger.info("Skipping correlation analysis due to memory constraints")
        # Return empty dataframe to maintain compatibility
        return pd.DataFrame({
            "Standard FBC feature": [],
            "Correlated features": [],
            "Correlation strength": [],
        })

    # Log memory usage after correlation calculation
    log_memory_usage(logger, "After correlation matrix calculation")

    # Find columns highly correlated with standard FBC features
    cols = []
    corr_feats = []
    corr_values = []

    for col in STANDARD_FBC_FEATURES:
        if col in correlation_matrix.columns:
            # Get correlated columns (correlation >= 0.8)
            correlated = correlation_matrix[col][
                (correlation_matrix[col] >= 0.8)
                & (correlation_matrix[col].index != col)
            ].index.tolist()

            for c in correlated:
                cols.append(col)
                corr_feats.append(c)
                corr_values.append(correlation_matrix.loc[col, c])

    # Create DataFrame of correlated columns
    correlated_columns = pd.DataFrame(
        {
            "Standard FBC feature": cols,
            "Correlated features": corr_feats,
            "Correlation strength": corr_values,
        }
    )

    # Save correlated columns (optional)
    if save_file:
        correlated_columns_path = (
            Path(output_dir) / f"{dataset_name}_correlated_columns.csv"
        )
        correlated_columns.to_csv(correlated_columns_path, index=False)
        logger.info(
            f"Found {len(set(corr_feats))} columns not in standard FBC features "
            f"but highly correlated (>= 0.8) with them"
        )
    else:
        logger.info(
            f"Found {len(set(corr_feats))} columns not in standard FBC features "
            f"but highly correlated (>= 0.8) with them (correlation file not saved)"
        )

    return correlated_columns


def remove_correlated_columns(df, correlated_columns, logger):
    """Remove columns highly correlated with standard FBC features."""
    logger.info("Removing columns highly correlated with standard FBC features")

    columns_to_remove = correlated_columns["Correlated features"].unique()
    df = df.drop(columns=columns_to_remove, errors="ignore")

    logger.info(f"Removed {len(columns_to_remove)} highly correlated columns")

    return df


def make_union_drop_column(df):
    # make one column which is the OR over all the other drop columns
    df["drop"] = df[get_drop_cols(df)].max(axis=1).astype(bool)
    return df.copy()
