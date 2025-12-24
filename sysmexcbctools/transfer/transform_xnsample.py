import argparse

import pandas as pd
from sysmexalign.alignment_1d import mad, transform_nonnormal
from sysmexalign.load_and_preprocess import load_sample_nos, load_sample_nos_from_config
from tqdm import tqdm

from .utils.data_loader import DataLoader


def main(args):
    # Use configuration if available, fallback to legacy paths
    if hasattr(args, 'source_dataset') and args.source_dataset:
        # New configuration-based approach
        config_loader = DataLoader(args.config_file if hasattr(args, 'config_file') else "config/data_paths.yaml")

        # Get source files from dataset
        source_files = config_loader.load_dataset_files('processed', args.source_dataset, '*.csv')
        SOURCE_INPUT = [str(f) for f in source_files]

        # Get target files from dataset
        target_files = config_loader.load_dataset_files('processed', args.target_dataset, '*.csv')
        TARGET_INPUT = [str(f) for f in target_files]

        # Output path
        output_path = config_loader.get_dataset_path('processed', args.output_dataset)
        output_path.mkdir(parents=True, exist_ok=True)
        TRANSFORMED_OUTPUT = str(output_path / "XN_SAMPLE_transformed.csv")

        SOURCE_SAMPLE_NOS = args.source_samples if hasattr(args, 'source_samples') and args.source_samples else None
        TARGET_SAMPLE_NOS = args.target_samples if hasattr(args, 'target_samples') and args.target_samples else None
    else:
        # Legacy approach with direct paths
        SOURCE_INPUT = args.source_inputs
        TRANSFORMED_OUTPUT = args.transformed_output
        TARGET_INPUT = args.target_inputs
        SOURCE_SAMPLE_NOS = args.source_sample_nos
        TARGET_SAMPLE_NOS = args.target_sample_nos

    TRANSFORM_COLS = args.transform_cols

    # read txt file TRANSFORM_COLS to get list of columns to transform
    with open(TRANSFORM_COLS) as f:
        transform_cols = f.readlines()
    transform_cols = [x.strip() for x in transform_cols]

    source_df = []
    target_df = []
    for filename in SOURCE_INPUT:
        source_df.append(pd.read_csv(filename, encoding="ISO-8859-1", low_memory=False))
    source_df = pd.concat(source_df, axis=0)
    # cast int columns to float to avoid later dtype conversion warnings
    source_df = source_df.astype(
        {
            col: float
            for col in source_df.select_dtypes(include=["int", "int64"]).columns
        }
    )

    for filename in TARGET_INPUT:
        target_df.append(pd.read_csv(filename, encoding="ISO-8859-1", low_memory=False))
    target_df = pd.concat(target_df, axis=0)
    # cast int columns to float to avoid later dtype conversion warnings
    target_df = target_df.astype(
        {
            col: float
            for col in target_df.select_dtypes(include=["int", "int64"]).columns
        }
    )

    # de-fragmenting the dataframes
    source_df = source_df.copy()
    target_df = target_df.copy()

    # calculate transform parameters using only the samples specified in SOURCE_SAMPLE_NOS and TARGET_SAMPLE_NOS (if provided)
    source_df["IsStandard"] = 1
    target_df["IsStandard"] = 1

    if SOURCE_SAMPLE_NOS is not None:
        if hasattr(args, 'source_dataset') and args.source_dataset:
            # New config-based approach - SOURCE_SAMPLE_NOS is dataset name
            print("Loading source sample numbers from config:", SOURCE_SAMPLE_NOS)
            config_file = args.config_file if hasattr(args, 'config_file') else "config/data_paths.yaml"
            source_sample_nos = load_sample_nos_from_config(SOURCE_SAMPLE_NOS, config_file)
        else:
            # Legacy approach - SOURCE_SAMPLE_NOS is file path
            print("Loading source sample numbers from file", SOURCE_SAMPLE_NOS)
            source_sample_nos = load_sample_nos(SOURCE_SAMPLE_NOS)
        source_df["IsStandard"] = (
            source_df["Sample No."].isin(source_sample_nos)
        ).astype(int)

    if TARGET_SAMPLE_NOS is not None:
        if hasattr(args, 'source_dataset') and args.source_dataset:
            # New config-based approach - TARGET_SAMPLE_NOS is dataset name
            print("Loading target sample numbers from config:", TARGET_SAMPLE_NOS)
            config_file = args.config_file if hasattr(args, 'config_file') else "config/data_paths.yaml"
            target_sample_nos = load_sample_nos_from_config(TARGET_SAMPLE_NOS, config_file)
        else:
            # Legacy approach - TARGET_SAMPLE_NOS is file path
            print("Loading target sample numbers from file", TARGET_SAMPLE_NOS)
            target_sample_nos = load_sample_nos(TARGET_SAMPLE_NOS)
        target_df["IsStandard"] = (
            target_df["Sample No."].isin(target_sample_nos)
        ).astype(int)

    source_df_transform_cols = source_df[transform_cols]
    target_df_transform_cols = target_df[transform_cols]

    # make dictionary where we record median and mad for each transform column
    source_transform_params = {}
    target_transform_params = {}

    print("\nTransformation parameters:")
    for col in transform_cols:
        source_standard_mask = source_df["IsStandard"] == 1
        target_standard_mask = target_df["IsStandard"] == 1
        masked_source = pd.to_numeric(
            source_df_transform_cols.loc[source_standard_mask, col], errors="coerce"
        ).dropna()
        masked_target = pd.to_numeric(
            target_df_transform_cols.loc[target_standard_mask, col], errors="coerce"
        ).dropna()
        source_transform_params[col] = {
            "median": masked_source.median(),
            "mad": mad(masked_source),
        }
        target_transform_params[col] = {
            "median": masked_target.median(),
            "mad": mad(masked_target),
        }
        print(
            f"{col}: Source (median={source_transform_params[col]['median']:.4f}, MAD={source_transform_params[col]['mad']:.4f}) → Target (median={target_transform_params[col]['median']:.4f}, MAD={target_transform_params[col]['mad']:.4f})"
        )

    # transform each column from source to target distribution using the MAD method and the median and MAD from the standards
    print("Transforming columns from source to target distribution...")
    for col in tqdm(transform_cols):
        # Create mask for numeric values
        numeric_mask = pd.to_numeric(
            source_df_transform_cols[col], errors="coerce"
        ).notna()
        # Store original values
        original_values = source_df_transform_cols[col].copy()
        # Convert numeric values
        source_df_transform_cols.loc[numeric_mask, col] = pd.to_numeric(
            source_df_transform_cols.loc[numeric_mask, col]
        )

        source_df_transform_cols.loc[numeric_mask, col] = transform_nonnormal(
            X=source_df_transform_cols.loc[numeric_mask, col].values,
            median_source=source_transform_params[col]["median"],
            median_target=target_transform_params[col]["median"],
            mad_source=source_transform_params[col]["mad"],
            mad_target=target_transform_params[col]["mad"],
        )
        # Restore non-numeric values
        source_df_transform_cols.loc[~numeric_mask, col] = original_values[
            ~numeric_mask
        ]

    # replace the original columns with the transformed columns
    source_df[transform_cols] = source_df_transform_cols

    # save transformed data to TRANSFORMED_OUTPUT
    source_df.to_csv(TRANSFORMED_OUTPUT, index=False, encoding="ISO-8859-1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transform XN_sample.csv data using MAD/median-based alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # New configuration-based interface:
  python transform_xnsample.py --source-dataset strides --target-dataset interval_36 --output-dataset strides_transformed_xn_sample --source-samples strides --target-samples interval_baseline_36
  
  # Legacy interface (deprecated):
  python transform_xnsample.py --source_inputs file1.csv file2.csv --target_inputs target.csv --transformed_output output.csv
        """
    )

    # New configuration-based arguments
    parser.add_argument(
        "--source-dataset",
        type=str,
        help="Source dataset name from config (e.g., 'strides', 'interval_36')",
    )
    parser.add_argument(
        "--target-dataset",
        type=str,
        help="Target dataset name from config (e.g., 'interval_36', 'strides')",
    )
    parser.add_argument(
        "--output-dataset",
        type=str,
        help="Output dataset name from config (e.g., 'strides_transformed_xn_sample')",
    )
    parser.add_argument(
        "--source-samples",
        type=str,
        help="Source samples dataset name from config (e.g., 'strides', 'interval_baseline_36')",
    )
    parser.add_argument(
        "--target-samples",
        type=str,
        help="Target samples dataset name from config (e.g., 'interval_baseline_36', 'strides')",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="config/data_paths.yaml",
        help="Path to configuration file (default: config/data_paths.yaml)",
    )

    # Legacy arguments (for backward compatibility)
    parser.add_argument(
        "--source_inputs",
        type=str,
        nargs="+",
        help="[DEPRECATED] Input files of source distribution. Use --source-dataset instead.",
    )
    parser.add_argument(
        "--target_inputs",
        type=str,
        nargs="+",
        help="[DEPRECATED] Input files of target distribution. Use --target-dataset instead.",
    )
    parser.add_argument(
        "--transformed_output",
        type=str,
        help="[DEPRECATED] Output file of transformed distribution. Use --output-dataset instead."
    )
    parser.add_argument(
        "--source_sample_nos",
        type=str,
        help="[DEPRECATED] file including sample numbers to use from source distribution. Use --source-samples instead.",
    )
    parser.add_argument(
        "--target_sample_nos",
        type=str,
        help="[DEPRECATED] file including sample numbers to use from target distribution. Use --target-samples instead.",
    )

    # Common arguments
    parser.add_argument(
        "--transform_cols",
        type=str,
        help="Path to file with columns to transform",
        default="XN_SAMPLE_transform_cols.txt",
    )

    args = parser.parse_args()

    # Validate arguments
    using_new_interface = bool(getattr(args, 'source_dataset', None) or getattr(args, 'target_dataset', None))
    using_legacy_interface = bool(args.source_inputs or args.target_inputs)

    if using_new_interface and using_legacy_interface:
        parser.error("Cannot mix new configuration-based arguments (--source-dataset) with legacy arguments (--source_inputs). Use one or the other.")

    if using_new_interface:
        if not (getattr(args, 'source_dataset', None) and getattr(args, 'target_dataset', None) and getattr(args, 'output_dataset', None)):
            parser.error("When using configuration-based interface, --source-dataset, --target-dataset, and --output-dataset are required.")
    else:
        if not (args.source_inputs and args.target_inputs and args.transformed_output):
            parser.error("When using legacy interface, --source_inputs, --target_inputs, and --transformed_output are required.")

    main(args)
