import argparse

import pandas as pd
from sysmexalign.alignment_1d import fit_gmm_plt, fit_gmm_rbc, transform_impedance_data
from sysmexalign.load_and_preprocess import load_sample_nos, load_sample_nos_from_config

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
        TRANSFORMED_OUTPUT = str(output_path / "OutputData_transformed.csv")

        SOURCE_SAMPLE_NOS = args.source_samples if hasattr(args, 'source_samples') and args.source_samples else None
        TARGET_SAMPLE_NOS = args.target_samples if hasattr(args, 'target_samples') and args.target_samples else None
    else:
        # Legacy approach with direct paths
        SOURCE_INPUT = args.source_inputs
        TRANSFORMED_OUTPUT = args.transformed_output
        TARGET_INPUT = args.target_inputs
        SOURCE_SAMPLE_NOS = args.source_sample_nos
        TARGET_SAMPLE_NOS = args.target_sample_nos

    print("Loading data...")

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

    # Convert target integer columns to float for consistency
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

    # get source and target standards RBC_RAW and PLT_RAW to calculate the transform
    source_standards = source_df[source_df["IsStandard"] == 1]
    target_standards = target_df[target_df["IsStandard"] == 1]

    print("Transforming RBC")
    source_df = transform_impedance_data(
        source_df, source_standards, target_standards, "RBC", fit_gmm_rbc, args
    )
    print("Transforming PLT")
    source_df = transform_impedance_data(
        source_df, source_standards, target_standards, "PLT", fit_gmm_plt, args
    )

    print("Saving transformed data to", TRANSFORMED_OUTPUT)

    source_df.to_csv(TRANSFORMED_OUTPUT, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transform OutputData.csv impedance column data (RBC_RAW, PLT_RAW)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # New configuration-based interface:
  python transform_impedance.py --source-dataset strides --target-dataset interval_36 --output-dataset strides_transformed_impedance --source-samples strides --target-samples interval_baseline_36
  
  # Legacy interface (deprecated):
  python transform_impedance.py --source_inputs file1.csv file2.csv --target_inputs target.csv --transformed_output output.csv
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
        help="Output dataset name from config (e.g., 'strides_transformed_impedance')",
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
        "--gmm_sample_size",
        type=int,
        default=10000,
        help="Number of samples to use to fit GMM (source and target)",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of jobs to run in parallel (default: -1, all CPUs)",
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
