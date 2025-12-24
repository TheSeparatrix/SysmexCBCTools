import argparse
import os
import pickle

import numpy as np
from joblib import Parallel, delayed
from sysmexalign.gmm_ot import transform_points_gmm
from sysmexalign.load_and_preprocess import (
    SysmexRawData,
    load_sample_nos,
    load_sample_nos_from_config,
    parse_sysmex_raw_filename,
)
from tqdm import tqdm

from .utils.data_loader import DataLoader


def main(args):
    # Use configuration if available, fallback to legacy paths
    if hasattr(args, 'source_dataset') and args.source_dataset:
        # New configuration-based approach
        config_loader = DataLoader(args.config_file if hasattr(args, 'config_file') else "config/data_paths.yaml")
        INPUT_FOLDER = str(config_loader.get_dataset_path('raw', args.source_dataset))
        OUTPUT_FOLDER = str(config_loader.get_dataset_path('processed', args.output_dataset)) if hasattr(args, 'output_dataset') and args.output_dataset else "data/processed"

        SOURCE_NAME = args.source_dataset.upper()
        TARGET_NAME = args.target_dataset.upper() if hasattr(args, 'target_dataset') and args.target_dataset else "TARGET"
    else:
        # Legacy approach with direct paths
        INPUT_FOLDER = args.input_folder
        OUTPUT_FOLDER = args.output_folder
        SOURCE_NAME = args.source_name
        TARGET_NAME = args.target_name

    CHANNEL = args.channel

    # Load data

    input_folder_files = os.listdir(INPUT_FOLDER)
    relevant_files = [
        f
        for f in input_folder_files
        if ((f.startswith(CHANNEL)) and (f.endswith("116.csv")))
    ]

    if hasattr(args, 'samples') and args.samples:
        # New config-based approach
        config_file = args.config_file if hasattr(args, 'config_file') else "config/data_paths.yaml"
        sample_nos = load_sample_nos_from_config(args.samples, config_file)
    elif hasattr(args, 'sample_nos') and args.sample_nos is not None:
        # Legacy approach
        sample_nos = load_sample_nos(args.sample_nos)
    else:
        sample_nos = None

    if sample_nos is not None:
        relevant_files = [
            filename
            for filename in relevant_files
            if parse_sysmex_raw_filename(os.path.basename(filename))["sample_number"]
            in sample_nos
        ]

    print("Loading transformation dict and source GMM...")

    # Auto-discover transformation files if using config-based approach
    if hasattr(args, 'source_dataset') and args.source_dataset and hasattr(args, 'transformation') and args.transformation:
        # Use transformation name to find files
        transport_dict_path = f"transformations/{args.transformation}_transport_dict_{CHANNEL}.pkl"
        source_gmm_path = f"transformations/{SOURCE_NAME}_{CHANNEL}_GMM.pkl"

        if not os.path.exists(transport_dict_path):
            raise FileNotFoundError(f"Transport dictionary not found: {transport_dict_path}")
        if not os.path.exists(source_gmm_path):
            raise FileNotFoundError(f"Source GMM not found: {source_gmm_path}")

        with open(transport_dict_path, "rb") as f:
            transport_dict = pickle.load(f)
        with open(source_gmm_path, "rb") as f:
            source_gmm = pickle.load(f)
    else:
        # Legacy approach with explicit file paths
        with open(args.transport_dict, "rb") as f:
            transport_dict = pickle.load(f)
        with open(args.source_gmm, "rb") as f:
            source_gmm = pickle.load(f)

    print("Loading, transforming, and saving data...")
    print("Number of relevant files:", len(relevant_files))

    # Set transformation parameters based on channel characteristics
    if CHANNEL == "WDF":
        # For differential channel, preserve rare populations more strictly
        rare_threshold = 0.005  # 0.5% probability threshold for rare populations
        omega_threshold = 0.03  # Lower threshold to capture more structure
    elif CHANNEL == "WNR":
        # For nuclear channel with distinct populations
        rare_threshold = 0.01
        omega_threshold = 0.05
    elif CHANNEL == "PLTF":
        # For platelet channel, more aggressive rare preservation
        rare_threshold = 0.003
        omega_threshold = 0.02
    else:
        # Default parameters for other channels
        rare_threshold = 0.01
        omega_threshold = 0.05

    print(f"Using rare threshold: {rare_threshold}, omega threshold: {omega_threshold}")

    def load_transform_save(file):
        data = SysmexRawData(os.path.join(INPUT_FOLDER, file))
        if len(data.data) == 0:
            return None

        data.data = transform_points_gmm(
            data.data,
            source_gmm,
            transport_dict,
            transport_method=args.transport_method,
            threshold=omega_threshold,
            preserve_rare=True,
            rare_probability_threshold=rare_threshold,
        )

        # set any value >255 to 255, and any value <0 to 0 after transform
        data.data = np.clip(data.data, 0, 255)

        # Ensure output folder exists
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        output_filename = os.path.join(OUTPUT_FOLDER, file[:-4] + "_transformed.csv")
        data.to_csv(output_filename)

    Parallel(n_jobs=args.n_jobs)(
        delayed(load_transform_save)(file) for file in tqdm(relevant_files)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transform Sysmex raw data using pre-computed transformation models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # New configuration-based interface:
  python transform.py --source-dataset strides_merged --output-dataset strides_transformed_to_interval36 --channel RET --transformation strides_to_interval36 --samples strides
  
  # Legacy interface (deprecated):
  python transform.py /path/to/input /path/to/output SOURCE TARGET RET transport_dict.pkl source_gmm.pkl
        """
    )

    # New configuration-based arguments
    parser.add_argument(
        "--source-dataset",
        type=str,
        help="Source dataset name from config (e.g., 'strides_merged', 'interval_36')",
    )
    parser.add_argument(
        "--output-dataset",
        type=str,
        help="Output dataset name from config (e.g., 'strides_transformed')",
    )
    parser.add_argument(
        "--channel",
        type=str,
        help="Sysmex Channel to consider (RET, WDF, WNR, PLTF)",
    )
    parser.add_argument(
        "--transformation",
        type=str,
        help="Transformation name (e.g., 'strides_to_interval36'). Files will be auto-discovered from transformations/ folder.",
    )
    parser.add_argument(
        "--samples",
        type=str,
        help="Samples dataset name from config (e.g., 'strides', 'interval_baseline_36')",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="config/data_paths.yaml",
        help="Path to configuration file (default: config/data_paths.yaml)",
    )

    # Legacy positional arguments (for backward compatibility)
    parser.add_argument(
        "input_folder",
        type=str,
        nargs='?',
        help="[DEPRECATED] Folder containing files to transform."
    )
    parser.add_argument(
        "output_folder",
        type=str,
        nargs='?',
        help="[DEPRECATED] Folder to save transformed files."
    )
    parser.add_argument(
        "source_name",
        type=str,
        nargs='?',
        help="[DEPRECATED] Name of the source distribution."
    )
    parser.add_argument(
        "target_name",
        type=str,
        nargs='?',
        help="[DEPRECATED] Name of the target distribution."
    )
    parser.add_argument(
        "legacy_channel",
        type=str,
        nargs='?',
        help="[DEPRECATED] Sysmex Channel to consider. Use --channel instead."
    )
    parser.add_argument(
        "transport_dict",
        type=str,
        nargs='?',
        help="[DEPRECATED] Path to transport dict."
    )
    parser.add_argument(
        "source_gmm",
        type=str,
        nargs='?',
        help="[DEPRECATED] Path to source GMM."
    )
    parser.add_argument(
        "--sample_nos",
        type=str,
        help="[DEPRECATED] Sample numbers of samples to transform. Use --samples instead.",
        default=None,
    )

    # Common arguments
    parser.add_argument(
        "--transport_method",
        type=str,
        help="Method to use for transport map computation.",
        default="rand",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        help="Number of jobs to run in parallel. (-1 for all available cores)",
        default=1,
    )

    args = parser.parse_args()

    # Validate arguments
    using_new_interface = bool(getattr(args, 'source_dataset', None))
    using_legacy_interface = bool(args.input_folder)

    if using_new_interface and using_legacy_interface:
        parser.error("Cannot mix new configuration-based arguments (--source-dataset) with legacy positional arguments. Use one or the other.")

    if using_new_interface:
        if not (getattr(args, 'source_dataset', None) and getattr(args, 'transformation', None)):
            parser.error("When using configuration-based interface, --source-dataset and --transformation are required.")
        if not getattr(args, 'channel', None):
            parser.error("When using configuration-based interface, --channel is required.")
    else:
        if not (args.input_folder and args.output_folder and args.source_name and args.target_name and args.transport_dict and args.source_gmm):
            parser.error("When using legacy interface, all positional arguments are required.")
        # Use legacy channel argument if --channel not provided
        if not getattr(args, 'channel', None) and hasattr(args, 'legacy_channel') and getattr(args, 'legacy_channel', None):
            args.channel = args.legacy_channel
        if not getattr(args, 'channel', None):
            parser.error("Channel must be specified via --channel or as a positional argument.")

    main(args)
