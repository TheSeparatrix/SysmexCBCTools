import argparse
import copy
import os
import pickle

import numpy as np
from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture
from sysmexalign.gmm_ot import (
    compute_gmm_transport_map,
    transform_points_gmm,
    validate_transformation,
)
from sysmexalign.load_and_preprocess import (
    SysmexRawData,
    concatenate_sysmex_data,
    load_sample_nos,
    load_sample_nos_from_config,
    parse_sysmex_raw_filename,
)
from tqdm import tqdm

from .utils.data_loader import DataLoader


def main(args):
    # Initialise random number generator for reproducibility
    rng = np.random.default_rng(seed=args.seed if hasattr(args, 'seed') and args.seed is not None else None)
    if hasattr(args, 'seed') and args.seed is not None:
        print(f"Using random seed {args.seed} for reproducible transformations")
    else:
        print("No random seed specified - results will be non-deterministic")

    # Use configuration if available, fallback to legacy paths
    if hasattr(args, 'source_dataset') and args.source_dataset:
        # New configuration-based approach
        config_loader = DataLoader(args.config_file if hasattr(args, 'config_file') else "config/data_paths.yaml")
        SOURCE_FOLDER = str(config_loader.get_dataset_path('raw', args.source_dataset))
        TARGET_FOLDER = str(config_loader.get_dataset_path('raw', args.target_dataset))
        OUTPUT_FOLDER = "transformations"  # Use standard transformations folder

        SOURCE_NAME = args.source_dataset.upper()
        TARGET_NAME = args.target_dataset.upper()
    else:
        # Legacy approach with direct paths
        SOURCE_FOLDER = args.source_dist_folder
        TARGET_FOLDER = args.target_dist_folder
        OUTPUT_FOLDER = args.output_folder

        SOURCE_NAME = args.source_name
        TARGET_NAME = args.target_name

    CHANNEL = args.channel

    # Load data
    # go through source folder and load all data
    source_folder_files = os.listdir(SOURCE_FOLDER)
    relevant_files = [
        f
        for f in source_folder_files
        if ((f.startswith(CHANNEL)) and (f.endswith("116.csv")))
    ]

    if hasattr(args, 'source_samples') and args.source_samples:
        # New config-based approach
        config_file = args.config_file if hasattr(args, 'config_file') else "config/data_paths.yaml"
        sample_nos_source = load_sample_nos_from_config(args.source_samples, config_file)
    elif hasattr(args, 'sample_nos_source') and args.sample_nos_source is not None:
        # Legacy approach
        sample_nos_source = load_sample_nos(args.sample_nos_source)
    else:
        sample_nos_source = None

    if sample_nos_source is not None:

        relevant_files = [
            filename
            for filename in relevant_files
            if parse_sysmex_raw_filename(os.path.basename(filename))["sample_number"]
            in sample_nos_source
        ]

    print("Loading source data...")
    print("Number of relevant source files:", len(relevant_files))
    source_data = Parallel(n_jobs=-1)(
        delayed(SysmexRawData)(os.path.join(SOURCE_FOLDER, file))
        for file in tqdm(relevant_files)
    )

    print("Loading target data...")
    # go through target folder and load all data
    target_folder_files = os.listdir(TARGET_FOLDER)
    relevant_files = [
        f
        for f in target_folder_files
        if ((f.startswith(CHANNEL)) and (f.endswith("116.csv")))
    ]

    if hasattr(args, 'target_samples') and args.target_samples:
        # New config-based approach
        config_file = args.config_file if hasattr(args, 'config_file') else "config/data_paths.yaml"
        sample_nos_target = load_sample_nos_from_config(args.target_samples, config_file)
    elif hasattr(args, 'sample_nos_target') and args.sample_nos_target is not None:
        # Legacy approach
        sample_nos_target = load_sample_nos(args.sample_nos_target)
    else:
        sample_nos_target = None

    if sample_nos_target is not None:

        relevant_files = [
            filename
            for filename in relevant_files
            if parse_sysmex_raw_filename(os.path.basename(filename))["sample_number"]
            in sample_nos_target
        ]

    print("Number of relevant target files:", len(relevant_files))
    target_data = Parallel(n_jobs=-1)(
        delayed(SysmexRawData)(os.path.join(TARGET_FOLDER, file))
        for file in tqdm(relevant_files)
    )

    if len(source_data) == 0:
        raise ValueError(f"No data found for channel {CHANNEL} in source distribution.")

    if len(target_data) == 0:
        raise ValueError(f"No data found for channel {CHANNEL} in target distribution.")

    print("Doing analysis on channel:", CHANNEL)
    print("Concatenating data...")

    # concatenate all source and target data
    source_cct = concatenate_sysmex_data(source_data)
    target_cct = concatenate_sysmex_data(target_data)

    # remove under/oversaturated flow cytometry measurements (0s and 255s) as they are not correctly measured and would mess up the GMMs
    source_cct.data = source_cct.data[
        (source_cct.data != 0).all(axis=1) & (source_cct.data != 255).all(axis=1)
    ]
    target_cct.data = target_cct.data[
        (target_cct.data != 0).all(axis=1) & (target_cct.data != 255).all(axis=1)
    ]

    print("number of source data samples:", source_cct.data.shape[0])
    print("number of target data samples:", target_cct.data.shape[0])

    if source_cct.data.shape[0] > 1_000_000:
        print("Downsampling source data to 1 million points...")
        indices = rng.choice(source_cct.data.shape[0], 1_000_000, replace=False)
        source_cct.data = source_cct.data[indices]

    if target_cct.data.shape[0] > 1_000_000:
        print("Downsampling target data to 1 million points...")
        indices = rng.choice(target_cct.data.shape[0], 1_000_000, replace=False)
        target_cct.data = target_cct.data[indices]

    print("Fitting GMMs...")

    # Increased number of components for better resolution, especially for WDF/PLT-F channels
    # NOTE: use 60 for all for now, possibly could use SWIFT to determine optimal initialisation
    n_components = 60

    print("Fitting source GMM...")
    gmm_source = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        reg_covar=1e-5,
        max_iter=200,
        n_init=3,
        verbose=1,
        random_state=args.seed,
    ).fit(source_cct.data)

    print("Fitting target GMM...")
    gmm_target = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        reg_covar=1e-5,
        max_iter=200,
        n_init=3,
        verbose=1,
        random_state=args.seed,
    ).fit(target_cct.data)

    print("Computing transport map...")
    transport_dict = compute_gmm_transport_map(gmm_source, gmm_target)

    print("Transforming concatenated source data for transform reference...")
    source_transformed = copy.deepcopy(source_cct)
    source_transformed.data = transform_points_gmm(
        source_cct.data,
        gmm_source,
        transport_dict,
        transport_method="rand",  # T_rand seems to create the most realistic (smooth) transformations
        threshold=0.05,
        preserve_rare=True,
        rare_probability_threshold=0.01,
    )

    print("Saving GMMs, transport map and source and target data...")
    # Save GMMs and transport map
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    with open(
        os.path.join(
            OUTPUT_FOLDER,
            f"{SOURCE_NAME}_to_{TARGET_NAME}_transport_dict_{CHANNEL}.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(transport_dict, f)

    with open(
        os.path.join(OUTPUT_FOLDER, f"{SOURCE_NAME}_{CHANNEL}_GMM.pkl"), "wb"
    ) as f:
        pickle.dump(gmm_source, f)

    with open(
        os.path.join(OUTPUT_FOLDER, f"{TARGET_NAME}_{CHANNEL}_GMM.pkl"), "wb"
    ) as f:
        pickle.dump(gmm_target, f)

    # Save data
    source_transformed.to_csv(
        os.path.join(OUTPUT_FOLDER, f"{SOURCE_NAME}_{CHANNEL}_cct_transformed.csv")
    )
    target_cct.to_csv(os.path.join(OUTPUT_FOLDER, f"{TARGET_NAME}_{CHANNEL}_cct.csv"))
    source_cct.to_csv(os.path.join(OUTPUT_FOLDER, f"{SOURCE_NAME}_{CHANNEL}_cct.csv"))

    # Validate transformation and save output image
    print("Validating transformation...")
    validate_transformation(
        original_data=source_cct.data,
        target_data=target_cct.data,
        transformed_data=source_transformed.data,
        gmm_target=gmm_target,
        output_path=os.path.join(
            OUTPUT_FOLDER,
            f"figs/{SOURCE_NAME}_to_{TARGET_NAME}_{CHANNEL}_validation.png",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load Sysmex raw data from provided source and target folders and compute registration, based on optimal transport between Gaussian Mixture models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # New configuration-based interface:
  python compute_transformation.py --source-dataset strides_merged --target-dataset interval_36 --channel RET --source-samples strides --target-samples interval_baseline_36
  
  # Legacy interface (deprecated):
  python compute_transformation.py /path/to/source /path/to/target /path/to/output SOURCE_NAME TARGET_NAME RET --sample_nos_source samples.npy
        """
    )

    # New configuration-based arguments
    parser.add_argument(
        "--source-dataset",
        type=str,
        help="Source dataset name from config (e.g., 'strides_merged', 'interval_36')",
    )
    parser.add_argument(
        "--target-dataset",
        type=str,
        help="Target dataset name from config (e.g., 'interval_36', 'strides')",
    )
    parser.add_argument(
        "--channel",
        type=str,
        help="Sysmex Channel to consider (RET, WDF, WNR, PLTF)",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible downsampling and GMM fitting (default: None for non-deterministic)",
    )

    # Legacy positional arguments (for backward compatibility)
    parser.add_argument(
        "source_dist_folder",
        type=str,
        nargs='?',
        help="[DEPRECATED] Path to folder containing Sysmex raw data of the source distribution.",
    )
    parser.add_argument(
        "target_dist_folder",
        type=str,
        nargs='?',
        help="[DEPRECATED] Path to folder containing Sysmex raw data of the target distribution.",
    )
    parser.add_argument(
        "output_folder",
        type=str,
        nargs='?',
        help="[DEPRECATED] Path to folder where the transport map and source and target GMMs will be saved.",
    )
    parser.add_argument(
        "source_name",
        type=str,
        nargs='?',
        help="[DEPRECATED] Name of the source distribution.",
    )
    parser.add_argument(
        "target_name",
        type=str,
        nargs='?',
        help="[DEPRECATED] Name of the target distribution.",
    )
    parser.add_argument(
        "legacy_channel",
        type=str,
        nargs='?',
        help="[DEPRECATED] Sysmex Channel to consider. Use --channel instead.",
    )
    parser.add_argument(
        "--sample_nos_source",
        type=str,
        default=None,
        help="[DEPRECATED] Filepath of TXT file containing sample numbers of source distribution data to be used. Use --source-samples instead.",
    )
    parser.add_argument(
        "--sample_nos_target",
        type=str,
        default=None,
        help="[DEPRECATED] Filepath of TXT file containing sample numbers of target distribution data to be used. Use --target-samples instead.",
    )

    args = parser.parse_args()

    # Validate arguments
    using_new_interface = bool(getattr(args, 'source_dataset', None) or getattr(args, 'target_dataset', None))
    using_legacy_interface = bool(args.source_dist_folder or args.target_dist_folder)

    if using_new_interface and using_legacy_interface:
        parser.error("Cannot mix new configuration-based arguments (--source-dataset) with legacy positional arguments. Use one or the other.")

    if using_new_interface:
        if not (getattr(args, 'source_dataset', None) and getattr(args, 'target_dataset', None)):
            parser.error("When using configuration-based interface, both --source-dataset and --target-dataset are required.")
        if not getattr(args, 'channel', None):
            parser.error("When using configuration-based interface, --channel is required.")
    else:
        if not (args.source_dist_folder and args.target_dist_folder and args.output_folder and args.source_name and args.target_name):
            parser.error("When using legacy interface, all positional arguments are required.")
        # Use legacy channel argument if --channel not provided
        if not getattr(args, 'channel', None) and hasattr(args, 'legacy_channel') and getattr(args, 'legacy_channel', None):
            args.channel = args.legacy_channel
        if not getattr(args, 'channel', None):
            parser.error("Channel must be specified via --channel or as a positional argument.")

    main(args)
