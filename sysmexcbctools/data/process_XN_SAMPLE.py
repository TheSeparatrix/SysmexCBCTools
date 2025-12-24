"""
Command-line interface for processing Sysmex XN_SAMPLE data files.

This script provides a CLI wrapper around the XNSampleProcessor class,
maintaining backward compatibility with the original script interface.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Handle both direct execution and module import
try:
    from .sysmexclean import XNSampleProcessor
except ImportError:
    # Running as script, add parent to path
    sys.path.insert(0, str(Path(__file__).parent))
    from sysmexclean import XNSampleProcessor


def main():
    """Main function to process Sysmex XN_SAMPLE data."""
    # Start timer
    start_time = datetime.now()

    parser = argparse.ArgumentParser(
        description="Process Sysmex XN_SAMPLE data files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process using config file
  python process_XN_SAMPLE.py --config config.yaml

  # Process specific dataset from config
  python process_XN_SAMPLE.py --config config.yaml --dataset INTERVAL

  # Process files directly without config
  python process_XN_SAMPLE.py --files data1.csv data2.csv --output-dir ./results
        """,
    )
    parser.add_argument(
        "--config", type=str, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Specific dataset name to process (requires --config)"
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        help="XN_SAMPLE.csv file(s) to process directly (without config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=1,
        help="Increase verbosity (can be used multiple times)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.config is None and args.files is None:
        parser.error("Either --config or --files must be provided")

    if args.dataset and not args.config:
        parser.error("--dataset requires --config")

    try:
        if args.config:
            # Config-based processing
            processor = XNSampleProcessor(
                config_path=args.config,
                output_dir=args.output_dir,
                verbose=args.verbose
            )

            # Determine which datasets to process
            if args.dataset:
                datasets_to_process = [args.dataset]
            elif "use_dataset" in processor.config["input"] and processor.config["input"]["use_dataset"]:
                datasets_to_process = [processor.config["input"]["use_dataset"]]
            else:
                datasets_to_process = [
                    ds["name"] for ds in processor.config["input"]["datasets"]
                ]

            processor.logger.info(f"Starting XN_SAMPLE processing with config: {args.config}")

            # Process each dataset
            results = {}
            for dataset_name in datasets_to_process:
                try:
                    results[dataset_name] = processor.process(dataset_name)
                except Exception as e:
                    processor.logger.error(
                        f"Error processing dataset {dataset_name}: {e}", exc_info=True
                    )

        else:
            # Direct file processing (no config)
            processor = XNSampleProcessor(
                output_dir=args.output_dir,
                verbose=args.verbose
            )

            processor.logger.info(
                f"Starting XN_SAMPLE processing for {len(args.files)} file(s)"
            )

            # Process files with save_output=True for CLI usage
            df = processor.process_files(
                input_files=args.files,
                dataset_name="processed",
                save_output=True
            )

        processor.logger.info(
            "XN_SAMPLE processing complete, after %d seconds",
            int((datetime.now() - start_time).total_seconds()),
        )

    except Exception as e:
        print(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()
