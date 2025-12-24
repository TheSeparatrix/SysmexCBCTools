import os
import re
import warnings
from datetime import datetime
from typing import List, Optional

import flowio
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils.data_loader import DataLoader


def load_sample_nos(file):
    """
    Legacy function for loading sample numbers from files.
    
    DEPRECATED: Use load_sample_nos_from_config() for new code.
    """
    warnings.warn(
        "load_sample_nos() is deprecated. Use load_sample_nos_from_config() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if file.endswith(".txt"):
        with open(file) as f:
            return f.read().splitlines()
    elif file.endswith(".csv"):
        return pd.read_csv(file)["sample_number"].tolist()
    elif file.endswith(".npy"):
        return np.load(file, allow_pickle=True).tolist()


def load_sample_nos_from_config(dataset_name: str, config_file: str = "config/data_paths.yaml"):
    """
    Load sample numbers using configuration-based paths.
    
    Args:
        dataset_name: Dataset name from config (e.g., 'strides', 'interval_baseline_36')
        config_file: Path to configuration file
    
    Returns:
        List of sample numbers
    
    Example:
        # Instead of: load_sample_nos("./data/raw/strides_centile_samples.npy")  
        sample_nos = load_sample_nos_from_config('strides')
    """
    loader = DataLoader(config_file)

    try:
        # Try to load as centile samples first (most common case)
        data = loader.load_numpy_file('centile_samples', dataset_name, allow_pickle=True)
        return data.tolist() if hasattr(data, 'tolist') else list(data)
    except (KeyError, FileNotFoundError) as e:
        raise ValueError(f"Dataset '{dataset_name}' not found in configuration. Available datasets: "
                        f"strides, interval_baseline, interval_baseline_36, interval_baseline_41") from e


class SysmexRawData:
    def __init__(self, filepath: Optional[str] = None):
        self.filename = os.path.basename(filepath) if filepath else None
        if filepath:
            parsed = parse_sysmex_raw_filename(self.filename)
            self.channel_name = parsed["channel_name"]
            self.analyser_id = parsed["analyser_id"]
            self.datetime = parsed["datetime"]
            self.sample_number = parsed["sample_number"]
            self.flowcyto_channels = parsed["data_channel_names"]
            if self.channel_name == "PLTF":
                self.data = preprocess_sct_pltf(filepath)
            elif self.channel_name == "RET":
                self.data = preprocess_sct_ret(filepath)
            elif self.channel_name == "WDF":
                self.data = preprocess_sct_wdf(filepath)
            elif self.channel_name == "WNR":
                self.data = preprocess_sct_wnr(filepath)
            elif self.channel_name == "WPC":
                self.data = preprocess_sct_wpc(filepath)
            elif self.channel_name == "PLT" or self.channel_name == "RBC":
                self.data = preprocess_impedence(filepath)
        else:
            self.data = None
            self.channel_name = None
            self.analyser_id = None
            self.datetime = None
            self.sample_number = None
            self.flowcyto_channels = None

    def to_csv(self, filepath):
        if self.data is None:
            raise ValueError("No data to save")
        data = pd.DataFrame(self.data, columns=self.flowcyto_channels)

        if self.channel_name == "RET":
            fsc = data["FSC"].to_numpy().astype(float)
            fsc_to_log = np.log10(fsc + 1) * (255 / np.log10(255))
            data["FSClog"] = fsc_to_log
        data.to_csv(filepath, index=False)


def is_overflow_file(filepath: str) -> bool:
    """
    Check if a file is an overflow file.

    Overflow files are named like: filename.116(1).csv, filename.116(2).csv

    Args:
        filepath: Path to check

    Returns:
        True if this is an overflow file, False otherwise
    """
    import re
    return bool(re.search(r'\.116\(\d+\)\.csv$', str(filepath)))


def filter_overflow_files(file_paths: List[str]) -> List[str]:
    """
    Filter out overflow files from a list of file paths.

    Only returns base files (files ending in .116.csv without overflow numbers).
    Overflow files will be automatically merged when the base file is loaded.

    Args:
        file_paths: List of file paths (may include both base and overflow files)

    Returns:
        List of base file paths only (overflow files removed)

    Example:
        >>> files = [
        ...     'RET_[...].116.csv',         # Keep
        ...     'RET_[...].116(1).csv',      # Remove
        ...     'WDF_[...].116.csv',         # Keep
        ...     'WDF_[...].116_transformed.csv'  # Keep
        ... ]
        >>> filter_overflow_files(files)
        ['RET_[...].116.csv', 'WDF_[...].116.csv', 'WDF_[...].116_transformed.csv']
    """
    return [f for f in file_paths if not is_overflow_file(f)]


def get_overflow_files(filepath: str) -> List[str]:
    """
    Find all overflow files for a given base file.

    Overflow files are named like: filename.116(1).csv, filename.116(2).csv

    Args:
        filepath: Path to the base .116.csv file

    Returns:
        List of overflow file paths, sorted by overflow number.
        Returns empty list if no overflow files exist.
    """
    from pathlib import Path

    path = Path(filepath)

    # Only check for .116(N).csv pattern (not transformed files)
    if not str(path).endswith('.116.csv'):
        return []

    # Derive overflow filenames: base.116.csv -> base.116(1).csv, base.116(2).csv, etc.
    base = str(path)[:-4]  # Remove .csv
    overflow_files = []

    # Check for overflow files (1), (2), (3), ... up to some reasonable limit
    # Most files won't have more than a few overflow files
    for n in range(1, 100):  # Check up to (99) - should be more than enough
        overflow_path = f"{base}({n}).csv"
        if os.path.exists(overflow_path):
            overflow_files.append(overflow_path)
        elif n > 1:
            # If we've found at least one overflow and now hit a gap, stop
            # (assumes overflow files are sequential: (1), (2), (3)... no gaps)
            break

    return overflow_files


def merge_csv_with_overflows(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file and merge with any overflow files.

    Merges raw CSV rows first, before any preprocessing.

    Args:
        filepath: Path to the base .116.csv file

    Returns:
        Merged pandas DataFrame with all rows from base + overflow files
    """
    # Load base file
    df = pd.read_csv(filepath)

    # Find and load overflow files
    overflow_files = get_overflow_files(filepath)

    if overflow_files:
        # Load all overflow files and concatenate
        overflow_dfs = [pd.read_csv(f) for f in overflow_files]
        df = pd.concat([df] + overflow_dfs, axis=0, ignore_index=True)

    return df


def concatenate_sysmex_data(data: List[SysmexRawData]):
    # Check that all data is from the same channel
    channels = {d.channel_name for d in data}
    if len(channels) > 1:
        raise ValueError(
            f"All data must be from the same channel, but found channels: {channels}"
        )

    # Concatenate the data
    valid_data = [d.data for d in data if d.data is not None]
    if not valid_data:
        raise ValueError("No valid data to concatenate")
    concatenated_data = np.concatenate(valid_data, axis=0)

    # Create a new SysmexRawData object for the concatenated data
    concatenated_data_obj = SysmexRawData()
    concatenated_data_obj.data = concatenated_data
    concatenated_data_obj.channel_name = data[0].channel_name
    concatenated_data_obj.analyser_id = "CONCATENATED"
    concatenated_data_obj.datetime = "CONCATENATED"
    concatenated_data_obj.sample_number = "CONCATENATED"
    concatenated_data_obj.flowcyto_channels = data[0].flowcyto_channels

    return concatenated_data_obj


def parse_sysmex_raw_filename(filename):
    # Pattern to match the filename components (including overflow files like .116(1).csv)
    pattern = r"""
        ^                           # Start of string
        ([A-Z]+)                   # Channel name (uppercase letters)
        _\[                        # Underscore and opening bracket
        ([^\]]+)                   # Analyser ID (anything until closing bracket)
        \]                         # Closing bracket
        \[[^\]]+\]                 # Skip the middle bracketed section
        \[                         # Opening bracket for datetime
        (\d{8}_\d{6})             # DateTime (YYYYMMDD_HHMMSS)
        \]                         # Closing bracket
        \[\s*([^\]]+)\]           # Sample number (allowing for spaces)
        \.116(?:_transformed)?(?:\(\d+\))?\.csv$  # File extension (with optional overflow number like (1))
    """

    match = re.match(pattern, filename, re.VERBOSE)

    if not match:
        raise ValueError(f"Filename '{filename}' does not match expected pattern")

    channel_name, analyser_id, dt_str, sample_number = match.groups()

    # Convert datetime string to datetime object
    dt = datetime.strptime(dt_str, "%Y%m%d_%H%M%S")

    # Strip any leading/trailing whitespace from sample number
    sample_number = sample_number.strip()

    # add flowcytometry channel names from Sysmex channel name
    data_channel_names = get_channel_names_from_sysmex_channel(channel_name)

    return {
        "channel_name": channel_name,
        "analyser_id": analyser_id,
        "datetime": dt,
        "sample_number": sample_number,
        "data_channel_names": data_channel_names,
    }


def make_fsc_from_raw_sct(filename: str, save_path: str, analyser_id: str):
    current_file = pd.read_csv(filename)
    date = get_datetime(filename)
    sample_number = get_sample_number(filename)
    # sysmex channel name is the first part of the filename (after the last / and before the first _)
    sysmex_channel = get_sysmex_channel_name(filename)

    channel_names = get_channel_names_from_sysmex_channel(sysmex_channel)

    df = current_file[channel_names]

    # delete all rows with 0 in any column or with the maximum of the df in any column
    df = df[(df != 0).all(1)]
    max_value = df.max().max()
    df = df[(df != max_value).all(1)]

    data_set = (
        np.vstack(
            [
                df.values,
            ]
        )
        .flatten()
        .tolist()
    )

    metadata = {
        "Analyser": analyser_id,
        "Date": date,
        "Filename": filename,
        "Channel": sysmex_channel,
        "SampleNumber": sample_number,
    }

    fh = open(save_path, "wb")
    flowio.create_fcs(fh, data_set, channel_names, metadata_dict=metadata)
    fh.close()


def get_data_paths(
    haas_output_path: str, qc_only: bool = True, remove_endbracket_samples: bool = True
):
    data_paths = {}
    # get impedance
    dist_folders = crawl_subfolders_for_foldername(haas_output_path, "DIST")
    imp_rbc_files = get_filepaths_from_filestart(dist_folders, "RBC")
    imp_plt_files = get_filepaths_from_filestart(dist_folders, "PLT")

    # get SCT
    sct_folders = crawl_subfolders_for_foldername(haas_output_path, "SCT")
    sct_ret = get_filepaths_from_filestart(sct_folders, "RET")
    sct_pltf = get_filepaths_from_filestart(sct_folders, "PLTF")
    sct_wdf = get_filepaths_from_filestart(sct_folders, "WDF")
    sct_wnr = get_filepaths_from_filestart(sct_folders, "WNR")

    if qc_only:
        imp_rbc_files = [f for f in imp_rbc_files if "QC" in f]
        if len(imp_rbc_files) == 0:
            raise ValueError("No QC files found in impedance RBC data")
        imp_plt_files = [f for f in imp_plt_files if "QC" in f]
        sct_ret = [f for f in sct_ret if "QC" in f]
        sct_pltf = [f for f in sct_pltf if "QC" in f]
        sct_wdf = [f for f in sct_wdf if "QC" in f]
        sct_wnr = [f for f in sct_wnr if "QC" in f]

    if remove_endbracket_samples:
        imp_rbc_files = [f for f in imp_rbc_files if not f.endswith(").csv")]
        imp_plt_files = [f for f in imp_plt_files if not f.endswith(").csv")]
        sct_ret = [f for f in sct_ret if not f.endswith(").csv")]
        sct_pltf = [f for f in sct_pltf if not f.endswith(").csv")]
        sct_wdf = [f for f in sct_wdf if not f.endswith(").csv")]
        sct_wnr = [f for f in sct_wnr if not f.endswith(").csv")]

    data_paths["imp_rbc"] = imp_rbc_files
    data_paths["imp_plt"] = imp_plt_files
    data_paths["sct_ret"] = sct_ret
    data_paths["sct_pltf"] = sct_pltf
    data_paths["sct_wdf"] = sct_wdf
    data_paths["sct_wnr"] = sct_wnr
    return data_paths


def load_saved_data(file_path: str):
    # get sample number from file name as the string between "[" and "]"
    sample_number = re.search(r"\[(.*?)\]", file_path).group(1)
    # get datetime as value after second last "_" and before file extension (".npy")
    datetime = re.search(r"\d{8}_\d{6}", file_path).group(0)
    data = np.load(file_path)
    sysmex_channel = get_sysmex_channel_name(file_path)
    channel_names = get_channel_names_from_sysmex_channel(sysmex_channel)
    return data, sample_number, datetime, sysmex_channel, channel_names


def process_and_save_data(data_paths: dict, output_dir: str):
    processor_map = {
        "imp_rbc": preprocess_impedence,
        "imp_plt": preprocess_impedence,
        "sct_ret": preprocess_sct_ret,
        "sct_pltf": preprocess_sct_pltf,
        "sct_wdf": preprocess_sct_wdf,
        "sct_wnr": preprocess_sct_wnr,
    }

    for key, processor in processor_map.items():
        print(f"Processing {key} data...")
        if key not in data_paths:
            continue
        for file_path in tqdm(data_paths[key]):
            data = processor(file_path)
            sample_number = get_sample_number(file_path)
            date_time = get_datetime(file_path)
            output_file = os.path.join(
                output_dir, f"{key}_[{sample_number}]_{date_time}.npy"
            )
            np.save(output_file, data)


def preprocess_impedence(file_path: str):
    # Merge with overflow files first, then preprocess
    data = merge_csv_with_overflows(file_path)
    data = data.dropna(subset=["Data"])
    return data["Data"].values


def preprocess_sct_ret(file_path: str):
    # Merge with overflow files first, then preprocess
    data = merge_csv_with_overflows(file_path)
    data = data.dropna(subset=["SFL", "FSC"])
    if "RepeatCount" in data.columns:
        data = data[data["RepeatCount"] == 0]
    data = data[["SFL", "FSC"]]
    # data = data[(data != 0).all(1)]
    # data = data[(data != 255).all(1)]
    return data.values


def preprocess_sct_pltf(file_path: str):
    # Merge with overflow files first, then preprocess
    data = merge_csv_with_overflows(file_path)
    data = data.dropna(subset=["SFL", "FSC", "SSC"])
    data = data[data["RepeatCount"] == 0]
    data.loc[data["Phase"] == "A", "FSCW"] = (
        255  # current convention to set FSCW to 255 for Phase A in PLT-F channel
    )
    data = data[["SFL", "FSC", "SSC", "FSCW"]]
    # data = data[(data != 0).all(1)]
    # data = data[(data != 255).all(1)]
    return data.values


def preprocess_sct_wdf(file_path: str):
    # Merge with overflow files first, then preprocess
    data = merge_csv_with_overflows(file_path)
    data = data.dropna(subset=["SSC", "SFL", "FSC", "FSCW"])
    data = data[data["RepeatCount"] == 0]
    data = data[["SSC", "SFL", "FSC", "FSCW"]]
    # data = data[(data != 0).all(1)]
    # data = data[(data != 255).all(1)]
    return data.values


def preprocess_sct_wnr(file_path: str):
    # Merge with overflow files first, then preprocess
    data = merge_csv_with_overflows(file_path)
    data = data.dropna(subset=["SFL", "FSC", "SSC", "FSCW"])
    data = data[data["RepeatCount"] == 0]
    data = data[["SFL", "FSC", "SSC", "FSCW"]]
    # data = data[(data != 0).all(1)]
    # data = data[(data != 255).all(1)]
    return data.values


def preprocess_sct_wpc(file_path: str):
    # Merge with overflow files first, then preprocess
    data = merge_csv_with_overflows(file_path)
    data = data.dropna(
        subset=["SSC", "FSC", "SFL"]
    )  # not using FSCW as seems to be always NaN for WPC
    data = data[data["RepeatCount"] == 0]
    data = data[["SSC", "FSC", "SFL"]]
    # data = data[(data != 0).all(1)]
    # data = data[(data != 255).all(1)]
    return data.values


def crawl_subfolders_for_foldername(root_folder: str, folder_name: str):
    """Search for folders with a specific name within a root directory and its subdirectories.

    This function performs a recursive search through a directory tree starting from the
    root_folder and finds all occurrences of folders matching the specified folder_name.

    Args:
        root_folder (str): The path to the root directory where the search begins.
        folder_name (str): The name of the folder to search for.

    Returns:
        list: A list of full paths to all matching folders found.

    Example:
        >>> crawl_subfolders_for_foldername('/path/to/root', 'target_folder')
        ['/path/to/root/dir1/target_folder', '/path/to/root/dir2/target_folder']
    """
    subfolders = []
    for root, dirs, files in os.walk(root_folder):
        if folder_name in dirs:
            subfolders.append(os.path.join(root, folder_name))
    return subfolders


def get_filepaths_from_filestart(path: str, filestart: str):
    """Return a list of full file paths from multiple directories that match a given file prefix.

    This function searches through multiple directories for files with names starting with
    a specified prefix and returns their complete paths.

    Args:
        path (str): A list of directory paths to search in
        filestart (str): The prefix to match file names against

    Returns:
        list: A list of full file paths for all matching files across all directories

    Example:
        >>> paths = ['/home/data1', '/home/data2']
        >>> get_filepaths_from_filestart(paths, 'test_')
        ['/home/data1/test_1.txt', '/home/data2/test_2.txt']
    """
    file_list = []
    for folder in path:
        file_list += [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.startswith(filestart)
        ]
    return file_list


def get_sample_number(file_name: str):
    # Pattern handles overflow files like .116(1).csv
    pattern = r"(\S+)\]\.116(?:\(\d+\))?\.csv$"
    try:
        sample_number = re.search(pattern, file_name).group(1)
    except AttributeError:
        print(f"Error extracting sample number from {file_name}")
        sample_number = None
    return sample_number


def get_datetime(file_name: str):
    matches = re.findall(r"\[(.*?)\]", file_name)
    if len(matches) >= 2:
        date_time = matches[-2]  # Penultimate match
    else:
        date_time = None
    return date_time


def get_sysmex_channel_name(filepath):
    # Get the filename from the full path by splitting on '/' and taking the last part
    filename = filepath.split("/")[-1]

    # Split on underscore and take the first part
    base_name = filename.split("_")[0]

    return base_name


def get_channel_names_from_sysmex_channel(sysmex_channel: str):
    valid_channels = ["RET", "PLTF", "WDF", "WNR", "WPC", "PLT", "RBC"]
    if sysmex_channel not in valid_channels:
        raise ValueError(
            f"Invalid sysmex channel: {sysmex_channel}. "
            f"Valid channels are: {', '.join(valid_channels)}"
        )

    if sysmex_channel == "RET":
        return ["SFL", "FSC"]
    elif sysmex_channel == "PLTF":
        return ["SFL", "FSC", "SSC", "FSCW"]
    elif sysmex_channel == "WDF":
        return ["SSC", "SFL", "FSC", "FSCW"]
    elif sysmex_channel == "WNR":
        return ["SFL", "FSC", "SSC", "FSCW"]
    elif sysmex_channel == "WPC":
        return ["SSC", "FSC", "SFL"]
    elif sysmex_channel == "PLT" or sysmex_channel == "RBC":
        return []
