import argparse
import logging
import os

import cryptpandas
import dvc.api
import numpy as np
import pandas as pd
from irondeficiency.utils import summarize_dataframe
from tqdm import tqdm

# useful list of FBC parameters in XN_SAMPLE notation:
standard_fbc_decrypt = [
    "WBC(10^3/uL)",
    "RBC(10^6/uL)",
    "HGB(g/dL)",
    "HCT(%)",
    # "MCV(fL)",
    # "MCH(pg)",
    # "MCHC(g/dL)",
    "PLT(10^3/uL)",
]

logger = logging.getLogger(__name__)
logging.basicConfig(filename="logs/technical_clean.log", level=logging.INFO)

if __name__ == "__main__":
    params = dvc.api.params_show()
    out_dir = params["out_dir"]
    interval_raw_folder = params["interval_haas_folder"]
    strides_raw_folder = params["strides_haas_folder"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="INTERVAL",
        help="Which dataset to use. Must be STRIDES or INTERVAL.",
    )
    args = parser.parse_args()
    DATASET = args.dataset

    print("DATASET:", DATASET)

    if DATASET == "INTERVAL":
        CSV_DIRECTORY = interval_raw_folder

        csv_files = [
            "XN-10^11036/XN_SAMPLE.csv",
            "XN-10^11041/XN_SAMPLE.csv",
        ]
    elif DATASET == "STRIDES":
        CSV_DIRECTORY = strides_raw_folder

        csv_files = [
            "simon_batch1_part1/XN_SAMPLE.csv",
            "simon_batch1_part3/XN_SAMPLE.csv",
            "simon_batch1_part8/XN_SAMPLE.csv",
            "simon_batch1_part2/XN_SAMPLE.csv",
            "simon_batch1_part5/XN_SAMPLE.csv",
            "simon_batch2/XN_SAMPLE.csv",
        ]
    else:
        raise ValueError("Unknown dataset, must be STRIDES or INTERVAL")

    dfs = []
    for file in csv_files:
        path = os.path.join(CSV_DIRECTORY, file)
        df = pd.read_csv(path, encoding="ISO-8859-1", low_memory=False)
        dfs.append(df)

    df = pd.concat(dfs, axis=0).reset_index(drop=True)

    logger.info("Initial shape of XN_SAMPLE dataframe: %s", df.shape)

    og_columns = list(df)

    # save original columns in txt file (if wished)
    with open(f"{out_dir}/{DATASET}_originalcolumns.txt", "w") as file:
        for item in og_columns:
            file.write("%s\n" % item)

    # drop rows that are the exact same
    shape_before = df.shape
    df = df.drop_duplicates()
    logger.info("Shape after dropping duplicates: %s", df.shape)
    logger.info("Dropped %s rows", shape_before[0] - df.shape[0])

    # Remove all rows that are Quality Control (QC) samples
    qc_group = df[df["Sample No."].str.startswith("QC-")]
    g_group = df[df["Sample No."].str.startswith("=G")]
    int_group = df[df["Sample No."].str.startswith("INT")]
    other_group = df[
        ~df["Sample No."].str.startswith("QC-")
        & ~df["Sample No."].str.startswith("=G")
        & ~df["Sample No."].str.startswith("INT")
    ]

    # Calculate the percentages of each group
    total_rows = len(df)
    qc_percentage = (len(qc_group) / total_rows) * 100
    g_percentage = (len(g_group) / total_rows) * 100
    int_percentage = (len(int_group) / total_rows) * 100
    other_percentage = (len(other_group) / total_rows) * 100
    logger.info("Percentage of rows starting with 'QC-': %.2f%%", qc_percentage)
    logger.info("Percentage of rows starting with '=G': %.2f%%", g_percentage)
    logger.info("Percentage of rows starting with 'INT': %.2f%%", int_percentage)
    logger.info("Percentage of other rows: %.2f%%", other_percentage)

    n_samples = df.shape[0]
    # Remove all rows that are not INT or =G samples
    df = df.drop_duplicates()
    df = df[
        df["Sample No."].str.startswith("=G") | df["Sample No."].str.startswith("INT")
    ]
    logger.info(
        "Dropped %s samples that don't have INT or G-number sample numbers",
        n_samples - df.shape[0],
    )

    logger.info(
        "Shape before cleaning up columns and dropping unreliable samples: %s", df.shape
    )
    logger.info(
        "Number of unique samples in the dataset: %s", df["Sample No."].nunique()
    )

    # remove columns we don't need / can't interpret
    trash_cols = [
        "Nickname",
        "Rack",
        "Position",
        "Sample Inf.",
        "Order Type",
        # "Reception Date",
        "Measurement Mode",
        "Discrete",
        "Patient ID",
        "Analysis Info.",
        "Judgment",
        "Order Info.",
        "WBC Info.",
        "PLT Info.",
        "Rule Result",
        "Validate",
        "Validator",
        "Action Message (Check)",  # should be useful but don't understand Sysmex encoding
        "Action Message (Review)",  # should be useful but don't understand Sysmex encoding
        "Action Message (Retest)",  # should be useful but don't understand Sysmex encoding
        "Sample Comment",
        "Patient Name",
        "Birth",
        "Sex",
        "Patient Comment",
        "Ward Name",
        "Doctor Name",
        "Output",
        "Sequence No.",  # not sure what these mean, all int
        "Unclassified()",  # not sure what these mean, all int
        "HF-BF#(10^3/uL)",  # almost all NaN #! might be INTERVAL only
        "HF-BF%(/100WBC)",  # almost all NaN #! might be INTERVAL only
    ]
    extra_trash = []
    for col in df.columns:
        if ("(Reserved)" in col) or ("Unnamed" in col):
            extra_trash.append(col)

    trash_cols = trash_cols + extra_trash  # for later reference
    df.drop(columns=trash_cols, inplace=True)

    # encode Positive(Diff/Morph/Count)
    for col in df.columns:
        if col.startswith("Positive") or col.startswith("Error"):
            df.loc[df[col].isna(), col] = 0
            df.loc[~(df[col] == 0), col] = 1
            df[col] = df[col].astype(int)

    # encode Abnormal or Suspect flags
    for col in df.columns:
        if col.endswith("Abnormal") or col.endswith("Suspect"):
            df.loc[df[col].isna(), col] = 0
            df[col] = df[col].astype(int)

    # encode IP flags
    for col in df.columns:
        if col.startswith("IP "):
            df.loc[df[col].isna(), col] = 0
            df[col] = df[col].astype(int)

    # for the Q-Flags, if the entry is a number (or a string that could be a number), keep the number
    # if it's "DISCRETE" or "ERROR", insert NaN and make two separate columns for each IP that binary-encode DISCRETE and ERROR
    for col in df.columns:
        if col.startswith("Q-Flag"):
            df[col + "_err"] = df[col].apply(lambda x: 1 if x == "ERROR" else 0)
            df.loc[df[col] == "ERROR", col] = np.nan

            df[col + "_disc"] = df[col].apply(lambda x: 1 if x == "DISCRETE" else 0)
            df.loc[df[col] == "DISCRETE", col] = np.nan

            df.loc[:, col] = pd.to_numeric(df.loc[:, col], errors="coerce")
            df[col] = df[col].astype(float)

    # encoding asterisks "*" in the /M columns as binary. (* means suspected unreliable measurement)
    for col in df.columns:
        if col.endswith("/M"):
            df.loc[:, col] = (df[col] == "*").astype(int)
            df[col] = df[col].astype(int)

    # exclude columns that end in a .X (X = int)
    df = df.loc[:, ~df.columns.str.contains(r"\.\d+$")]

    len_before = df.shape[0]
    # print("MCHC metrics before removing flagged measurements:")
    # df["MCHC(g/dL)"] = pd.to_numeric(df["MCHC(g/dL)"], errors="coerce")
    # print(df["MCHC(g/dL)"].describe())
    # exclude technically unreliable samples
    df = df[~(df["IP SUS(RBC)Turbidity/HGB Interf?"] == 1)]
    df = df[~(df["IP SUS(RBC)RBC Agglutination?"] == 1)]
    df = df[~(df["IP SUS(PLT)PLT Clumps?"] == 1)]

    for col in standard_fbc_decrypt:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # exclude samples with parallel highly unlikely measurement values
    df = df[
        ~((df["PLT(10^3/uL)"] < 50) & (df["WBC(10^3/uL)"] < 1) & (df["HCT(%)"] < 20))
    ]

    logger.info(
        "Shape after cleaning up columns and dropping unreliable samples: %s", df.shape
    )
    logger.info("Dropped %s rows", len_before - df.shape[0])
    logger.info(
        "Number of unique samples in the dataset: %s", df["Sample No."].nunique()
    )
    # print("MCHC metrics after removing flagged measurements:")
    # print(df["MCHC(g/dL)"].describe())

    # Whether a tested sample was the initial test or a "reflex" test should normally be noted
    # in the "Order Type" column. In INTERVAL there seem to be no reflex entries.
    # For this we will have to look at the distributions of the initial measurements which have the same sample number.
    # Find columns containing the string "reflex" (case-insensitive)
    def contains_reflex(x):
        if isinstance(x, str):
            return "reflex" in x.lower()
        return False

    columns_with_reflex = df.applymap(contains_reflex).any()

    logger.info("Columns containing 'reflex' (case-insensitive):")
    logger.info("%s", columns_with_reflex[columns_with_reflex].index)

    # check for multiple entries of the same sample number (same blood sample, separate measurement)
    id_counts = df["Sample No."].value_counts()
    multiple_entries = id_counts[id_counts > 1]
    logger.info(
        "Number of samples with more than one entry: %s/%s (%.1f%%)",
        len(multiple_entries),
        len(df["Sample No."].unique()),
        100 * len(multiple_entries) / len(df["Sample No."].unique()),
    )

    # IF IT'S MORE THAN ONE: KEEP THE FIRST IF THEY MATCH ON CORE FBC FEATURES (current matching: within 1 std)
    initial_len = df.shape[0]

    for fbc_measurement in standard_fbc_decrypt:
        df[fbc_measurement] = pd.to_numeric(df[fbc_measurement], errors="coerce")

    grouped = df.groupby("Sample No.")
    stds = {}
    for fbc_measurement in standard_fbc_decrypt:
        stds[fbc_measurement] = df[fbc_measurement].std()

    STD_THRESHOLD = 1.0

    multi_sample_rows = []
    odd_samples = []
    only_one_different = []
    logger.info("Checking for multiple measurements of the same sample number.")
    for sample_id, group in tqdm(grouped):
        # Skip groups with less than two rows (i.e. where there is only one sample anyway)
        if len(group) < 2:
            multi_sample_rows.append(group.iloc[:1])
            continue
        if len(group) >= 2:
            group = group.sort_values(["Sample No.", "Date", "Time"], ascending=True)

            # check difference for each standard FBC measurement
            check_list = []
            for fbc_measurement in standard_fbc_decrypt:
                difference = abs(
                    group.iloc[0][fbc_measurement] - group.iloc[1][fbc_measurement]
                )
                check = ~(
                    difference > STD_THRESHOLD * stds[fbc_measurement]
                )  # written this way to keep NaNs
                if not check:
                    logger.info(
                        "Sample %s failed check on %s with value 1 %.2f and value 2 %.2f (threshold difference: %.2f)",
                        sample_id,
                        fbc_measurement,
                        group.iloc[0][fbc_measurement],
                        group.iloc[1][fbc_measurement],
                        STD_THRESHOLD * stds[fbc_measurement],
                    )
                check_list.append(check)
            # only keep measurement if all checks were true
            if sum(check_list) >= len(check_list) - 1:  # at most one check failed
                multi_sample_rows.append(
                    group.iloc[:1]
                )  # use [:1] for formatting (becomes list [0] rather than just element 0)
                if sum(check_list) == len(check_list) - 1:  # only one check failed
                    only_one_different.append(group)
            else:
                odd_samples.append(group)

    # Concatenate all rows at once using pd.concat (faster than appending during loop)
    df = pd.concat(multi_sample_rows)

    # can save samples that failed the check for manual review with a clinician
    try:
        many_samples_df = pd.concat(odd_samples)
    except ValueError:
        many_samples_df = pd.DataFrame()
    try:
        only_one_different_df = pd.concat(only_one_different)
    except ValueError:
        only_one_different_df = pd.DataFrame()

    logger.info(
        "Dropped %s identical sample number rows, keeping the one with the earliest date and time if the two samples matched within %s stds on the standard FBC measurements.",
        initial_len - df.shape[0],
        STD_THRESHOLD,
    )
    logger.info(
        "Number of unique samples in the dataset: %s", df["Sample No."].nunique()
    )

    from sklearn.preprocessing import LabelEncoder

    if len(many_samples_df) > 0:
        many_samples_df["Sample No."] = LabelEncoder().fit_transform(
            many_samples_df["Sample No."].values
        )
        many_samples_df.to_csv(f"{out_dir}/{DATASET}_oddmultiplemeasurements.csv")

    if len(only_one_different_df) > 0:
        only_one_different_df["Sample No."] = LabelEncoder().fit_transform(
            only_one_different_df["Sample No."].values
        )
        only_one_different_df.to_csv(
            f"{out_dir}/{DATASET}_onlyonedifferentmeasurement.csv"
        )

    # Find all columns with non numeric values
    non_numeric_columns = df.columns[df.dtypes == "object"].tolist()
    logger.info(
        "Number of columns with non numeric values: %s", len(non_numeric_columns)
    )
    logger.info("Columns with non numeric values: %s", non_numeric_columns)

    # Some columns contain mostly numerical values but are encoded as `string` (e.g., ['3', '----', '5', '1',...]).
    # $\rightarrow$ if $\ge$ `numeric_threshold` $%$ is numerical and there are $\ge x$ unique values,
    # then convert all values to numerical. If a value can't be converted, then it is set to `nan`.
    # This is the case for some values that are encoded as `['----']` or `['      ']` (6 spaces).
    # - `['      ']` blank appears if a prerequisite for the judgment was not met. Also, if the suspect judgment was not performed due to blank data, etc.
    # - `['----']` analysis impossible: Indicates that an analysis error or a parsing error has occurred, and the value cannot be displayed.

    df.reset_index(drop=True, inplace=True)

    # deal with dashes and spaces directly
    for col in df.columns:
        if "----" in df[col].unique():
            df[col].replace("----", np.nan, inplace=True)
        if "      " in df[col].unique():
            df[col].replace("      ", np.nan, inplace=True)

    # Function to convert dates (day and month) to sine and cosine components
    def date_to_cyclical(date):
        day_of_year = date.dt.dayofyear
        days_in_year = 365 + date.dt.is_leap_year
        return np.sin(2 * np.pi * day_of_year / days_in_year), np.cos(
            2 * np.pi * day_of_year / days_in_year
        )

    # Function to convert time to sine and cosine components
    def time_to_cyclical(time):
        seconds_in_day = 24 * 60 * 60
        time_in_seconds = time.dt.hour * 60 * 60 + time.dt.minute * 60 + time.dt.second
        return np.sin(2 * np.pi * time_in_seconds / seconds_in_day), np.cos(
            2 * np.pi * time_in_seconds / seconds_in_day
        )

    # Print all columns with nan values and the count of nan values
    nan_columns = df.columns[df.isna().any()].tolist()
    print(f"Columns with nan values: {nan_columns}\n")
    print("Percentage of nan values in each column:")
    for col in nan_columns:
        print(f"{col}: {round(df[col].isna().sum()/len(df[col])*100, 2)}%")
    # Total percentage of NaN values
    nan_percentage = (df.isna().sum().sum() / df.size) * 100
    print(f"\nTotal percentage of NaN values: {round(nan_percentage, 2)}%")

    # Manually take out columns we know to be redundant / not informative.
    redundant_hgb = [
        "HGB_NONSI(g/dL)",
        "HGB_SI(mmol/L)",
        "HGB_SI2(mmol/L)",
        "HGB_NONSI2(g/dL)",
        "HGB-S(g/dL)",
    ]

    redundant_cols = redundant_hgb
    df.drop(columns=redundant_cols, inplace=True)

    # cast all columns to numeric and skip if not possible
    logger.info("Converting all columns to numeric.")
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            logger.info(f"Could not convert column {col} to numeric.")

    standard_fbc_features = [
        "PLT(10^3/uL)",
        "MPV(fL)",
        "PCT(%)",
        "RBC(10^6/uL)",
        "HGB(g/dL)",
        "HCT(%)",
        "MCV(fL)",
        "MCH(pg)",
        "MCHC(g/dL)",
        "RDW-SD(fL)",
        "WBC(10^3/uL)",
        "NEUT#(10^3/uL)",
        "LYMPH#(10^3/uL)",
        "MONO#(10^3/uL)",
        "EO#(10^3/uL)",
        "BASO#(10^3/uL)",
        "NEUT%(%)",
        "LYMPH%(%)",
        "MONO%(%)",
        "EO%(%)",
        "BASO%(%)",
    ]

    subset_df = df[standard_fbc_features]

    # Calculate correlation matrix
    correlation_matrix = df.corr(numeric_only=True)

    # Get columns with correlation >= 0.8 with any column in standard_fbc_features
    cols = []
    corr_feats = []
    corr_values = []
    for col in standard_fbc_features:
        # get correlated columns, and save them and the correlation value
        correlated = correlation_matrix[col][
            (correlation_matrix[col] >= 0.8) & (correlation_matrix[col].index != col)
        ].index.tolist()
        for c in correlated:
            cols.append(col)
            corr_feats.append(c)
            corr_values.append(correlation_matrix.loc[col, c])

    correlated_columns = {}
    correlated_columns["Standard FBC feature"] = cols
    correlated_columns["Correlated features"] = corr_feats
    correlated_columns["Correlation strength"] = corr_values
    correlated_columns = pd.DataFrame(correlated_columns)
    correlated_columns.to_csv(
        f"{out_dir}/{DATASET}_correlated_columns.csv", index=False
    )

    logger.info(
        "Columns not in the standard FBC features but highly correlated (>= 0.8) with any column in them:"
    )
    logger.info(correlated_columns.to_string(index=False))

    # We can remove the highly correlated columns if we like.
    # df = df.drop(columns=correlated_columns["Correlated features"].unique())

    new_columns = list(df)
    with open(f"{out_dir}/{DATASET}_newcolumns.txt", "w") as file:
        for item in new_columns:
            file.write("%s\n" % item)

    logger.info("Final dataframe shape after technical cleaning: %s", df.shape)
    logger.info(
        "Final number of unique samples in the dataset: %s", df["Sample No."].nunique()
    )

    cryptpandas.to_encrypted(
        df,
        password=os.environ["IRON_PASSWORD"],
        path=f"{out_dir}/data/{DATASET}_decrypted.crypt",
    )

    df_summary = summarize_dataframe(df)
    df_summary.to_csv(f"{out_dir}/{DATASET}_XNSAMPLE_summary.csv", index=False)
