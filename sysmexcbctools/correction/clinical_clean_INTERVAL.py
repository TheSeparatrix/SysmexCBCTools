import argparse
import importlib
import os

import cryptpandas
import dvc.api
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from joblib import Parallel, delayed
from scipy.special import expit, logit
from tqdm import tqdm

sns.set_theme()
sns.set(rc={"figure.dpi": 150})
sns.set_style("whitegrid", {"legend.frameon": True, "grid.linestyle": "--"})

from irondeficiency.gam_functions import GAM_predict, spline_fit
from irondeficiency.shape_tracker import log_data_shape
from irondeficiency.utils import mad, tqdm_joblib


def do_fit(df, col, timepoint, analyser_ids):
    fit_df = df.loc[
        :,
        [col, f"FBC_analyser_ID_{timepoint}"] + pred_features,
    ].dropna()
    print("Fit df shape:", fit_df.shape)
    try:
        gams = spline_fit(fit_df, col, timepoint, analyser_ids)
        return gams
    except ValueError:
        print(f"Not enough unique values in column {col}, skipping...")
        return col, None, None


ret_channel_features = [
    "RET_PCT_bl",
    "RET_PCT_24m",
    "RET_PCT_48m",
    "RET_10_6_uL_bl",
    "RET_10_6_uL_24m",
    "RET_10_6_uL_48m",
    "RET_He_pg_bl",
    "RET_He_pg_24m",
    "RET_He_pg_48m",
    "RET_Y_ch_bl",
    "RET_Y_ch_24m",
    "RET_Y_ch_48m",
    "RET_RBC_Y_ch_bl",
    "RET_RBC_Y_ch_24m",
    "RET_RBC_Y_ch_48m",
    "RET_UPP_bl",
    "RET_UPP_24m",
    "RET_UPP_48m",
    "RET_TNC_bl",
    "RET_TNC_24m",
    "RET_TNC_48m",
    "IRF_PCT_bl",
    "IRF_PCT_24m",
    "IRF_PCT_48m",
    "IRF_Y_ch_bl",
    "IRF_Y_ch_24m",
    "IRF_Y_ch_48m",
    "LFR_PCT_bl",
    "LFR_PCT_24m",
    "LFR_PCT_48m",
    "MFR_PCT_bl",
    "MFR_PCT_24m",
    "MFR_PCT_48m",
    "HFR_PCT_bl",
    "HFR_PCT_24m",
    "HFR_PCT_48m",
    "IPF_bl",
    "IPF_24m",
    "IPF_48m",
    "IPFx_10_9_L_bl",
    "IPFx_10_9_L_24m",
    "IPFx_10_9_L_48m",
]

if __name__ == "__main__":
    params = dvc.api.params_show()
    out_dir = params["out_dir"]
    outlier_MAD_threshold = params["outlier_MAD_threshold"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type", help="Whether decrypted or export INTERVAL", default="decrypted"
    )
    args = parser.parse_args()

    feature_lists = importlib.import_module(
        f"irondeficiency._feature_lists_{args.type}"
    )
    baseline_blood_feats = getattr(feature_lists, "baseline_blood_feats")
    standard_fbc = getattr(feature_lists, "standard_fbc")

    assert args.type in ["decrypted", "export"], "Invalid type argument"

    SHAPE_TRACKER_DIR = f"outs/preprocessing_prisma/interval_{args.type}/"
    STEP_NAME = "02_clinical_clean_INTERVAL"

    df = cryptpandas.read_encrypted(
        path=f"{out_dir}/data/INTERVAL_queried_{args.type}.crypt",
        password=os.environ["IRON_PASSWORD"],
    )

    print("Initial data shape:", df.shape)
    log_data_shape(STEP_NAME, df, "data_loading", SHAPE_TRACKER_DIR)

    # Step 1: Identify timepoints with valid sample ages (between 0 and 30 hours inclusive)
    MAX_SAMPLE_AGE_H = 36
    print(f"Identifying valid samples (age between 0-{MAX_SAMPLE_AGE_H}h)")
    bl_valid_age = (df["FBC_sampleage_h_bl"] >= 0) & (
        df["FBC_sampleage_h_bl"] <= MAX_SAMPLE_AGE_H
    )
    m24_valid_age = (df["FBC_sampleage_h_24m"] >= 0) & (
        df["FBC_sampleage_h_24m"] <= MAX_SAMPLE_AGE_H
    )
    m48_valid_age = (df["FBC_sampleage_h_48m"] >= 0) & (
        df["FBC_sampleage_h_48m"] <= MAX_SAMPLE_AGE_H
    )

    # Diagnostic checks before invalidating measurements
    print(f"Samples with invalid baseline age: {(~bl_valid_age).sum()}")
    print(f"Samples with invalid 24m age: {(~m24_valid_age).sum()}")
    print(f"Samples with invalid 48m age: {(~m48_valid_age).sum()}")
    log_data_shape(STEP_NAME, df, "identified_invalid_sample_ages", SHAPE_TRACKER_DIR)

    # Step 2: Invalidate FBC measurements for timepoints with invalid sample ages
    print(
        f"Invalidating FBC measurements for samples outside 0-{MAX_SAMPLE_AGE_H}h age range"
    )
    df.loc[~bl_valid_age, "HGB_g_L_bl"] = np.nan
    df.loc[~m24_valid_age, "HGB_g_L_24m"] = np.nan
    df.loc[~m48_valid_age, "HGB_g_L_48m"] = np.nan
    log_data_shape(
        STEP_NAME, df, "invalidated_fbc_for_bad_sample_ages", SHAPE_TRACKER_DIR
    )

    # Diagnostic checks after invalidating measurements
    print(
        f"Samples missing baseline HGB after age validation: {df['HGB_g_L_bl'].isna().sum()}"
    )
    print(f"Samples missing baseline FERR: {df['FERR_bl'].isna().sum()}")
    print(
        f"Samples missing 24m HGB after age validation: {df['HGB_g_L_24m'].isna().sum()}"
    )
    print(f"Samples missing 24m FERR: {df['FERR_24m'].isna().sum()}")
    print(
        f"Samples missing 48m HGB after age validation: {df['HGB_g_L_48m'].isna().sum()}"
    )
    print(f"Samples missing 48m FERR: {df['FERR_48m'].isna().sum()}")

    # Step 3: Identify timepoints with both valid FBC and ferritin
    print("Identifying timepoints with both valid FBC and ferritin")
    bl_complete = df["HGB_g_L_bl"].notna() & df["FERR_bl"].notna()
    m24_complete = df["HGB_g_L_24m"].notna() & df["FERR_24m"].notna()
    m48_complete = df["HGB_g_L_48m"].notna() & df["FERR_48m"].notna()

    print(f"Samples with complete baseline data: {bl_complete.sum()}")
    print(f"Samples with complete 24m data: {m24_complete.sum()}")
    print(f"Samples with complete 48m data: {m48_complete.sum()}")
    log_data_shape(STEP_NAME, df, "identified_complete_timepoints", SHAPE_TRACKER_DIR)

    # Step 4: Keep donors with at least one complete timepoint
    print("Filtering to keep donors with at least one complete timepoint")
    len_before = len(df)
    df = df[bl_complete | m24_complete | m48_complete]
    print(f"Dropped {len_before - len(df)} donors")
    log_data_shape(
        STEP_NAME,
        df,
        "keep_donors_with_at_least_one_complete_timepoint",
        SHAPE_TRACKER_DIR,
    )

    # Final summary of remaining data
    valid_bl = df["HGB_g_L_bl"].notna() & df["FERR_bl"].notna()
    valid_m24 = df["HGB_g_L_24m"].notna() & df["FERR_24m"].notna()
    valid_m48 = df["HGB_g_L_48m"].notna() & df["FERR_48m"].notna()

    print("=== SUMMARY OF REMAINING DATA ===")
    print(f"Total remaining donors: {len(df)}")
    print(
        f"Donors with valid baseline data: {valid_bl.sum()} ({valid_bl.sum()/len(df)*100:.1f}%)"
    )
    print(
        f"Donors with valid 24m data: {valid_m24.sum()} ({valid_m24.sum()/len(df)*100:.1f}%)"
    )
    print(
        f"Donors with valid 48m data: {valid_m48.sum()} ({valid_m48.sum()/len(df)*100:.1f}%)"
    )
    print(
        f"Donors with valid data at all three timepoints: {(valid_bl & valid_m24 & valid_m48).sum()}"
    )
    print("================================")
    # df = df[~df["HGB_g_L_bl"].isna()]
    # log_data_shape(STEP_NAME, df, "drop_no_baseline_fbc", SHAPE_TRACKER_DIR)
    # df = df[~df["FERR_bl"].isna()]
    # log_data_shape(STEP_NAME, df, "drop_no_baseline_ferritin", SHAPE_TRACKER_DIR)
    print(f"Dropped {len_before - len(df)} rows")

    # print("Dropping rows with platelet clumping flag")
    # len_before = len(df)
    # df = df[df["IP SUS(PLT)PLT Clumps?_1.0_binary_bl"] != 1.0]
    # # df = df[df["IP SUS(PLT)PLT Clumps?_1.0_binary_24m"] != 1.0]
    # # df = df[df["IP SUS(PLT)PLT Clumps?_1.0_binary_48m"] != 1.0]
    # print(f"Dropped {len_before - len(df)} rows")

    # # drop all entries with MPV > 13 as these are flagged as unreliable by Sysmex instruments (Sysmex only)
    # print("Dropping rows with MPV > 13 or < 0")
    # len_before = len(df)
    # df = df[~((df["MPV_fL_bl"] > 13) | (df["MPV_fL_bl"] < 0))]
    # # df = df[~((df["MPV_fL_24m"] > 13) | (df["MPV_fL_24m"] < 0))]
    # # df = df[~((df["MPV_fL_48m"] > 13) | (df["MPV_fL_48m"] < 0))]
    # print(f"Dropped {len_before - len(df)} rows")

    # exclude impossibilities
    # df = df[~(df["FERR_bl"] < 1.0)]
    print("Keeping rows with 0.5m < height < 2.5m or 30kg < weight < 190kg")
    len_before = len(df)
    df = df[~((df["subject_height_bl"] > 2.5) | (df["subject_height_bl"] < 0.5))]
    df = df[~((df["subject_weight_bl"] > 190.0) | (df["subject_weight_bl"] < 30.0))]
    print(f"Dropped {len_before - len(df)} rows")
    log_data_shape(STEP_NAME, df, "drop_unrealistic_height_weight", SHAPE_TRACKER_DIR)
    # df = df[df["PLT_10_9_L_bl"] > 50]

    # grabbing WBC for adjustment figure
    wbc_bl = df[["WBC_10_9_L_bl", "FBC_sampleage_h_bl"]].dropna()
    wbc_bl["Adjustment"] = "Before"

    # we will need to adjust non-binary features based on Will's analysis
    non_adjust_columns = []
    for column in df.columns:
        if (
            (df[column].nunique() < 3)
            or (column.startswith("Q-Flag"))
            or (column.startswith("FBC"))
            or (column.startswith("subject"))
            or ("mean" in column)
            or ("cov" in column)
        ):
            non_adjust_columns.append(column)

    non_adjust_columns = non_adjust_columns + [
        "blood6m_bl",
        "blood2y_bl",
        "bloodLife_bl",
        "CRP_bl",
        "FERR_bl",
        "IRON_bl",
        "TRANSFS_bl",
        "UIBC_bl",
        "TRANSF_bl",
        "HEP_bl",
        "menopause_bl",
        "Sample No._bl",
        "ironFreq_24m",
        "CRP_24m",
        "FERR_24m",
        "HEP_24m",
        "Sample No._24m",
        "CRP_48m",
        "FERR_48m",
        "Sample No._48m",
        "analyser_datetime_bl",
        "analyser_datetime_24m",
        "analyser_datetime_48m",
    ]

    analyser_ids = ["XN-10^11041", "XN-10^11036"]

    logit_cols = []
    for timepoint in ["bl", "24m", "48m"]:
        print("GAM-adjusting data at timepoint", timepoint)
        adjust_cols = []
        for col in df.columns:
            if col.endswith(f"_{timepoint}") and col not in non_adjust_columns:
                adjust_cols.append(col)

        for col in adjust_cols:
            if col.endswith(f"PCT_{timepoint}") or "%" in col:
                # check that all entries are between 0 and 1
                if df[col].min() >= 0 and df[col].max() <= 1:
                    logit_cols.append(col)
                else:
                    df[col] = df[col] / 100
                    logit_cols.append(col)

        pred_features = [
            f"FBC_t_d_{timepoint}",
            f"FBC_tday_h_{timepoint}",
            f"FBC_sampleage_h_{timepoint}",
            # f"FBC_tyear_d_{timepoint}",
            f"FBC_weekday_{timepoint}",
        ]

        for col in adjust_cols:
            # print("Attempting adjustment of column", col)
            if col in logit_cols:
                # df[col] = np.log(1e-7 + df[col] / (1e-7 + 1 - df[col]))
                df[col] = logit(1e-7 + df[col])
            elif df[col].min() < 0:
                print("Not adjusting column", col, "as it contains negative values")
            elif ("mean" in col) or ("cov" in col):
                print(
                    "Not adjusting column",
                    col,
                    "as it is a mean or covariance (flow cytometry data)",
                )
            else:
                df[col] = np.log(1e-7 + df[col])

        log_data_shape(
            STEP_NAME, df, f"prepare_df_for_GAM_{timepoint}", SHAPE_TRACKER_DIR
        )

        with tqdm_joblib(
            tqdm(desc="Fitting GAMs", total=len(adjust_cols))
        ) as progress_bar:
            gam_results = Parallel(n_jobs=-1)(
                delayed(do_fit)(df, col, timepoint, analyser_ids) for col in adjust_cols
            )
        for col, gam1, gam2 in gam_results:
            if col in ret_channel_features:
                adjustment_condition = (df[f"FBC_sampleage_h_{timepoint}"] <= 18) & (
                    df[f"FBC_analyser_ID_{timepoint}"]
                    == "XN-10^11036"  # other machine had problems in the RET channel and is not reliable
                )
            else:
                adjustment_condition = df[f"FBC_sampleage_h_{timepoint}"] <= 18
            if gam1 is not None:
                adjustable_mask = (
                    df.loc[:, col].notnull()
                    & df.loc[:, f"FBC_analyser_ID_{timepoint}"].notnull()
                    & df.loc[:, pred_features].notna().all(axis=1)
                    & (df.loc[:, f"FBC_sampleage_h_{timepoint}"] <= MAX_SAMPLE_AGE_H)
                )
                df.loc[adjustable_mask, col] = df.loc[adjustable_mask, col] - (
                    GAM_predict(
                        df[adjustable_mask], gam1, gam2, timepoint, analyser_ids
                    )
                    - df.loc[
                        adjustment_condition,
                        col,
                    ].mean()
                )

        log_data_shape(
            STEP_NAME, df, f"adjust_df_with_GAM_{timepoint}", SHAPE_TRACKER_DIR
        )
        # for col in adjust_cols:
        #     fit_df = df.loc[
        #         :,
        #         [col, f"FBC_analyser_ID_{timepoint}"] + pred_features,
        #     ].dropna()
        #     try:
        #         gams = spline_fit(
        #             fit_df, col, timepoint, analyser_ids
        #         )  # adjust data based on GAM prediction, only on rows where we have all information
        #         adjustable_mask = (
        #             df.loc[:, col].notnull()
        #             & df.loc[:, f"FBC_analyser_ID_{timepoint}"].notnull()
        #             & df.loc[:, pred_features].notna().all(axis=1)
        #         )
        #         df.loc[adjustable_mask, col] = df.loc[adjustable_mask, col] - (
        #             GAM_predict(
        #                 df[adjustable_mask], gams[1], gams[2], timepoint, analyser_ids
        #             )
        #             - df.loc[
        #                 (df[f"FBC_sampleage_h_{timepoint}"] <= 18)
        #                 & (df[f"FBC_analyser_ID_{timepoint}"] == "XN-10^11036"),
        #                 col,
        #             ].mean()
        #         )  # only use less than 18 hour old samples for data mean, also only use XN-10^11036 as other machine had abberant period
        #     except ValueError:
        #         print(f"Not enough unique values in column {col}, skipping...")

    # make a new dataframe with all rows of df where any of the adjust_cols columns are over 4.5 median absolute deviations away from their column median
    print("Dropping rows with outliers")
    len_before = len(df)

    adjust_cols = []
    for col in df.columns:
        if (
            col.endswith("_bl") or col.endswith("_24m") or col.endswith("_48m")
        ) and col not in non_adjust_columns:
            adjust_cols.append(col)

    outlier_mask = np.zeros(len(df), dtype=bool)
    standard_fbc_bl = [string + "_bl" for string in standard_fbc]
    standard_fbc_24m = [string + "_24m" for string in standard_fbc]
    standard_fbc_48m = [string + "_48m" for string in standard_fbc]

    for col in standard_fbc_bl + standard_fbc_24m + standard_fbc_48m:
        condition_column = np.array(["" for _ in range(len(df))])
        condition_high = (df[col] - df[col].median()) > outlier_MAD_threshold * mad(
            df[col]
        )
        condition_low = (df[col] - df[col].median()) < outlier_MAD_threshold * mad(
            df[col]
        )
        outlier_mask = outlier_mask | condition_high | condition_low
        condition_column[condition_high] = "high"
        condition_column[condition_low] = "low"
        df[f"{col}_outlier"] = condition_column

    df = df.copy()  # defragmenting

    # rescale df back to original
    for col in adjust_cols:
        if col in logit_cols:
            # df[col] = df[col] / (1 + np.exp(df[col]))
            df[col] = expit(df[col])
        else:
            df[col] = np.exp(df[col])
        if col.startswith("RET_PCT"):
            print(df[col].describe())

    log_data_shape(STEP_NAME, df, "rescale_df", SHAPE_TRACKER_DIR)

    outlier_df = df[outlier_mask]
    # df = df[~outlier_mask] #! skipping outlier removal for now
    print(f"Dropped {len_before - len(df)} rows")
    log_data_shape(STEP_NAME, df, "drop_outliers", SHAPE_TRACKER_DIR)
    outlier_df.to_csv("data/INTERVAL_outliers.csv")

    # drop outlier description columns again
    for col in standard_fbc_bl + standard_fbc_24m + standard_fbc_48m:
        df.drop(f"{col}_outlier", axis=1, inplace=True)

    log_data_shape(STEP_NAME, df, "drop_outlier_description_columns", SHAPE_TRACKER_DIR)

    print("Final data shape:", df.shape)

    cryptpandas.to_encrypted(
        df,
        password=os.environ["IRON_PASSWORD"],
        path=f"{out_dir}/data/INTERVAL_clean_{args.type}.crypt",
    )

    wbc_bl_after = df[["WBC_10_9_L_bl", "FBC_sampleage_h_bl"]].dropna()
    wbc_bl_after["Adjustment"] = "After"

    wbc_bl = pd.concat([wbc_bl, wbc_bl_after])
    wbc_bl["FBC_sampleage_h_bl"] = wbc_bl["FBC_sampleage_h_bl"].round()
    sns.lineplot(
        data=wbc_bl,
        x="FBC_sampleage_h_bl",
        y="WBC_10_9_L_bl",
        hue="Adjustment",
        style="Adjustment",
        markers=True,
        dashes=False,
    )
    plt.xlabel("Delay between sample acquisition and analysis (hours)")
    plt.ylabel(r"White blood cell count ($10^9$ L$^{-1}$)")
    # plt.tight_layout()
    plt.savefig(f"outs/figures/GAM_adjustment_figure_{args.type}.pdf")

    print(f"Number of baselines: {sum(~df['HGB_g_L_bl'].isna())}")
    print(f"Number of 24 months: {sum(~df['HGB_g_L_24m'].isna())}")
    print(f"Number of 48 months: {sum(~df['HGB_g_L_48m'].isna())}")
