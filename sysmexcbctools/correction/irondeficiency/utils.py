import contextlib

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_curve
from tqdm import tqdm

from tqdm.auto import tqdm
from joblib import Parallel

from ._feature_lists_decrypted import (
    baseline_blood_feats,
    rich_with_units,
    twoyear_blood_feats,
    fouryear_blood_feats,
)


class reference_range_ID_predictor:
    def __init__(self):
        self.reference_ranges = {
            "HGB_male": {"ReferenceLow": 130, "ReferenceHigh": 180},
            "HGB_female": {"ReferenceLow": 115, "ReferenceHigh": 165},
            "MCV": {"ReferenceLow": 80, "ReferenceHigh": 100},
            "MCH": {"ReferenceLow": 27, "ReferenceHigh": 32},
        }
        pass

    def predict(self, df, HGB_col_name, MCV_col_name, MCH_col_name, sex_col_name):
        df = self._prepare_dataframe(
            df, HGB_col_name, MCV_col_name, MCH_col_name, sex_col_name
        )
        df["iron_class"] = df.apply(self._predict_one_row, axis=1)
        return df["iron_class"].values

    def predict_singles(
        self, df, HGB_col_name, MCV_col_name, MCH_col_name, sex_col_name
    ):
        df = self._prepare_dataframe(
            df, HGB_col_name, MCV_col_name, MCH_col_name, sex_col_name
        )
        df["iron_class"] = df.apply(self._predict_one_row_singles, axis=1)
        return df["iron_class"].values

    def _prepare_dataframe(
        self, df, HGB_col_name, MCV_col_name, MCH_col_name, sex_col_name
    ):
        df = df[[HGB_col_name, MCV_col_name, MCH_col_name, sex_col_name]]
        df.rename(
            columns={
                HGB_col_name: "HGB",
                MCV_col_name: "MCV",
                MCH_col_name: "MCH",
                sex_col_name: "Sex",
            },
            inplace=True,
        )
        return df

    def _check_reference_threshold_one_row(self, row):
        trigger_dict = {"HGB_low": False, "MCV_low": False, "MCH_low": False}
        if row["Sex"] == "M":
            if row["HGB"] < self.reference_ranges["HGB_male"]["ReferenceLow"]:
                trigger_dict["HGB_low"] = True
        elif row["Sex"] == "F":
            if row["HGB"] < self.reference_ranges["HGB_female"]["ReferenceLow"]:
                trigger_dict["HGB_low"] = True

        if row["MCV"] < self.reference_ranges["MCV"]["ReferenceLow"]:
            trigger_dict["MCV_low"] = True

        if row["MCH"] < self.reference_ranges["MCH"]["ReferenceLow"]:
            trigger_dict["MCH_low"] = True

        return trigger_dict

    def _predict_one_row(self, row):
        trigger_dict = self._check_reference_threshold_one_row(row)
        if (
            trigger_dict["HGB_low"]
            and trigger_dict["MCV_low"]
            and trigger_dict["MCH_low"]
        ):
            return 1
        else:
            return 0

    def _predict_one_row_singles(self, row):
        trigger_dict = self._check_reference_threshold_one_row(row)
        if (
            trigger_dict["HGB_low"]
            or trigger_dict["MCV_low"]
            or trigger_dict["MCH_low"]
        ):
            return 1
        else:
            return 0


def plot_feature_importance_logreg(model, feature_names: list, top_n: int = 5):
    sorted_indices = sorted(
        range(len(model.coef_.flatten())),
        key=lambda i: abs(model.coef_.flatten()[i]),
        reverse=True,
    )
    sorted_coefficients = [model.coef_.flatten()[i] for i in sorted_indices]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    colours = ["red" if c < 0 else "blue" for c in sorted_coefficients]
    plt.barh(
        range(len(sorted_coefficients[:top_n])),
        sorted_coefficients[:top_n],
        color=colours,
    )
    plt.yticks(range(len(sorted_coefficients[:top_n])), sorted_feature_names[:top_n])
    plt.ylim(-1, top_n)
    plt.ylabel(f"Top {top_n} Features")
    plt.xlabel("Coefficient Magnitude")
    plt.gca().invert_yaxis()
    plt.tight_layout()


def classify_baseline_iron_status(row):
    ferr = row["FERR_bl"]
    crp = row["CRP_bl"]
    iron = row["IRON_bl"]
    tsat = (iron / row["TRANSF_bl"]) * 70.9  # PubMed 20722575
    # tsat = row["TSAT_bl"]
    if ferr < 15:
        return 1
    elif (crp > 5) and (ferr < 150) and (tsat < 15):
        return 1
    elif (crp == np.nan) or (ferr == np.nan) or (tsat == np.nan):
        return np.nan
    else:
        return 0


def classify_iron_status_strict(row, timestring: str):
    ferr = row["FERR" + timestring]
    crp = row["CRP" + timestring]
    if ferr < 15:
        return 1
    if (crp < 5) and (ferr > 30):
        return 0
    return np.nan


def make_sex_iron_class(row):
    sex = row["subject_gender"]
    iron_status = row["baseline_iron_class"]
    if (sex == "M") and (iron_status == 0):
        return "Iron replete male"
    elif (sex == "M") and (iron_status == 1):
        return "Iron deficient male"
    elif (sex == "F") and (iron_status == 0):
        return "Iron replete female"
    elif (sex == "F") and (iron_status == 1):
        return "Iron deficient female"
    else:
        pass


def add_menopause_updated_sex(row):
    sex = row["subject_gender"]
    meno = row["menopause_bl"]
    if sex == "M":
        return "M"
    elif sex == "F":
        if meno == "yes":
            return "F - post-menopausal"
        elif meno == "no":
            return "F - pre-menopausal"
        else:
            return np.nan
    else:
        return np.nan


def make_normalranges_tablerow(blood_feature, df):
    table_row = []
    table_row.append(blood_feature)
    table_row.append(df[blood_feature].median())
    table_row.append(df[blood_feature].median() - 2 * df[blood_feature].std())
    table_row.append(df[blood_feature].median() + 2 * df[blood_feature].std())
    table_row.append(
        stats.ttest_ind(
            df[blood_feature][df["subject_gender"] == "M"].values,
            df[blood_feature][df["subject_gender"] == "F"].values,
            equal_var=False,
            nan_policy="omit",
        ).statistic
    )
    table_row.append(
        stats.ttest_ind(
            df[blood_feature][df["menopause_bl"] == "no"].values,
            df[blood_feature][df["menopause_bl"] == "yes"].values,
            equal_var=False,
            nan_policy="omit",
        ).statistic
    )
    table_row.append(
        stats.ttest_ind(
            df[blood_feature][df["subject_ethnicity"] == "white"].values,
            df[blood_feature][df["subject_ethnicity"] == "non-white"].values,
            equal_var=False,
            nan_policy="omit",
        ).statistic
    )
    table_row.append(
        stats.ttest_ind(
            df[blood_feature][df["crp_over_5"] == True].values,
            df[blood_feature][df["crp_over_5"] == False].values,
            equal_var=False,
            nan_policy="omit",
        ).statistic
    )
    table_row.append(
        stats.ttest_ind(
            df[blood_feature][df["baseline_iron_class"] == 0].values,
            df[blood_feature][df["baseline_iron_class"] == 1].values,
            equal_var=False,
            nan_policy="omit",
        ).statistic
    )
    return table_row


def make_normalranges_comparison_tablerow(blood_feature, df, df2):
    table_row = []
    table_row.append(blood_feature)
    table_row.append(df[blood_feature].median())
    table_row.append(df[blood_feature].median() - 2 * df[blood_feature].std())
    table_row.append(df[blood_feature].median() + 2 * df[blood_feature].std())
    table_row.append(df2[blood_feature].median())
    table_row.append(df2[blood_feature].median() - 2 * df2[blood_feature].std())
    table_row.append(df2[blood_feature].median() + 2 * df2[blood_feature].std())
    table_row.append(
        stats.ttest_ind(
            df[blood_feature].values,
            df2[blood_feature].values,
            equal_var=False,
            nan_policy="omit",
        ).statistic
    )
    table_row.append(
        stats.ttest_ind(
            df[blood_feature].values,
            df2[blood_feature].values,
            equal_var=False,
            nan_policy="omit",
        ).pvalue
    )
    return table_row


def format_code_variable_string(input_string):
    if input_string in baseline_blood_feats:
        idx = baseline_blood_feats.index(input_string)
        return rich_with_units[idx]
    # Replace underscores with "\_"
    formatted_string = input_string.replace("_", "\\_")
    # Wrap the formatted string with "\texttt{}"
    latex_formatted_string = "\\texttt{" + formatted_string + "}"
    return latex_formatted_string


def format_p_value(p_value):
    if p_value < 0.0001:
        return r"$<0.0001$"
    else:
        return f"{p_value:.4f}"


def get_threshold_top_left_roc(model, x, y_true):
    y_scores = model.predict_proba(x)[:, 1]
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    distances_roc = np.sqrt(fpr**2 + (1 - tpr) ** 2)
    index_roc = np.argmin(distances_roc)
    return thresholds_roc[index_roc]


def mad(x: pd.core.series.Series) -> float:
    """calculate Median Absolute Deviation for x"""
    return (x - x.median()).abs().median()


def centralise(
    df: pd.core.frame.DataFrame, x: pd.core.series.Series, threshold: float
) -> pd.core.frame.DataFrame:
    """keep only the data differing from the median by
    less than threshold times median absolute deviations"""
    low = x.median() - threshold * mad(x)
    high = x.median() + threshold * mad(x)
    return df[x.between(low, high, inclusive="neither")].copy()


def summarize_dataframe(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    summary = []

    for col in df.columns:
        unique_values = df[col].nunique()
        top_values_series = df[col].value_counts().head(min(5, unique_values))

        # Check if column is numeric to round top values
        if pd.api.types.is_numeric_dtype(df[col]):
            # Round each top value individually
            rounded_top_values = [
                round(float(value), 2) for value in top_values_series.index
            ]
        else:
            rounded_top_values = top_values_series.index.tolist()

        # Initialize the dictionary with common information
        col_summary = {
            "Column": col,
            "Unique Values": unique_values,
            "Top Values": rounded_top_values,
            "Frequencies": top_values_series.values.tolist(),
        }

        # Add statistical summaries for numerical columns
        if pd.api.types.is_numeric_dtype(df[col]):
            col_summary["Min"] = round(df[col].min(), 2)
            col_summary["Max"] = round(df[col].max(), 2)
            col_summary["Mean"] = round(df[col].mean(), 2)
            col_summary["Median"] = round(df[col].median(), 2)

        summary.append(col_summary)

    return pd.DataFrame(summary)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def custom_round(number, base=5):
    return base * round(number / base)
