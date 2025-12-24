import numpy as np
import torch
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, roc_curve

from .metrics import positive_predictive_value


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def ml_clean(df, used_feats: list, dropna: bool = True):
    # cleaning for XN_SAMPLE rich FBC data before ML model
    binary_cols = []
    for col in used_feats:
        if col.startswith("Positive") or col.startswith("Error"):
            binary_cols.append(col)
        if col.startswith("IP "):
            binary_cols.append(col)
        if col.startswith("subject_gender"):
            binary_cols.append(col)
        if col.endswith(f"Abnormal") or col.endswith(f"Suspect"):
            binary_cols.append(col)
        if col.endswith(f"_err") or col.endswith(f"_disc"):
            binary_cols.append(col)
        if col.endswith(f"/M"):
            binary_cols.append(col)
        if col.endswith("_linearity"):
            binary_cols.append(col)
        if col.endswith("_unreliable"):
            binary_cols.append(col)
        if col.endswith("analyser_ID"):
            binary_cols.append(col)
        for timepoint in ["bl", "24m", "48m"]:
            if col.endswith(f"Abnormal_{timepoint}") or col.endswith(
                f"Suspect_{timepoint}"
            ):
                binary_cols.append(col)
            if col.endswith(f"_err_{timepoint}") or col.endswith(f"_disc_{timepoint}"):
                binary_cols.append(col)
            if col.endswith(f"/M_{timepoint}"):
                binary_cols.append(col)

    non_binaries = list(set(used_feats) - set(binary_cols))

    for col in used_feats:
        if col.startswith("Q-Flag"):
            df[col] = df[col].fillna(0)

    if dropna:
        df_return = df.dropna(subset=non_binaries)
    else:
        df_return = df

    df_return.loc[:, binary_cols] = df_return[binary_cols].fillna(0)

    # check that binary columns have at most 2 unique values
    for col in binary_cols:
        assert (
            df_return[col].nunique() <= 2
        ), f"{col} has more than 2 unique values, they are {df_return[col].unique()}"

    return df_return, binary_cols, non_binaries


def top_permutation_importance(model, x, y, feature_names):
    r = permutation_importance(
        model,
        x,
        y,
        n_repeats=10,
        # random_state=0,
        n_jobs=-1,
    )
    top_counter = 0
    top_features = []
    for i in r.importances_mean.argsort()[::-1]:
        if top_counter < 10 and r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(
                f"{feature_names[i]:<8}  "
                f"{r.importances_mean[i]:.3f}  "
                f" +/- {r.importances_std[i]:.3f}"
            )
            top_counter += 1
            top_features.append(
                {
                    "Feature": feature_names[i],
                    "Importance": r.importances_mean[i],
                    "Std": r.importances_std[i],
                }
            )
    return top_features


def find_best_thresholds(y_true, y_proba, n_classes):
    thresholds = []
    for i in range(n_classes):
        # Create binary labels for class i vs rest
        y_true_binary = (y_true == i).astype(int)
        fpr, tpr, threshold = roc_curve(y_true_binary, y_proba[:, i])

        # Find the threshold closest to the top-left (0,1) corner
        distances_roc = np.sqrt(fpr**2 + (1 - tpr) ** 2)
        idx = np.argmin(distances_roc)
        best_thresh = threshold[idx]
        thresholds.append(best_thresh)

    return thresholds


def find_lowest_threshold_for_ppv(
    y_true, y_proba, class_label, ppv_target=0.95, return_precision: bool = False
):
    y_proba_class = y_proba[:, class_label]
    y_true_binary = (y_true == class_label).astype(int)
    precision, recall, thresholds = precision_recall_curve(y_true_binary, y_proba_class)

    # Find all indices where precision >= ppv_target
    valid_indices = np.where(precision >= ppv_target)[0]

    if len(valid_indices) == 1:
        print(
            f"Warning: No threshold achieves the target PPV of {ppv_target}. Returning threshold for max precision."
        )
        if return_precision:
            return thresholds[np.argmax(precision[:-1])], np.max(precision[:-1])
        return thresholds[np.argmax(precision[:-1])]

    # Return the threshold corresponding to the lowest valid index
    if return_precision:
        return thresholds[valid_indices[0]], precision[valid_indices[0]]
    return thresholds[valid_indices[0]]


# Add a docstring for better documentation
find_lowest_threshold_for_ppv.__doc__ = """
Find the lowest threshold that achieves a target Positive Predictive Value (PPV) for a given class.

Args:
    y_true (array-like): True labels.
    y_proba (array-like): Predicted probabilities.
    class_label (int): The class label to focus on.
    ppv_target (float, optional): The target PPV to achieve. Defaults to 0.95.

Returns:
    float: The lowest threshold that achieves the target PPV, or the threshold for max precision if target cannot be met.
"""


def get_algorithm_class(
    algorithm_name: str, input_shape: int, n_classes: int, random_state: int = 42
):
    if algorithm_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            bootstrap=True,
            criterion="gini",
            max_depth=10,
            max_features="sqrt",
            min_samples_leaf=2,
            min_samples_split=5,
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        )  # found through TPOT + W&B Hyperparameter Tuning
    elif algorithm_name == "mlp":
        from .models import PyTorchClassifier

        return PyTorchClassifier(
            input_dim=input_shape,
            output_dim=n_classes,
            hidden_dims=[input_shape, input_shape // 2, input_shape // 4],
            dropout=0.2,
        )
    elif algorithm_name == "xgboost":
        from xgboost import XGBClassifier

        # return XGBClassifier(
        #     learning_rate=0.01,
        #     max_depth=9,
        #     min_child_weight=5,
        #     n_estimators=100,
        #     n_jobs=-1,
        #     subsample=0.5,
        #     verbosity=0,
        #     random_state=random_state,
        # )  # found through TPOT
        return XGBClassifier(
            colsample_bytree=0.837,
            gamma=0.055,
            learning_rate=0.03,
            max_depth=12,
            min_child_weight=2,
            n_estimators=350,
            n_jobs=-1,
            reg_alpha=0.005,
            reg_lambda=0.03,
            subsample=0.8,
        )  # found through W&B
    elif algorithm_name == "tabpfn":
        device = get_device()
        print(f"Using device: {device} for TabPFN")
        from tabpfn import TabPFNClassifier

        return TabPFNClassifier(device=device)
    elif algorithm_name == "autotabpfn":
        device = get_device()
        print(f"Using device: {device} for AutoTabPFN")
        from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
            AutoTabPFNClassifier,
        )

        return AutoTabPFNClassifier(
            max_time=600, device=device
        )  # 10 minutes tuning time
    else:
        raise NotImplementedError(f"Algorithm {algorithm_name} not recognised")


class EnsembleClassifier:
    def __init__(self, estimators: dict, train_scalers: list = None):
        self.estimators = estimators
        self.scalers = train_scalers

    def predict_proba(self, X):
        if self.scalers is not None:
            return np.mean(
                [
                    model.predict_proba(self.scalers[i].transform(X))
                    for i, model in enumerate(self.estimators.values())
                ],
                axis=0,
            )
        return np.mean(
            [model.predict_proba(X) for model in self.models.values()], axis=0
        )

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
