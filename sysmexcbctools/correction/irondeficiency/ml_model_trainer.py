from sklearn.model_selection import cross_val_predict, KFold, GroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn import preprocessing
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np


def cross_val_training_ferritin(
    model,
    X: pd.DataFrame,
    y: pd.DataFrame,
    non_binary_indices,
    target,
    ferritin_col_name,
    target_probas: list,
    balance_data="smote",
    n_splits: int = 5,
    random_state: int = 42,
    grouping=None,
):
    """
    Train the model using cross-validation.
    """
    label_type = "hard"
    if len(target_probas) > 0:
        label_type = "soft"

    print("target:", target)
    print("ferritin_col_name:", ferritin_col_name)
    print("target_probas:", target_probas)
    print("label_type:", label_type)

    cv = GroupKFold(n_splits=n_splits)
    y_preds = []
    y_pred_probas = []
    y_tests = []
    y_ferritins = []
    feature_importances = []
    for fold, (train, test) in enumerate(cv.split(X, y, groups=grouping)):
        print(f"Fold {fold+1}/{n_splits}")
        X_train, X_val = X.iloc[train, :].values, X.iloc[test, :].values
        y_train, y_val = y.iloc[train, :], y.iloc[test, :]
        y_ferritin = y_val[ferritin_col_name].astype(float).values
        y_train_ferritin = y_train[ferritin_col_name].astype(float).values
        y_val = y_val[target].astype(int).values
        if label_type == "hard":
            y_train = y_train[target].astype(int).values
        else:
            y_train = y_train[target_probas].astype(float).values

        train_scaler = preprocessing.StandardScaler().fit(
            X_train[:, non_binary_indices]
        )
        X_train[:, non_binary_indices] = train_scaler.transform(
            X_train[:, non_binary_indices]
        )
        X_val[:, non_binary_indices] = train_scaler.transform(
            X_val[:, non_binary_indices]
        )
        if balance_data == "smote":
            if label_type == "soft":
                pass
            else:
                from imblearn.over_sampling import SMOTE

                smote = SMOTE(random_state=random_state)
                X_train, y_train = smote.fit_resample(X_train, y_train)
        else:
            raise NotImplementedError(f"Balancing method {balance_data} not found")

        model.fit(X_train, y_train)
        model = CalibratedClassifierCV(model, cv="prefit", method="sigmoid").fit(
            X_train, y_train.astype(int)
        )
        print("Class distribution in val set:\n", pd.Series(y_val).value_counts())
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)
        if y_pred_proba.shape[1] > 1:
            y_pred_proba = y_pred_proba[:, 1]

        if hasattr(model, "feature_importances_"):
            feature_importance = model.feature_importances_
            feature_importance = feature_importance.reshape(-1, 1)
        else:
            feature_importance = permutation_importance(
                model, X_val, y_val, n_repeats=10, n_jobs=-1
            )["importances_mean"]
            feature_importance = feature_importance.reshape(-1, 1)
        #! can use feature_importances_ attribute for quick check, permutation importance for longer check
        # feature_importance = None
        y_tests.append(y_val)
        y_preds.append(y_pred)
        y_pred_probas.append(y_pred_proba)
        y_ferritins.append(y_ferritin)
        feature_importances.append(feature_importance)

    return y_tests, y_preds, y_pred_probas, y_ferritins, feature_importances, model
