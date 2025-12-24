import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.colors import ListedColormap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from irondeficiency.metrics import sensitivity, specificity

SEABORN_PALETTE = "colorblind"
seaborn_colors = sns.color_palette(SEABORN_PALETTE)


def find_best_sensspec_threshold(y_true, y_pred_probas):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probas)
    square_distance = (1 - tpr) ** 2 + fpr**2
    best_threshold = thresholds[np.argmin(square_distance)]
    return best_threshold


def find_best_pr_threshold(y_true, y_pred_probas):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_probas)
    square_distance = (1 - precision) ** 2 + (1 - recall) ** 2
    best_threshold = thresholds[np.argmin(square_distance)]
    return best_threshold


def find_thresholds(y_trues_list, y_pred_probas_list, target_accuracy=0.95):
    """
    Find thresholds such that the accuracy for negative and positive classes is at least the target accuracy.

    Parameters:
    - y_trues_list: List of arrays of true binary labels from multiple model runs.
    - y_pred_probas_list: List of arrays of predicted probabilities from multiple model runs.
    - target_accuracy: The minimum desired accuracy for both the negative and positive classes (default = 0.95).

    Returns:
    - neg_threshold: Threshold where the negative class accuracy is at least 95%.
    - pos_threshold: Threshold where the positive class accuracy is at least 95%.
    - percent_classified_neg: Percentage of samples still classified as negative.
    - percent_classified_pos: Percentage of samples still classified as positive.
    """

    # Concatenate all training runs' y_trues and y_pred_probas into single arrays
    y_trues = np.concatenate(y_trues_list)
    y_pred_probas = np.concatenate(y_pred_probas_list)

    # Sort predicted probabilities and true labels based on increasing predicted probabilities
    sorted_indices = np.argsort(y_pred_probas)
    sorted_probs = y_pred_probas[sorted_indices]
    sorted_trues = y_trues[sorted_indices]

    # Calculate the confusion matrix at different thresholds for negative and positive
    neg_threshold = None
    pos_threshold = None
    percent_classified_neg = None
    percent_classified_pos = None

    # 1. Loop to find negative threshold, searching from 1 to 0
    neg_thresholds = np.linspace(1, 0, 1000)  # thresholds from 1 to 0
    for threshold in neg_thresholds:
        y_preds = (y_pred_probas >= threshold).astype(
            int
        )  # Predict based on the current threshold
        tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()

        neg_class_accuracy = tn / (tn + fn) if (tn + fn) > 0 else 0

        if neg_class_accuracy >= target_accuracy:
            neg_threshold = threshold
            percent_classified_neg = (tn + fn) / len(
                y_trues
            )  # Percentage still classified as negative
            break

    # 2. Loop to find positive threshold, searching from 0 to 1
    pos_thresholds = np.linspace(0, 1, 1000)  # thresholds from 0 to 1
    for threshold in pos_thresholds:
        y_preds = (y_pred_probas >= threshold).astype(
            int
        )  # Predict based on the current threshold
        tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()

        pos_class_accuracy = tp / (tp + fp) if (tp + fp) > 0 else 0

        if pos_class_accuracy >= target_accuracy:
            pos_threshold = threshold
            percent_classified_pos = (tp + fp) / len(
                y_trues
            )  # Percentage still classified as positive
            break

    return {
        "neg_threshold": neg_threshold,
        "pos_threshold": pos_threshold,
        "percent_classified_neg": (
            percent_classified_neg * 100 if percent_classified_neg is not None else 0.0
        ),
        "percent_classified_pos": (
            percent_classified_pos * 100 if percent_classified_pos is not None else 0.0
        ),
    }


def boxplot_with_counts(
    data_display, data_overall, x_continuous, y_categorical, hue_categorical
):
    # Create the boxplot
    plt.figure(figsize=(10, 6))

    # Get unique values for hue to calculate dynamic offsets
    hue_values = data_display[hue_categorical].dropna().unique()
    num_hues = len(hue_values)

    ax = sns.boxplot(
        data=data_display,
        x=x_continuous,
        y=y_categorical,
        hue=hue_categorical,
        hue_order=hue_values,
        showfliers=False,
        whis=[2.5, 97.5],
        palette=seaborn_colors,
    )

    # Group the data to calculate counts and medians
    plot_counts = (
        data_display.groupby([y_categorical, hue_categorical])
        .size()
        .reset_index(name="count")
    )
    plot_counts_overall = (
        data_overall.groupby([y_categorical, hue_categorical])
        .size()
        .reset_index(name="count")
    )
    medians = (
        data_display.groupby([y_categorical, hue_categorical])[x_continuous]
        .median()
        .reset_index()
    )
    high_percentiles = (
        data_display.groupby([y_categorical, hue_categorical])[x_continuous]
        .quantile(0.975)
        .reset_index()
    )

    # Loop through each combination of y and hue to add counts
    for i, (y_val, hue_val) in enumerate(
        plot_counts[[y_categorical, hue_categorical]].values
    ):
        # Find the median x position of the box
        median_val = medians[
            (medians[y_categorical] == y_val) & (medians[hue_categorical] == hue_val)
        ][x_continuous].values[0]
        percentile_val = high_percentiles[
            (medians[y_categorical] == y_val) & (medians[hue_categorical] == hue_val)
        ][x_continuous].values[0]

        # Find the corresponding count
        count_val = plot_counts[
            (plot_counts[y_categorical] == y_val)
            & (plot_counts[hue_categorical] == hue_val)
        ]["count"].values[0]
        count_val_overall = plot_counts_overall[
            (plot_counts_overall[y_categorical] == y_val)
            & (plot_counts_overall[hue_categorical] == hue_val)
        ]["count"].values[0]

        # Calculate dynamic offset based on the hue index
        hue_index = list(hue_values).index(hue_val)
        # default boxwidth is 0.8, so depending on how many hue variables we have the boxes get slimmer and add up to 0.8
        boxwidth = 0.8 / num_hues
        offset = -0.4 + 0.5 * boxwidth + hue_index * boxwidth

        # # Plot the count text near the median line
        # ax.text(
        #     median_val + 0.5,
        #     (i // num_hues) + offset,
        #     f"{count_val}/{count_val_overall}",
        #     va="center",
        #     color="black",
        # )
        #! revisit text in box again for paper

    ax.legend(loc="lower right")  # , bbox_to_anchor=(1.1, 1))

    return ax


def plot_confusion_matrix_from_predictions(
    y_true_list, y_pred_list, path: str, class_names=None
):
    """
    Plots a confusion matrix showing the mean ± standard deviation from multiple runs.

    Args:
    y_true_list (list of lists): A list of ground truth (true) labels for each run.
    y_pred_list (list of lists): A list of predicted labels for each run.
    class_names (list): A list of class names for the labels, e.g., ['Class 1', 'Class 2'].
                        Defaults to ['Negative', 'Positive'] for binary classification.

    Returns:
    None: Displays the confusion matrix plot.
    """
    # Default class names for binary classification if not provided
    if class_names is None:
        class_names = ["Negative", "Positive"]

    # Initialize a list to hold confusion matrices for each run
    confusion_matrices = []
    n_runs = len(y_true_list)

    # Compute the confusion matrix for each pair of y_true and y_pred
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        cm = confusion_matrix(y_true, y_pred)
        confusion_matrices.append(cm)

    # Convert the list of confusion matrices into a numpy array for easier computation
    confusion_matrices = np.array(confusion_matrices)

    # Calculate mean and standard deviation
    mean_matrix = np.mean(confusion_matrices, axis=0)
    std_matrix = np.std(confusion_matrices, axis=0)

    # Create the labels with mean ± std for each cell
    labels = np.array(
        [
            [f"{mean_matrix[i, j]:.2f} ± {std_matrix[i, j]:.2f}" for j in range(2)]
            for i in range(2)
        ]
    )

    # Set up the confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        mean_matrix,
        annot=labels,
        fmt="",
        cmap="Reds",
        cbar=False,
        square=True,
        annot_kws={"size": 14},
        linewidths=2,
        linecolor="white",
    )

    # Add labels for clarity
    plt.title(f"Confusion Matrix (Mean ± Std) over {n_runs} runs")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Adjust tick marks using the provided class names
    plt.xticks(ticks=[0.5, 1.5], labels=class_names, ha="center")
    plt.yticks(ticks=[0.5, 1.5], labels=class_names, va="center")

    plt.tight_layout()
    plt.savefig(path)


def plot_confusion_matrix_with_severity_single_run(
    y_true,
    y_pred,
    ferritin_true,
    ax,
    title=None,
    iron_status_colours: bool = True,
):
    """
    Plots a confusion matrix with true ferritin levels on the x-axis and predicted ferritin on y-axis.

    Args:
    y_true (array): Ground truth (true) binary labels.
    y_pred (array): Predicted binary labels.
    ferritin_true (array): Actual ferritin concentrations.
    ax: The matplotlib axis to plot on.
    title (str, optional): Title for the plot.
    iron_status_colours (bool): Whether to use iron status colors or severity colors.
    """

    total = len(y_true)
    # Initialize a 2x3 confusion matrix for different ferritin ranges (transposed from original)
    cm = np.zeros((2, 3))

    # Loop over true and predicted labels to assign them to the appropriate category
    for true_label, pred_label, ferritin in zip(y_true, y_pred, ferritin_true):
        if pred_label == 1:  # Model predicts ferritin < 15
            if ferritin < 15:  # True ferritin < 15
                cm[0, 0] += 1
            elif 15 <= ferritin <= 30:  # True ferritin between 15 and 30
                cm[0, 1] += 1
            elif ferritin > 30:  # True ferritin > 30
                cm[0, 2] += 1
        else:  # Model predicts ferritin >= 15
            if ferritin < 15:  # True ferritin < 15
                cm[1, 0] += 1
            elif 15 <= ferritin <= 30:  # True ferritin between 15 and 30
                cm[1, 1] += 1
            elif ferritin > 30:  # True ferritin > 30
                cm[1, 2] += 1

    # Create the labels for the confusion matrix
    labels = np.array(
        [
            [
                f"{int(cm[i, j])}\n({int(100*cm[i, j]/total)}%)"
                for j in range(cm.shape[1])
            ]
            for i in range(cm.shape[0])
        ]
    )

    # Define colors based on selected scheme
    no = "#88E0A3"  # green --> no severity
    low = "#B2DF8A"  # light green --> low severity
    medium = "#FFCC80"  # orange --> medium severity
    high = "#E57373"  # red --> high severity

    if not iron_status_colours:
        # Manually assign colours based on severity for each cell
        colors = np.array(
            [
                [no, low, medium],
                [high, no, no],
            ]
        )
    else:
        colors = np.array(
            [
                ["salmon", "bisque", "lightblue"],
                ["salmon", "bisque", "lightblue"],
            ]
        )

    # Plot the heatmap
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Greys",
        cbar=False,
        linecolor="white",
        xticklabels=[r"$<15$", "15-30", r"$>30$"],
        yticklabels=[r"$<15$", r"$\geq15$"],
        mask=np.zeros_like(cm),
        ax=ax,
    )

    # Overlay custom colours manually for each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.add_patch(
                plt.Rectangle(
                    (j, i),
                    1,
                    1,
                    fill=True,
                    color=colors[i, j],
                    edgecolor="white",
                    lw=2,
                )
            )
            ax.text(
                j + 0.5,
                i + 0.5,
                labels[i, j],
                ha="center",
                va="center",
                color="black",
            )

    # Add dividing lines
    for i in range(1, cm.shape[0]):
        ax.axhline(i, color="black", lw=2, linestyle="--")
    for j in range(1, cm.shape[1]):
        ax.axvline(j, color="black", lw=2, linestyle="--")

    ax.set_title(title)
    ax.set_xlabel("True ferritin (µg/L)")
    ax.set_ylabel("Predicted ferritin (µg/L)")


def plot_confusion_matrix_with_severity_heatmap(
    y_true,
    y_pred,
    ferritin_true,
    ax,
    title=None,
):
    total = len(y_true)
    # Initialize a 3x2 confusion matrix for different ferritin ranges
    cm = np.zeros((3, 2))

    # Loop over true and predicted labels to assign them to the appropriate severity category
    for true_label, pred_label, ferritin in zip(y_true, y_pred, ferritin_true):
        if pred_label == 1:  # Model predicts ferritin < 15
            if ferritin < 15:  # True ferritin < 15 (No severity)
                cm[0, 0] += 1
            elif 15 <= ferritin <= 30:  # True ferritin between 15 and 30 (Low severity)
                cm[1, 0] += 1
            elif ferritin > 30:  # True ferritin > 30 (Medium severity)
                cm[2, 0] += 1
        else:  # Model predicts ferritin >= 15
            if ferritin < 15:  # True ferritin < 15 (High severity)
                cm[0, 1] += 1
            elif 15 <= ferritin <= 30:  # True ferritin between 15 and 30 (No severity)
                cm[1, 1] += 1
            elif ferritin > 30:  # True ferritin > 30 (No severity)
                cm[2, 1] += 1

    labels = np.array(
        [
            [
                f"{int(cm[i, j])}\n({int(100*cm[i, j]/total)}%)"  # \n{label_severity_array[i,j]}"
                for j in range(cm.shape[1])
            ]
            for i in range(cm.shape[0])
        ]
    )

    # Plot an empty heatmap (for formatting)
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="viridis",
        cbar=False,
        # square=True,
        xticklabels=[r"$<15$", r"$\geq15$"],
        yticklabels=[r"$<15$", "15-30", r"$>30$"],
        ax=ax,
    )
    ax.set_ylabel("True ferritin (µg/L)")
    ax.set_xlabel("Predicted ferritin (µg/L)")
    if title is not None:
        ax.set_title(title)


def plot_roc_curve_multipleruns(
    y_trues, y_pred_probas, ax_roc, label: str = "", i: int = 0
):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for j in range(len(y_trues)):
        fpr, tpr, _ = roc_curve(y_trues[j], y_pred_probas[j])
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax_roc.plot(
        mean_fpr,
        mean_tpr,
        color=seaborn_colors[i],
        label=f"{label} (AUC = {mean_auc:.2f} +/- {std_auc:.2f})",
        lw=2,
        alpha=1.0,
    )
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax_roc.fill_between(
        mean_fpr, tprs_lower, tprs_upper, color=seaborn_colors[i], alpha=0.2
    )


def plot_precision_recall_curve_multipleruns(
    y_trues, y_pred_probas, ax_pr, label: str = "", i: int = 0
):
    precisions = []
    aps = []
    mean_recall = np.linspace(0, 1, 100)

    for j in range(len(y_trues)):
        precision, recall, _ = precision_recall_curve(y_trues[j], y_pred_probas[j])

        # Ensure recall and precision are sorted in ascending recall order
        recall, precision = zip(*sorted(zip(recall, precision)))

        # Interpolate precision for the mean recall points
        interp_precision = np.interp(mean_recall, recall, precision)
        interp_precision[0] = (
            1.0  # Set the first precision value to 1.0 for the PR curve
        )

        precisions.append(interp_precision)
        aps.append(
            auc(recall, precision)
        )  # Use the original recall/precision to compute AP

    # Calculate the mean precision and average precision score
    mean_precision = np.mean(precisions, axis=0)
    mean_precision[-1] = 0.0  # Ensure the last precision value is 0.0

    mean_ap = auc(mean_recall, mean_precision)
    std_ap = np.std(aps)

    # Plot the mean precision-recall curve
    ax_pr.plot(
        mean_recall,
        mean_precision,
        color=seaborn_colors[i],
        label=f"{label} (AP = {mean_ap:.2f} ± {std_ap:.2f})",
        lw=2,
        alpha=1.0,
    )

    # Plot the standard deviation as a shaded region
    std_precision = np.std(precisions, axis=0)
    precisions_upper = np.minimum(mean_precision + std_precision, 1)
    precisions_lower = np.maximum(mean_precision - std_precision, 0)

    ax_pr.fill_between(
        mean_recall,
        precisions_lower,
        precisions_upper,
        color=seaborn_colors[i],
        alpha=0.2,
    )


def plot_calibration_curve_multipleruns(
    y_trues,
    y_pred_probas,
    ax_calibration,
    label: str = "",
    i: int = 0,
    n_bins: int = 10,
):
    prob_true_list = []
    prob_pred_list = []

    for j in range(len(y_trues)):
        prob_true, prob_pred = calibration_curve(
            y_trues[j], y_pred_probas[j], n_bins=n_bins, strategy="uniform"
        )
        prob_true_list.append(prob_true)
        prob_pred_list.append(prob_pred)

    # Interpolating over a common range for averaged calibration curve
    mean_prob_pred = np.linspace(0, 1, 100)
    prob_true_interp_list = []

    for prob_true, prob_pred in zip(prob_true_list, prob_pred_list):
        interp_prob_true = np.interp(mean_prob_pred, prob_pred, prob_true)
        prob_true_interp_list.append(interp_prob_true)

    mean_prob_true = np.mean(prob_true_interp_list, axis=0)
    std_prob_true = np.std(prob_true_interp_list, axis=0)

    # Plot mean calibration curve
    ax_calibration.plot(
        mean_prob_pred,
        mean_prob_true,
        color=seaborn_colors[i],
        label=f"{label}",
        lw=2,
        alpha=1.0,
    )

    # Plot uncertainty region (std deviation of calibration curves)
    prob_true_upper = np.minimum(mean_prob_true + std_prob_true, 1)
    prob_true_lower = np.maximum(mean_prob_true - std_prob_true, 0)
    ax_calibration.fill_between(
        mean_prob_pred,
        prob_true_lower,
        prob_true_upper,
        color=seaborn_colors[i],
        alpha=0.2,
    )


def evaluation_pipeline(
    results: List[dict],
    ferritin_threshold: int = 15,
    out_dir: str = None,
    type: str = "classification",
):
    if type == "classification":
        return evaluation_classification(
            results,
            ferritin_threshold,
            out_dir,
        )
    elif type == "regression":
        return evaluation_regression()

    else:
        raise ValueError("Invalid type argument")


def evaluation_classification(
    results: List[dict],
    ferritin_threshold: int = 15,
    out_dir: str = None,
):
    # get all unique feature names
    feature_names = list(results.keys())
    # get all unique model names
    model_names = list(results[feature_names[0]].keys())

    for model_name in model_names:
        fig_roc, ax_roc = plt.subplots()
        fig_pr, ax_pr = plt.subplots()
        if out_dir is not None:
            text_path = out_dir + "/text_outs/"
            fig_path = out_dir + "/figures/"
            text_file_path = text_path + f"classification_evaluation_{model_name}.txt"
            # create text file (overwrite if it already exists)
            with open(text_file_path, "w") as f:
                f.write(f"Evaluation of {model_name}\n")
        print(f"Evaluating {model_name}")

        ferritin_distances = []
        for feature_name in feature_names:
            y_pred = results[feature_name][model_name]["y_pred"]
            y_pred_proba = results[feature_name][model_name]["y_pred_proba"]
            y_ferritin = results[feature_name][model_name]["y_ferritin"]
            y_test = results[feature_name][model_name]["y_test"]
            feature_importances = results[feature_name][model_name][
                "feature_importances"
            ]
            feature_list = results[feature_name][model_name]["feature_list"]
            print(f"Feature: {feature_name}")
            print("Classification report:")
            print(classification_report(y_test, y_pred))
            print("Confusion matrix:")
            print(confusion_matrix(y_test, y_pred))
            print("ROC AUC score:")
            print(roc_auc_score(y_test, y_pred_proba))
            print("Average precision score:")
            print(average_precision_score(y_test, y_pred_proba))
            if feature_importances is not None:
                # print top 10 features
                print("Top 10 features:")
                feature_importances_df = pd.DataFrame(
                    {"Feature importance": feature_importances, "Feature": feature_list}
                )
                feature_importances_df = feature_importances_df.sort_values(
                    "Feature importance", ascending=False
                )
                feature_importances_df.reset_index(drop=True, inplace=True)
                print(feature_importances_df.head(10))

            if out_dir is not None:
                # Create the text_outs directory if it doesn't exist
                os.makedirs(text_path, exist_ok=True)
                # Append to text file
                with open(text_file_path, "a") as f:
                    f.write(f"Feature list: {feature_name}\n")
                    f.write("Classification report:\n")
                    f.write(str(classification_report(y_test, y_pred)))
                    f.write("\n")
                    f.write("Confusion matrix:\n")
                    f.write(str(confusion_matrix(y_test, y_pred)))
                    f.write("\n")
                    f.write("ROC AUC score:\n")
                    f.write(str(roc_auc_score(y_test, y_pred_proba)))
                    f.write("\n")
                    f.write("Average precision score:\n")
                    f.write(str(average_precision_score(y_test, y_pred_proba)))
                    f.write("\n")
                    if feature_importances is not None:
                        f.write("Top 10 features:\n")
                        f.write(str(feature_importances_df.head(10)))
                        f.write("\n")

            roc_display = RocCurveDisplay.from_predictions(
                y_test, y_pred_proba, name=feature_name
            )
            roc_display.plot(ax=ax_roc)

            pr_display = PrecisionRecallDisplay.from_predictions(
                y_test, y_pred_proba, name=feature_name
            )
            pr_display.plot(ax=ax_pr)

            # calculate distance from ferritin threshold for wrong predictions
            wrong_indices = y_test != y_pred
            wrong_ferritin = y_ferritin[wrong_indices]
            wrong_pred_proba = y_pred_proba[wrong_indices]
            wrong_pred = y_pred[wrong_indices]
            wrong_distance = abs(wrong_ferritin - ferritin_threshold)
            ferritin_distances.append(
                {
                    "Features used": feature_name,
                    "distance": wrong_distance,
                    "probas": wrong_pred_proba,
                    "preds": wrong_pred,
                }
            )

        ax_roc.set_title(f"ROC curve")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend()
        fig_roc.tight_layout()
        if out_dir is not None:
            fig_roc.savefig(fig_path + f"roc_curve_{model_name}.pdf")

        ax_pr.set_title(f"Precision-Recall curve")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.legend()
        fig_pr.tight_layout()
        if out_dir is not None:
            fig_pr.savefig(fig_path + f"pr_curve_{model_name}.pdf")

        # plot ferritin distances
        # make dataframe of distances, feature names, probabilities, and predictions
        distances_df = pd.DataFrame()
        for distance_dict in ferritin_distances:
            distances_df = pd.concat(
                [
                    distances_df,
                    pd.DataFrame(distance_dict),
                ],
            )
        distances_df.reset_index(drop=True, inplace=True)
        proba_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        proba_bin_labels = [f"{i}-{j}" for i, j in zip(proba_bins[:-1], proba_bins[1:])]
        distances_df["class_1_confidence_bin"] = pd.cut(
            distances_df["probas"], proba_bins, labels=proba_bin_labels
        )
        plt.clf()
        sns.boxplot(
            x="class_1_confidence_bin",
            y="distance",
            hue="Features used",
            data=distances_df,
            showfliers=False,
        )
        plt.xlabel("Class 1 Confidence")
        plt.ylabel("Distance from Ferritin Threshold")
        plt.title("Distance from Ferritin Threshold for Wrong Predictions")
        plt.tight_layout()
        if out_dir is not None:
            plt.savefig(fig_path + f"ferritin_distances_{model_name}.pdf")


def evaluation_regression():
    pass


def plot_roc_curve(y_true, y_pred_proba, label=None):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")


def plot_precision_recall_curve(y_true, y_pred_proba, label=None):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)

    plt.plot(recall, precision, label=label)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")


def get_metrics(y_true, y_pred, y_score):
    """
    Calculate metrics.
    """
    f1 = f1_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    sens = sensitivity(y_true, y_pred)
    spec = specificity(y_true, y_pred)
    # Other metrics can be added here

    return f1, bal_acc, auc, pr_auc, sens, spec


def plot_waterfall_explanation(single_explanation, train_scaler, non_binary_indices):
    single_explanation.data[non_binary_indices] = (
        train_scaler.inverse_transform(
            single_explanation.data.reshape(1, -1)[:, non_binary_indices]
        )
        .reshape(-1)
        .flatten()
    )
    shap.plots.waterfall(
        single_explanation,
        show=False,
        max_display=10,
    )


def custom_waterfall_plot(shap_values, threshold=0.5, max_display=10):
    """
    Create a customized SHAP waterfall plot with modified labels and threshold line for ferritin 15 experiment
    """

    # Create the base waterfall plot with show=False to allow modifications
    ax = shap.plots.waterfall(shap_values, max_display=max_display, show=False)
    fig = plt.gcf()

    # Get the base value and predicted value
    base_values = float(shap_values.base_values)
    fx = base_values + shap_values.values.sum()

    # Clear existing axes that have the E[f(X)] and f(x) labels
    for ax_obj in fig.axes[1:]:
        ax_obj.remove()

    ax = fig.axes[0]

    # Add threshold line
    ax.axvline(
        x=threshold,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Operating\nthreshold ({threshold:.2f})",
    )
    # ax.legend()

    # Add x-label at bottom
    ax.set_xlabel("Estimated prob. of ferritin < 15 µg/L")  # , fontsize=12)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Style cleanup
    for axis in [ax]:
        axis.spines["right"].set_visible(False)
        axis.spines["top"].set_visible(False)
        axis.spines["left"].set_visible(False)

    return fig
