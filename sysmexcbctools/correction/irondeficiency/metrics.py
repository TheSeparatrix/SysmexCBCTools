import numpy as np
from sklearn.metrics import confusion_matrix


def specificity(y_true, y_pred):
    # Filter out NaNs
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    tn, fp, fn, tp = confusion_matrix(y_true_filtered, y_pred_filtered).ravel()
    return tn / (tn + fp)


def sensitivity(y_true, y_pred):
    # Filter out NaNs
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    tn, fp, fn, tp = confusion_matrix(y_true_filtered, y_pred_filtered).ravel()
    return tp / (tp + fn)


def positive_predictive_value(y_true, y_pred):
    # Filter out NaNs
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    tn, fp, fn, tp = confusion_matrix(y_true_filtered, y_pred_filtered).ravel()
    return tp / (tp + fp)


def negative_predictive_value(y_true, y_pred):
    # Filter out NaNs
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    tn, fp, fn, tp = confusion_matrix(y_true_filtered, y_pred_filtered).ravel()
    return tn / (tn + fn)


def picp(y_true, y_pred_lower, y_pred_upper):
    """
    Calculate Prediction Interval Coverage Probability (PICP)

    :param y_true: Array of true values.
    :param y_pred_lower: Array of lower bounds of prediction intervals.
    :param y_pred_upper: Array of upper bounds of prediction intervals.
    :return: PICP value.
    """
    n = len(y_true)
    coverage = np.mean((y_true >= y_pred_lower) & (y_true <= y_pred_upper))
    return coverage


def ace(y_true, y_pred_lower, y_pred_upper, alpha):
    """
    Calculate Average Coverage Error (ACE)

    :param y_true: Array of true values.
    :param y_pred_lower: Array of lower bounds of prediction intervals.
    :param y_pred_upper: Array of upper bounds of prediction intervals.
    :param alpha: Significance level (e.g., 0.05 for a 95% prediction interval).
    :return: ACE value.
    """
    n = len(y_true)
    coverage = calculate_picp(y_true, y_pred_lower, y_pred_upper)
    ace = np.abs(coverage - (1 - alpha))
    return ace


def pinaw(y_true, y_pred_lower, y_pred_upper):
    """
    Calculate Prediction Interval Normalized Average Width (PINAW)

    :param y_true: Array of true values.
    :param y_pred_lower: Array of lower bounds of prediction intervals.
    :param y_pred_upper: Array of upper bounds of prediction intervals.
    :return: PINAW value.
    """
    n = len(y_true)
    range_y = np.max(y_true) - np.min(y_true)
    if range_y == 0:
        raise ValueError(
            "The range of y_true is zero, which would result in division by zero in PINAW calculation."
        )

    average_width = np.mean(y_pred_upper - y_pred_lower)
    pinaw = average_width / range_y
    return pinaw


import numpy as np


def calculate_nri(y_true, y_pred_base, y_pred_new):
    # Verify inputs
    if len(y_true) != len(y_pred_base) or len(y_true) != len(y_pred_new):
        raise ValueError("All input arrays must have the same length")

    if (
        not np.all(np.isin(y_true, [0, 1]))
        or not np.all(np.isin(y_pred_base, [0, 1]))
        or not np.all(np.isin(y_pred_new, [0, 1]))
    ):
        raise ValueError("All arrays must contain only binary values (0 or 1)")

    # Create masks for events and non-events
    event_mask = y_true == 1
    nonevent_mask = y_true == 0

    # Count the number of events and non-events
    n_events = np.sum(event_mask)
    n_nonevents = np.sum(nonevent_mask)

    if n_events == 0 or n_nonevents == 0:
        raise ValueError(
            "Both events and non-events must be present in the true labels"
        )

    # For events (true label = 1)
    # Count cases where new model correctly reclassifies up (improvement)
    events_up = np.sum((y_pred_base == 0) & (y_pred_new == 1) & event_mask)
    # Count cases where new model incorrectly reclassifies down (worsening)
    events_down = np.sum((y_pred_base == 1) & (y_pred_new == 0) & event_mask)

    # For non-events (true label = 0)
    # Count cases where new model correctly reclassifies down (improvement)
    nonevents_down = np.sum((y_pred_base == 1) & (y_pred_new == 0) & nonevent_mask)
    # Count cases where new model incorrectly reclassifies up (worsening)
    nonevents_up = np.sum((y_pred_base == 0) & (y_pred_new == 1) & nonevent_mask)

    # Calculate component NRIs
    event_nri = (events_up - events_down) / n_events
    nonevent_nri = (nonevents_down - nonevents_up) / n_nonevents

    # Total NRI
    nri = event_nri + nonevent_nri

    # Prepare detailed statistics
    details = {
        "event_counts": {
            "total": int(n_events),
            "reclassified_up": int(events_up),
            "reclassified_down": int(events_down),
            "unchanged": int(n_events - events_up - events_down),
        },
        "nonevent_counts": {
            "total": int(n_nonevents),
            "reclassified_up": int(nonevents_up),
            "reclassified_down": int(nonevents_down),
            "unchanged": int(n_nonevents - nonevents_up - nonevents_down),
        },
        "event_proportions": {
            "reclassified_up": float(events_up / n_events),
            "reclassified_down": float(events_down / n_events),
            "unchanged": float((n_events - events_up - events_down) / n_events),
        },
        "nonevent_proportions": {
            "reclassified_up": float(nonevents_up / n_nonevents),
            "reclassified_down": float(nonevents_down / n_nonevents),
            "unchanged": float(
                (n_nonevents - nonevents_up - nonevents_down) / n_nonevents
            ),
        },
    }

    return nri, event_nri, nonevent_nri, details
