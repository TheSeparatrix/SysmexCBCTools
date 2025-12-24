"""
Training utilities for Dis-AE 2

Includes early stopping and other training helpers.
"""


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.

    Parameters
    ----------
    patience : int, default=16
        Number of epochs to wait before stopping if no improvement
    min_delta : float, default=0.0
        Minimum change in monitored value to qualify as improvement
    mode : str, default='min'
        One of 'min' or 'max'. In 'min' mode, training stops when metric stops decreasing;
        in 'max' mode, training stops when metric stops increasing.

    Attributes
    ----------
    best_score : float or None
        Best score observed so far
    counter : int
        Number of epochs since last improvement
    early_stop : bool
        Whether early stopping criterion has been met
    """

    def __init__(self, patience: int = 16, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_metric: float):
        """
        Check if early stopping criterion is met.

        Parameters
        ----------
        val_metric : float
            Current validation metric value
        """
        if self.mode == 'min':
            score = -val_metric
        else:
            score = val_metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def reset(self):
        """Reset early stopping state."""
        self.best_score = None
        self.counter = 0
        self.early_stop = False
