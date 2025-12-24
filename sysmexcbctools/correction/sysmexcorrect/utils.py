"""
Utility functions for GAM-based correction.
"""

import contextlib

import joblib
import numpy as np
import pandas as pd


def mad(x):
    """
    Calculate Median Absolute Deviation.

    Parameters
    ----------
    x : pd.Series or np.ndarray
        Input data

    Returns
    -------
    float
        Median absolute deviation

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 3, 4, 5, 100])
    >>> mad(data)
    1.5
    """
    if isinstance(x, pd.Series):
        return (x - x.median()).abs().median()
    else:
        x = np.asarray(x)
        median = np.median(x)
        return np.median(np.abs(x - median))


def centralise(df, x, threshold=3.5):
    """
    Filter data to keep only values within threshold MADs from the median.

    This function removes outliers by keeping only data points that differ
    from the median by less than threshold times the median absolute deviation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to filter
    x : pd.Series
        Column to use for filtering (must have same index as df)
    threshold : float, default=3.5
        Number of median absolute deviations from median

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1, 2, 3, 4, 100], 'b': [10, 20, 30, 40, 50]})
    >>> x = df['a']
    >>> centralise(df, x, threshold=2.0)
         a   b
    0    1  10
    1    2  20
    2    3  30
    3    4  40
    """
    low = x.median() - threshold * mad(x)
    high = x.median() + threshold * mad(x)
    return df[x.between(low, high, inclusive="neither")].copy()


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar.

    Parameters
    ----------
    tqdm_object : tqdm
        Progress bar object

    Yields
    ------
    tqdm
        The progress bar object

    Examples
    --------
    >>> from joblib import Parallel, delayed
    >>> from tqdm import tqdm
    >>> with tqdm_joblib(tqdm(desc="Processing", total=10)) as progress_bar:
    ...     results = Parallel(n_jobs=2)(delayed(lambda x: x**2)(i) for i in range(10))
    """
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
