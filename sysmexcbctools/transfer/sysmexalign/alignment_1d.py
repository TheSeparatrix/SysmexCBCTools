import numpy as np
from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from .gmm_ot import compute_gmm_transport_map, transform_points_gmm


def transform_nonnormal(
    X: np.ndarray,
    median_source: float,
    median_target: float,
    mad_source: float,
    mad_target: float,
) -> np.ndarray:
    if mad_source == 0:
        mad_source = 1e-10
    if mad_target == 0:
        mad_target = 1e-10
    return (X - median_source) / mad_source * mad_target + median_target


def mad_score(x: np.ndarray) -> np.ndarray:
    median = np.median(x)
    mad_value = mad(x)
    return (x - median) / mad_value


def mad(x: np.ndarray) -> float:
    return np.median(np.abs(x - np.median(x)))


def sample_impedance_array(args, impedance_data, random_state=None):
    """
    Sample from histogram bins to create point representation for GMM fitting.

    Uses probabilistic rounding to preserve distribution shape when downsampling.

    Problem with naive int() rounding:
    - Bins with low counts get rounded to 0 (completely lost)
    - Example: count=7000, factor=0.00012 → 0.84 → int() → 0 samples ❌

    Improved approach with probabilistic rounding:
    - 0.84 → 84% chance of 1 sample, 16% chance of 0 samples ✓
    - Preserves expected value: E[samples] = 0.84
    - Better represents true distribution shape

    Parameters
    ----------
    args : object with gmm_sample_size attribute
        Configuration object specifying target sample size.
    impedance_data : pd.Series
        Histogram bin counts (e.g., RBC_RAW_000 to RBC_RAW_127).
    random_state : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    sample_array : np.ndarray
        Array of bin indices, with each index repeated according to bin count.
        Shape: (n_samples, 1) where n_samples ≈ gmm_sample_size (if downsampling).
    """
    total_counts = impedance_data.sum()

    if total_counts > args.gmm_sample_size:
        factor = args.gmm_sample_size / total_counts
    else:
        factor = 1

    # Compute scaled counts (may be fractional)
    scaled_counts = factor * impedance_data.values

    # Split into integer and fractional parts
    integer_parts = np.floor(scaled_counts).astype(int)
    fractional_parts = scaled_counts - integer_parts

    # Probabilistic rounding: add 1 with probability = fractional part
    # This preserves expected value and distribution shape much better than int()
    rng = np.random.default_rng(seed=random_state)
    random_additions = (rng.random(len(fractional_parts)) < fractional_parts).astype(int)
    final_counts = integer_parts + random_additions

    # Create point samples by repeating bin indices
    sample_array = np.repeat(
        np.arange(len(impedance_data)).reshape(-1, 1),
        final_counts,
        axis=0,
    )

    return sample_array


def fit_gmm_rbc(rbc_impedance_samples):
    gmm = GaussianMixture(
        n_components=3,
        covariance_type="full",
        n_init=5,
        means_init=[
            [1],
            [35],
            [80],
        ],
        # verbose=1,
    ).fit(rbc_impedance_samples)
    return gmm


def fit_gmm_plt(log_plt_impedance_samples):
    # assuming PLT samples have been log-transformed (makes it more Gaussian)
    gmm = GaussianMixture(
        n_components=3,
        covariance_type="full",
        n_init=5,
        means_init=[
            [4.5],
            [3.0],
            [2.5],
        ],
        # verbose=1,
    ).fit(log_plt_impedance_samples)
    return gmm


def transform_impedance_data(
    source_df, source_standards, target_standards, col_prefix, fit_gmm_func, args, random_state=None
):
    def process_row(row, cols, args, gmm_source, transport_dict):
        rbc_sample = sample_impedance_array(args, row[cols], random_state=random_state)
        if len(rbc_sample) == 0:
            return np.zeros(128)
        if col_prefix == "PLT":
            rbc_sample = np.log(rbc_sample + 1)
        rbc_sample_transformed = transform_points_gmm(
            rbc_sample, gmm_source, transport_dict, transport_method="rand", random_state=random_state
        )
        if col_prefix == "PLT":
            rbc_sample_transformed = np.exp(rbc_sample_transformed) - 1
        transformed_hist, _ = np.histogram(
            rbc_sample_transformed,
            bins=np.arange(-0.5, 128.5),  # 128 bins with bin centers at integers 0-127
        )
        return transformed_hist

    source_data = source_standards.filter(like=f"{col_prefix}_RAW_").sum(axis=0)
    target_data = target_standards.filter(like=f"{col_prefix}_RAW_").sum(axis=0)

    X_source = sample_impedance_array(args, source_data, random_state=random_state)
    X_target = sample_impedance_array(args, target_data, random_state=random_state)

    if col_prefix == "PLT":
        X_source = np.log(X_source + 1)
        X_target = np.log(X_target + 1)

    gmm_source = fit_gmm_func(X_source)
    gmm_target = fit_gmm_func(X_target)

    transport_dict = compute_gmm_transport_map(gmm_source, gmm_target)

    cols = source_df.filter(like=f"{col_prefix}_RAW_").columns

    print(f"Processing {len(source_df)} rows with parallel processing...")
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_row)(source_df.iloc[i], cols, args, gmm_source, transport_dict)
        for i in tqdm(range(len(source_df)))
    )

    source_df.loc[:, cols] = np.array(results)
    return source_df
