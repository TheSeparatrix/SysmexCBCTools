from typing import Dict, List, Optional, Tuple

import numpy as np
import ot
import scipy
import scipy.linalg
from sklearn.mixture import GaussianMixture


def validate_transformation(
    original_data, target_data, transformed_data, gmm_target, output_path=None
):
    """
    Validate transformation results by comparing distributions and plotting with target PDF
    """
    import matplotlib.pyplot as plt
    from scipy.stats import wasserstein_distance

    # Calculate likelihoods under target model
    source_ll = gmm_target.score_samples(original_data)
    target_ll = gmm_target.score_samples(target_data)
    transformed_ll = gmm_target.score_samples(transformed_data)

    # Determine the dimensionality
    n_dims = original_data.shape[1]

    # Create a figure with subplots
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 5 * n_dims))
    if n_dims == 1:
        axes = [axes]

    w_distances = []

    for i in range(n_dims):
        ax = axes[i]

        # Calculate 1D Wasserstein distance
        w_dist = wasserstein_distance(original_data[:, i], transformed_data[:, i])
        w_distances.append(w_dist)

        # Determine the plot range
        all_data = np.concatenate([original_data[:, i], transformed_data[:, i]])
        min_val, max_val = np.min(all_data), np.max(all_data)
        range_val = max_val - min_val
        min_val -= range_val * 0.1
        max_val += range_val * 0.1

        # Plot histograms with density=True to make them comparable to PDF
        ax.hist(
            original_data[:, i],
            bins=100,
            alpha=0.3,
            label="Original",
            density=True,
            range=(min_val, max_val),
            color="blue",
        )
        ax.hist(
            target_data[:, i],
            bins=100,
            alpha=0.3,
            label="Target",
            density=True,
            range=(min_val, max_val),
            color="orange",
        )
        ax.hist(
            transformed_data[:, i],
            bins=100,
            alpha=0.3,
            label="Transformed",
            density=True,
            range=(min_val, max_val),
            color="green",
        )

        # Generate points for plotting the target GMM PDF
        x_range = np.linspace(min_val, max_val, 1000)

        # Compute the marginal PDF for this dimension
        pdf_values = np.zeros_like(x_range)
        for j in range(gmm_target.n_components):
            # Extract mean and std dev for this dimension
            mean = gmm_target.means_[j, i]
            var = (
                gmm_target.covariances_[j, i, i]
                if gmm_target.covariance_type == "full"
                else gmm_target.covariances_[j][i]
            )
            std = np.sqrt(var)

            # Add weighted Gaussian PDF
            from scipy.stats import norm

            pdf_values += gmm_target.weights_[j] * norm.pdf(x_range, mean, std)

        # Plot the target GMM PDF
        ax.plot(x_range, pdf_values, "r-", linewidth=2, label="Target GMM")

        # Add detailed annotations
        ax.set_title(f"Dimension {i} - Wasserstein distance: {w_dist:.4f}")
        ax.text(
            0.05,
            0.95,
            f"Mean log-likelihood under target:\n"
            f"  Original: {np.mean(source_ll):.2f}\n"
            f"  Target: {np.mean(target_ll):.2f}\n"
            f"  Transformed: {np.mean(transformed_ll):.2f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(facecolor="white", alpha=0.8),
        )

        ax.set_xlabel(f"Feature {i} value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Saved validation plot to {output_path}")
    else:
        plt.show()

    return {
        "wasserstein_distances": w_distances,
        "mean_source_likelihood": np.mean(source_ll),
        "mean_target_likelihood": np.mean(target_ll),
        "mean_transformed_likelihood": np.mean(transformed_ll),
        "likelihood_improvement": np.mean(transformed_ll) - np.mean(source_ll),
    }


def compute_gmm_transport_map(
    gmm_source: GaussianMixture, gmm_target: GaussianMixture
) -> Dict[str, np.ndarray]:
    n_components_source = gmm_source.n_components
    n_components_target = gmm_target.n_components

    # Compute Wasserstein matrix and resulting weighting (transport plan on the component level)
    wasserstein_matrix = np.zeros((n_components_source, n_components_target))
    for i in range(n_components_source):
        for j in range(n_components_target):
            wasserstein_matrix[i, j] = compute_wasserstein(
                gmm_source.means_[i],
                gmm_target.means_[j],
                gmm_source.covariances_[i],
                gmm_target.covariances_[j],
            )

    omega = ot.emd(gmm_source.weights_, gmm_target.weights_, wasserstein_matrix)

    # Compute transport maps between all Gaussian components (transport plan on the point level)
    transport_maps = np.zeros(
        (
            n_components_source,
            n_components_target,
            gmm_source.means_.shape[1],
            gmm_target.means_.shape[1],
        )
    )

    for i in range(n_components_source):
        for j in range(n_components_target):
            if omega[i, j] > 0:  # Only compute if there's transport between components
                transport_maps[i, j], _ = compute_gaussian_T_A(
                    gmm_source.covariances_[i], gmm_target.covariances_[j]
                )

    return {
        "transport_maps": transport_maps,
        "means_source": gmm_source.means_,
        "means_target": gmm_target.means_,
        "omega": omega,
    }


def transform_points_gmm(
    X: np.ndarray,
    gmm_source: GaussianMixture,
    transport_map_dict: Dict[str, np.ndarray],
    batch_size: Optional[int] = None,
    transport_method: str = "weight",
    threshold: float = 0.05,
    preserve_rare: bool = True,
    rare_probability_threshold: float = 0.01,
    random_state: Optional[int] = None,
) -> np.ndarray:
    # Extract transport map components
    transport_maps = transport_map_dict["transport_maps"]
    means_source = transport_map_dict["means_source"]
    means_target = transport_map_dict["means_target"]
    omega = transport_map_dict["omega"]

    # Create RNG for reproducibility
    rng = np.random.default_rng(seed=random_state)

    def process_batch(X_batch: np.ndarray) -> np.ndarray:
        # Calculate probabilities for source components
        log_probs = gmm_source._estimate_log_prob(X_batch)

        # Validate that _estimate_log_prob returned the expected format
        if not isinstance(log_probs, np.ndarray):
            raise TypeError(
                f"GMM._estimate_log_prob() should return a numpy array, "
                f"but got {type(log_probs)}. This may indicate a sklearn version "
                f"incompatibility."
            )

        if log_probs.ndim != 2:
            raise ValueError(
                f"GMM._estimate_log_prob() should return a 2D array of shape "
                f"(n_samples, n_components), but got shape {log_probs.shape} "
                f"with {log_probs.ndim} dimensions."
            )

        if log_probs.shape[0] != X_batch.shape[0]:
            raise ValueError(
                f"GMM._estimate_log_prob() returned array with {log_probs.shape[0]} rows, "
                f"but expected {X_batch.shape[0]} rows to match input batch size."
            )

        log_probs = np.array(log_probs, dtype=np.float64)
        source_probs = np.exp(log_probs)
        source_prob_overall = np.exp(gmm_source.score_samples(X_batch))

        if transport_method == "weight":
            # Identify the most likely source component for each point
            source_components = np.argmax(source_probs, axis=1)

            # Prepare result array
            transformed_points = np.zeros_like(X_batch)

            # For each point, apply T_weight transformation
            for i in range(len(X_batch)):
                k1 = source_components[i]

                # Find target components with omega above threshold
                valid_targets = np.where(omega[k1, :] >= threshold)[0]

                if len(valid_targets) == 0:
                    # If no valid targets, use the most likely target component
                    valid_targets = [np.argmax(omega[k1, :])]

                # Identify if this is potentially a rare population
                is_rare = np.max(source_probs[i]) < rare_probability_threshold

                if preserve_rare and is_rare:
                    # For rare populations, use probabilistic selection with bias toward preservation
                    # Normalize weights for valid targets
                    weights = omega[k1, valid_targets]
                    weights = weights / np.sum(weights)

                    # Select a target component based on weights
                    k2 = valid_targets[rng.choice(len(valid_targets), p=weights)]

                    # Apply transformation
                    transformed_points[i] = apply_transport_map_Gaussians(
                        X_batch[i : i + 1],
                        mean_source=means_source[k1],
                        mean_target=means_target[k2],
                        transport_matrix=transport_maps[k1, k2],
                    )
                else:
                    # For major populations, use weighted average of transformations
                    weights = omega[k1, valid_targets]
                    weights = weights / np.sum(weights)

                    # Initialize weighted sum as float64 to avoid casting errors
                    weighted_sum = np.zeros_like(X_batch[i], dtype=np.float64)

                    # Apply all valid transformations and weight them
                    for idx, k2 in enumerate(valid_targets):
                        transformed = apply_transport_map_Gaussians(
                            X_batch[i : i + 1],
                            mean_source=means_source[k1],
                            mean_target=means_target[k2],
                            transport_matrix=transport_maps[k1, k2],
                        )
                        weighted_sum += weights[idx] * transformed[0].astype(
                            np.float64
                        )  # [0] to unpack the single-element array

                    transformed_points[i] = weighted_sum

            return ensure_valid_fc_data(transformed_points)
        elif transport_method == "rand":
            # Reuse the already-computed and validated probabilities from above
            p_s = source_probs
            p_s_overall = source_prob_overall

            # Calculate component transition probabilities
            p_k1_k2 = np.zeros(
                (len(X_batch), gmm_source.n_components, transport_maps.shape[1])
            )
            for k1 in range(gmm_source.n_components):
                for k2 in range(transport_maps.shape[1]):
                    if omega[k1, k2] > 0:
                        p_k1_k2[:, k1, k2] = omega[k1, k2] * p_s[:, k1] / p_s_overall

            # Initialize component assignment dictionaries
            X_k1k2: Dict[Tuple[int, int], List[np.ndarray]] = {
                (i, j): []
                for i in range(gmm_source.n_components)
                for j in range(transport_maps.shape[1])
            }

            # Assign points to component pairs
            for i in range(len(X_batch)):
                k1, k2 = choose_index_from_prob_matrix(p_k1_k2[i], rng=rng)
                X_k1k2[(k1, k2)].append(X_batch[i])

            # Transform points for each component pair
            transformed_points = []
            for k1 in range(gmm_source.n_components):
                for k2 in range(transport_maps.shape[1]):
                    if len(X_k1k2[(k1, k2)]) > 0:
                        X_comp = np.array(X_k1k2[(k1, k2)])
                        transformed_points.append(
                            apply_transport_map_Gaussians(
                                X_comp,
                                mean_source=means_source[k1],
                                mean_target=means_target[k2],
                                transport_matrix=transport_maps[k1, k2],
                            )
                        )

            return (
                ensure_valid_fc_data(np.concatenate(transformed_points))
                if transformed_points
                else np.array([])
            )
        elif (
            transport_method == "max"
        ):  # like "weight" but only using the most likely target component
            # get source GMM data components
            k_s = gmm_source.predict(X_batch)
            k_s_t = omega.argmax(axis=1)
            X_trafo = []
            for k in np.unique(k_s):
                X = X_batch[k_s == k]
                X_trafo.append(
                    apply_transport_map_Gaussians(
                        X,
                        mean_source=means_source[k],
                        mean_target=means_target[k_s_t[k]],
                        transport_matrix=transport_maps[k, k_s_t[k]],
                    )
                )

            return (
                ensure_valid_fc_data(np.concatenate(X_trafo))
                if X_trafo
                else np.array([])
            )

        else:
            raise ValueError("Invalid transport method specified")

    # Process data in batches if specified
    if batch_size is not None:
        transformed_batches = []
        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size]
            transformed_batches.append(process_batch(batch))
        return np.concatenate(transformed_batches)
    else:
        return process_batch(X)


def choose_index_from_prob_matrix(prob_matrix, rng=None):
    # Flatten the matrix to 1D array
    flat_probs = np.array(prob_matrix).flatten()

    # Create RNG if not provided
    if rng is None:
        rng = np.random.default_rng()

    # Make random choice
    flat_index = rng.choice(len(flat_probs), p=flat_probs)

    # Convert back to 2D indices
    row = flat_index // prob_matrix.shape[1]
    col = flat_index % prob_matrix.shape[1]

    return row, col


def compute_wasserstein(m, n, V, U, inbetween=None):
    """
    Compute the Wasserstein distance between two Gaussians N(m, V) and N(n, U) (means and covariance matrices)
    """
    if inbetween is None:
        inbetween = scipy.linalg.sqrtm(sqrtm_cov(U) @ V @ sqrtm_cov(U))

    return np.linalg.norm(n - m) ** 2 + np.trace(V + U + inbetween)


def compute_gaussian_T_A(s: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the transport matrix T_A between two multivariate Gaussians with means 0 and source covariance s to target covariance t
    """
    inbetween = scipy.linalg.sqrtm(sqrtm_cov(s) @ t @ sqrtm_cov(s))
    return (inv_sqrtm_cov(s) @ inbetween @ inv_sqrtm_cov(s)), inbetween


def apply_transport_map_Gaussians(
    X: np.ndarray,
    mean_source: np.ndarray,
    mean_target: np.ndarray,
    transport_matrix: np.ndarray,
) -> np.ndarray:
    """
    Apply the transport matrix to a set of points X from a source Gaussian with mean mean_source to a target Gaussian with mean mean_target
    """
    b = mean_target - (transport_matrix @ mean_source.T).T
    return (transport_matrix @ X.T).T + b


def sqrtm_cov(A: np.ndarray) -> np.ndarray:
    """
    Compute the square root of a covariance matrix (positive definite and symmetric)

    Notes
    -----
    Covariance matrices should be positive semi-definite, but numerical errors can
    introduce small negative eigenvalues. This function validates eigenvalues and
    clips small negative values to zero to ensure numerical stability.

    Raises
    ------
    ValueError
        If the matrix has significantly negative eigenvalues (< -1e-10), indicating
        it is not positive semi-definite.
    """
    eigenvals, eigenvecs = np.linalg.eigh(A)

    # Check for significantly negative eigenvalues
    if np.any(eigenvals < -1e-10):
        raise ValueError(
            f"Matrix is not positive semi-definite. "
            f"Minimum eigenvalue: {eigenvals.min():.2e}"
        )

    # Clip small negative eigenvalues to zero (numerical errors)
    eigenvals = np.maximum(eigenvals, 0)

    return eigenvecs @ np.diag(np.sqrt(eigenvals)) @ eigenvecs.T


def inv_sqrtm_cov(A: np.ndarray) -> np.ndarray:
    """
    Compute the inverse square root of a covariance matrix (positive definite and symmetric)

    Notes
    -----
    This function prevents division by zero by clipping eigenvalues to a minimum
    threshold (1e-10). This ensures numerical stability when working with
    ill-conditioned or singular covariance matrices.

    The minimum eigenvalue threshold (1e-10) is chosen to balance numerical
    stability against precision. Covariance matrices with eigenvalues smaller
    than this threshold are treated as effectively singular.
    """
    eigenvals, eigenvecs = np.linalg.eigh(A)

    # Clip eigenvalues to prevent division by zero
    # Use maximum to ensure all eigenvalues are at least 1e-10
    eigenvals = np.maximum(eigenvals, 1e-10)

    return eigenvecs @ np.diag(1 / np.sqrt(eigenvals)) @ eigenvecs.T


def ensure_valid_fc_data(transformed_points: np.ndarray) -> np.ndarray:
    # Ensure values between 0 and 255
    transformed_points = np.maximum(transformed_points, 0)
    transformed_points = np.minimum(transformed_points, 255)

    # Additional channel-specific constraints could be added here

    return transformed_points
