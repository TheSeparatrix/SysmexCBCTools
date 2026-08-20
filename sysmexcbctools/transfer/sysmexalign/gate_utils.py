"""
Gate-aware utilities for flow cytometry data

This module provides utilities for using manually-defined flow cytometry gates
to improve GMM fitting by ensuring rare populations are adequately represented.

Uses the FlowGate infrastructure from flow_gating_pipeline.py for gate definitions.

⚠️ **Important Note on Gates**:
The gate definitions currently available in flow_gates/ are manually derived to the
best of our ability through visual inspection of flow cytometry data. These gates
should be considered approximate and are intended for research purposes.

**Future Improvements:**
- Official Sysmex-provided gates would be preferable
- Gates derived from larger, validated datasets
- Adaptive gating methods that adjust to analyzer-specific characteristics
- Expert hematologist review and validation

For production use, consider validating the gate definitions against known standards
or obtaining official gating strategies from Sysmex.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .flow_gating_pipeline import FlowGate


def load_gates(gate_file: str, channel: str) -> Dict[str, FlowGate]:
    """
    Load gate definitions from a JSON file and convert to FlowGate objects.

    Parameters
    ----------
    gate_file : str
        Path to JSON file containing gate coordinates
    channel : str
        Channel name (RET, WDF, WNR, PLTF) to determine coordinate names

    Returns
    -------
    gates : dict
        Dictionary mapping population names to FlowGate objects
    """
    gate_path = Path(gate_file)
    if gate_path.suffix != '.json':
        raise ValueError(
            f"Unsupported gate file format: {gate_path.suffix}. Must be .json"
        )

    # Map channels to their coordinate names
    channel_coords = {
        'RET': ('SFL', 'FSC'),
        'WDF': ('SSC', 'SFL'),
        'WNR': ('SFL', 'FSC'),
        'PLTF': ('SFL', 'FSC'),
    }
    coords = channel_coords.get(channel, ('SFL', 'FSC'))

    with open(gate_file) as f:
        gates_data = json.load(f)

    return {
        pop_name: FlowGate(pop_name, vertices, coords)
        for pop_name, vertices in gates_data.items()
        if len(vertices) > 0
    }


def find_default_gate_file(channel: str) -> Optional[str]:
    """
    Find the default gate file for a channel by searching common locations.

    Parameters
    ----------
    channel : str
        Channel name (RET, WDF, WNR, PLTF)

    Returns
    -------
    gate_file : str or None
        Path to gate file if found, else None
    """
    filename = f'{channel}_gates.json'
    search_paths = [
        Path('flow_gates'),
        Path('../flow_gates'),
        Path('../../flow_gates'),
        Path(__file__).parent.parent / 'flow_gates',
    ]

    for search_path in search_paths:
        for candidate in (search_path / 'json_gates' / filename, search_path / filename):
            if candidate.exists():
                return str(candidate)

    return None


def classify_points_by_gate(
    data: np.ndarray,
    gates: Dict[str, FlowGate],
    column_indices: Tuple[int, int] = (0, 1),
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Classify each data point into gated populations using FlowGate objects.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        Flow cytometry data points (can be more than 2D, will extract relevant columns)
    gates : dict
        Dictionary mapping population names to FlowGate objects
    column_indices : tuple of int, default=(0, 1)
        Which columns of data to use for gating (usually first two for 2D gates)

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Population label for each point (integer index or -1 for ungated)
    label_map : dict
        Mapping from integer label to population name
    """
    # Extract relevant columns
    if data.ndim == 1:
        data_2d = data.reshape(-1, 2)
    elif data.shape[1] > 2:
        data_2d = data[:, list(column_indices)]
    else:
        data_2d = data

    n_samples = len(data_2d)
    labels = np.full(n_samples, -1, dtype=int)  # -1 = ungated

    # Create label map
    label_map = {}
    for i, pop_name in enumerate(gates.keys()):
        label_map[i] = pop_name

    # Classify points using FlowGate's contains_points method
    for i, (pop_name, gate) in enumerate(gates.items()):
        if len(gate.path_vertices) < 3:  # Need at least 3 points for a polygon
            continue

        in_gate = gate.contains_points(data_2d)

        # Assign to this population (later gates can override earlier ones)
        labels[in_gate] = i

    return labels, label_map


def initialize_gmm_means_from_gates(
    data: np.ndarray,
    gates: Dict[str, FlowGate],
    n_components: int,
    method: str = 'equal',
    random_state: Optional[int] = None,
    column_indices: Tuple[int, int] = (0, 1),
) -> np.ndarray:
    """
    Initialize GMM component means by distributing them across gate regions.

    Instead of random initialization or k-means++, this ensures GMM components
    are initialized spread across all gated populations, preventing collapse
    to high-density regions.

    **Important Note**: The component allocation methods (equal/sqrt/proportional) are
    all somewhat arbitrary. Ideally, components should be allocated based on how
    "non-Gaussian" each population's distribution is. If results are poor, revisit
    this allocation strategy using distribution complexity metrics (e.g., BIC-based).

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        Flow cytometry data
    gates : dict
        Dictionary mapping population names to FlowGate objects
    n_components : int
        Total number of GMM components to initialize
    method : str, default='equal'
        Method for allocating components to populations:
        - 'equal': Equal components per population
        - 'sqrt': Proportional to sqrt of population size
        - 'proportional': Proportional to population size
    random_state : int, optional
        Random seed for reproducibility
    column_indices : tuple of int, default=(0, 1)
        Which columns to use for gate classification

    Returns
    -------
    initial_means : ndarray of shape (n_components, n_features)
        Initial mean positions for GMM components, distributed across gates
    """
    rng = np.random.default_rng(seed=random_state)

    # Classify points by gate
    labels, label_map = classify_points_by_gate(data, gates, column_indices)

    # Count populations and assign components proportionally
    unique_labels = np.unique(labels)
    pop_counts = {label: np.sum(labels == label) for label in unique_labels}

    # Allocate components to populations based on method
    components_per_pop = {}
    assigned_components = 0

    if method == 'equal':
        # Equal components per population
        n_pops = len(unique_labels)
        base_alloc = n_components // n_pops
        remainder = n_components % n_pops

        for i, label in enumerate(sorted(unique_labels)):
            # Give extra component to first 'remainder' populations
            n_alloc = base_alloc + (1 if i < remainder else 0)
            components_per_pop[label] = n_alloc
            assigned_components += n_alloc

    elif method == 'sqrt':
        # Proportional to sqrt of population size
        sqrt_counts = {label: np.sqrt(count) for label, count in pop_counts.items()}
        total_sqrt = sum(sqrt_counts.values())

        for label in sorted(unique_labels):
            # Allocate proportionally, but ensure at least 2 components
            n_alloc = max(2, int(n_components * sqrt_counts[label] / total_sqrt))
            components_per_pop[label] = n_alloc
            assigned_components += n_alloc

    elif method == 'proportional':
        # Proportional to actual population size
        total_count = sum(pop_counts.values())

        for label in sorted(unique_labels):
            # Allocate proportionally, but ensure at least 2 components
            n_alloc = max(2, int(n_components * pop_counts[label] / total_count))
            components_per_pop[label] = n_alloc
            assigned_components += n_alloc
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'equal', 'sqrt', or 'proportional'")

    # Adjust if we over/under-allocated
    diff = n_components - assigned_components
    if diff != 0:
        # Give extra components to largest population or take from it
        largest_pop = max(pop_counts.keys(), key=lambda k: pop_counts[k])
        components_per_pop[largest_pop] += diff

    print(f"  Gate-informed GMM initialization (method='{method}'):")
    print(f"    Total components: {n_components}")
    for label in sorted(unique_labels):
        n_comp = components_per_pop[label]
        pop_name = label_map.get(label, 'Ungated')
        pop_pct = pop_counts[label] / len(data) * 100
        comp_pct = n_comp / n_components * 100
        print(f"    {pop_name}: {n_comp} components ({comp_pct:.1f}%) for {pop_counts[label]:,} points ({pop_pct:.1f}%)")

    # Initialize means by sampling from each population
    initial_means = []

    for label, n_comp in components_per_pop.items():
        pop_data = data[labels == label]

        if len(pop_data) == 0:
            continue

        if n_comp >= len(pop_data):
            # If we need more components than we have points, sample with replacement
            selected = pop_data[rng.choice(len(pop_data), n_comp, replace=True)]
        else:
            # Sample points from this population to use as initial means
            selected = pop_data[rng.choice(len(pop_data), n_comp, replace=False)]

        initial_means.append(selected)

    # Concatenate all initial means
    initial_means = np.vstack(initial_means)

    # Shuffle to avoid any ordering bias
    shuffle_idx = rng.permutation(len(initial_means))
    initial_means = initial_means[shuffle_idx]

    return initial_means[:n_components]  # Ensure exactly n_components
