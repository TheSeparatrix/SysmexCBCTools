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
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .flow_gating_pipeline import FlowGate


def load_gates_from_pickle(gate_file: str) -> Dict[str, FlowGate]:
    """
    Load gate definitions from pickle file.

    Parameters
    ----------
    gate_file : str
        Path to pickle file containing FlowGate objects

    Returns
    -------
    gates : dict
        Dictionary mapping population names to FlowGate objects
    """
    with open(gate_file, 'rb') as f:
        gates = pickle.load(f)
    return gates


def load_gates_from_json(gate_file: str, channel: str) -> Dict[str, FlowGate]:
    """
    Load gate definitions from JSON file and convert to FlowGate objects.

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

    # Convert to FlowGate objects
    gates = {}
    for pop_name, vertices in gates_data.items():
        if len(vertices) > 0:
            gates[pop_name] = FlowGate(pop_name, vertices, coords)

    return gates


def load_gates(gate_file: str, channel: str) -> Dict[str, FlowGate]:
    """
    Load gate definitions from pickle or JSON file.

    Parameters
    ----------
    gate_file : str
        Path to gate file (.pkl or .json)
    channel : str
        Channel name (RET, WDF, WNR, PLTF)

    Returns
    -------
    gates : dict
        Dictionary mapping population names to FlowGate objects
    """
    gate_path = Path(gate_file)

    if gate_path.suffix == '.pkl':
        return load_gates_from_pickle(gate_file)
    elif gate_path.suffix == '.json':
        return load_gates_from_json(gate_file, channel)
    else:
        raise ValueError(f"Unsupported gate file format: {gate_path.suffix}. "
                        f"Must be .pkl or .json")


def find_default_gate_file(channel: str, search_paths: Optional[List[str]] = None) -> Optional[str]:
    """
    Find the default gate file for a channel by searching common locations.

    Searches for both pickle (.pkl) and JSON (.json) files, preferring pickle.

    Parameters
    ----------
    channel : str
        Channel name (RET, WDF, WNR, PLTF)
    search_paths : list of str, optional
        Additional paths to search. Default searches:
        - ./flow_gates/
        - ../flow_gates/
        - ../../flow_gates/
        - Package-relative paths

    Returns
    -------
    gate_file : str or None
        Path to gate file if found, else None
    """
    if search_paths is None:
        search_paths = []

    # Add standard search paths
    standard_paths = [
        'flow_gates',
        '../flow_gates',
        '../../flow_gates',
        Path(__file__).parent.parent / 'flow_gates',
    ]
    search_paths = search_paths + standard_paths

    # Try pickle first, then JSON
    for ext in ['.pkl', '.json']:
        filename = f'{channel}_gates{ext}'

        for search_path in search_paths:
            search_path = Path(search_path)

            # Check root of search path
            gate_file = search_path / filename
            if gate_file.exists():
                return str(gate_file)

            # Check json_gates subdirectory (for JSON files)
            if ext == '.json':
                gate_file = search_path / 'json_gates' / filename
                if gate_file.exists():
                    return str(gate_file)

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


def stratified_sample_by_gates(
    data: np.ndarray,
    gates: Dict[str, FlowGate],
    target_samples: int = 1_000_000,
    min_samples_per_gate: int = 50_000,
    balance_method: str = 'sqrt',
    random_state: Optional[int] = None,
    column_indices: Tuple[int, int] = (0, 1),
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Perform stratified sampling using gate definitions to ensure rare populations
    are adequately represented.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        Flow cytometry data points
    gates : dict
        Dictionary mapping population names to FlowGate objects
    target_samples : int, default=1_000_000
        Target number of samples in the output
    min_samples_per_gate : int, default=50_000
        Minimum number of samples to draw from each gated population
    balance_method : str, default='sqrt'
        Method for balancing populations:
        - 'sqrt': Sample proportional to sqrt of population size (balances rare/common)
        - 'equal': Sample equally from each population
        - 'proportional': Sample proportionally (no balancing, equivalent to random sampling)
    random_state : int, optional
        Random seed for reproducibility
    column_indices : tuple of int, default=(0, 1)
        Which columns to use for gate classification

    Returns
    -------
    sampled_data : ndarray of shape (n_sampled, n_features)
        Stratified sample of data
    sample_counts : dict
        Number of samples drawn from each population
    """
    rng = np.random.default_rng(seed=random_state)

    # Classify all points
    labels, label_map = classify_points_by_gate(data, gates, column_indices)

    # Count points in each population
    unique_labels = np.unique(labels)
    pop_counts = {label: np.sum(labels == label) for label in unique_labels}

    print("  Gate-based stratified sampling:")
    print(f"    Total data points: {len(data):,}")
    for label, count in pop_counts.items():
        pop_name = label_map.get(label, 'Ungated')
        print(f"    {pop_name}: {count:,} ({count/len(data)*100:.1f}%)")

    # Compute sampling weights based on balance method
    if balance_method == 'sqrt':
        # Sample proportional to sqrt of population size
        # This gives rare populations more weight while not completely ignoring common ones
        weights = {label: np.sqrt(count) for label, count in pop_counts.items()}
    elif balance_method == 'equal':
        # Sample equally from each population
        weights = {label: 1.0 for label in pop_counts.keys()}
    elif balance_method == 'proportional':
        # Sample proportionally (no balancing)
        weights = {label: float(count) for label, count in pop_counts.items()}
    else:
        raise ValueError(f"Unknown balance_method: {balance_method}")

    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    weights = {label: w / total_weight for label, w in weights.items()}

    # Compute target samples per population
    target_per_pop = {}
    for label, weight in weights.items():
        n_target = int(weight * target_samples)
        # Ensure minimum samples for gated populations (not ungated)
        if label != -1:  # -1 is ungated
            n_target = max(n_target, min_samples_per_gate)
        # Don't sample more than available
        n_target = min(n_target, pop_counts[label])
        target_per_pop[label] = n_target

    # Sample from each population
    sampled_data_list = []
    sample_counts = {}

    for label, n_target in target_per_pop.items():
        pop_data = data[labels == label]
        if n_target > 0 and len(pop_data) > 0:
            if n_target >= len(pop_data):
                # Take all points
                sampled = pop_data
            else:
                # Random sample
                indices = rng.choice(len(pop_data), n_target, replace=False)
                sampled = pop_data[indices]

            sampled_data_list.append(sampled)
            pop_name = label_map.get(label, 'Ungated')
            sample_counts[pop_name] = len(sampled)

    # Concatenate all samples
    sampled_data = np.concatenate(sampled_data_list, axis=0)

    # Shuffle the combined dataset
    shuffle_indices = rng.permutation(len(sampled_data))
    sampled_data = sampled_data[shuffle_indices]

    print("  Stratified sampling results:")
    print(f"    Final sample size: {len(sampled_data):,}")
    for pop_name, count in sample_counts.items():
        print(f"    {pop_name}: {count:,} ({count/len(sampled_data)*100:.1f}%)")

    return sampled_data, sample_counts


def get_default_balance_method(channel: str) -> str:
    """
    Get the default balance method for a channel based on typical data characteristics.

    Parameters
    ----------
    channel : str
        Channel name (RET, WDF, WNR, PLTF)

    Returns
    -------
    balance_method : str
        Recommended balance method for this channel
    """
    # Channels with very rare populations benefit from more aggressive balancing
    rare_population_channels = ['RET', 'WNR']  # Reticulocytes and nucleated RBCs are rare

    if channel in rare_population_channels:
        return 'sqrt'  # Balanced sampling
    else:
        return 'sqrt'  # Default to sqrt for all channels


def get_default_min_samples_per_gate(channel: str) -> int:
    """
    Get the default minimum samples per gate for a channel.

    Parameters
    ----------
    channel : str
        Channel name (RET, WDF, WNR, PLTF)

    Returns
    -------
    min_samples : int
        Minimum samples to ensure from each gated population
    """
    # Ensure at least this many samples from each rare population
    return 50_000


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
