import json
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd


def log_data_shape(
    step_name: str,
    data: Union[pd.DataFrame, np.ndarray, List, Tuple],
    operation: str,
    metrics_dir: str = "metrics",
) -> None:
    """
    Log the shape of data after an operation in a structured format for DVC tracking.

    Args:
        step_name: Name of the current processing step
        data: The data object whose shape to log
        operation: Description of the operation performed
        metrics_dir: Directory to store metrics files
    """
    # Create metrics directory if it doesn't exist
    os.makedirs(metrics_dir, exist_ok=True)

    # Determine shape based on data type
    if isinstance(data, pd.DataFrame):
        shape = {
            "rows": data.shape[0],
            "columns": data.shape[1],
            "column_dtypes": str(data.dtypes.to_dict()),
        }
    elif isinstance(data, np.ndarray):
        shape = {"shape": list(data.shape), "dtype": str(data.dtype)}
    elif isinstance(data, (list, tuple)):
        shape = {"length": len(data)}
        if len(data) > 0 and isinstance(data[0], (list, tuple)):
            shape["sample_element_length"] = len(data[0])
    else:
        shape = {"type": str(type(data))}

    # Create or update the metrics file
    metrics_file = os.path.join(metrics_dir, f"{step_name}_shape.json")

    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {"step": step_name, "operations": []}

    # Add the new operation with timestamp
    import datetime

    metrics["operations"].append(
        {
            "operation": operation,
            "timestamp": datetime.datetime.now().isoformat(),
            "shape": shape,
        }
    )

    # Write updated metrics
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    # Print feedback for interactive use
    # Create a copy of shape without column_dtypes for printing
    print_shape = {k: v for k, v in shape.items() if k != "column_dtypes"}
    print(f"[{step_name}] After {operation}: {print_shape}")
