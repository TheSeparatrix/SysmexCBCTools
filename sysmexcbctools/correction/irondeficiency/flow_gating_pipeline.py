import os
import pickle

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output, display
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector
from sklearn.mixture import GaussianMixture

sysmex_channels_to_flowcyto_channels = {
    "WNR": ["SFL", "FSC", "SSC", "FSCW"],
    "WDF": ["SSC", "SFL", "FSC", "FSCW"],
    "RET": ["SFL", "FSC"],
    "PLTF": ["SFL", "FSC", "SSC", "FSCW"],
}


class FlowGate:
    """Class to represent a flow cytometry gate with a name and path"""

    def __init__(self, name, path_vertices, channels):
        self.name = name
        self.path_vertices = path_vertices
        self.channels = channels  # Store the channel names this gate was created for
        self.path = Path(path_vertices)

    def contains_points(self, points):
        """Check which points are inside the gate"""
        return self.path.contains_points(points)


class FlowGatingSystem:
    """Main class for the flow cytometry gating pipeline"""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.data = None
        self.gates = {}
        self.current_gate = None
        self.selector = None
        self.fig = None
        self.ax = None
        self.channels = None
        self.column_names = None

    def load_data(self, csv_file):
        """Load flow cytometry data from a CSV file"""
        self.data = pd.read_csv(csv_file)
        basename = os.path.basename(csv_file)
        if self.verbose:
            print(
                f"Loaded data with {len(self.data)} events and {len(self.data.columns)} parameters"
            )
            print(f"Available channels: {', '.join(self.data.columns)}")
        for sysmexname in sysmex_channels_to_flowcyto_channels.keys():
            if basename.startswith(sysmexname):
                self.column_names = sysmex_channels_to_flowcyto_channels[sysmexname]
                # print(f"Using channels: {', '.join(self.column_names)}")
                break
        return self.data

    def show_data(self, x_channel, y_channel, sample_size=None):
        """Create a scatter plot of two selected channels"""
        self.channels = (x_channel, y_channel)

        # Sample data if needed (for performance with large datasets)
        if sample_size and len(self.data) > sample_size:
            plot_data = self.data.sample(sample_size)
        else:
            plot_data = self.data

        # Create the plot
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.scatter(
            plot_data[x_channel], plot_data[y_channel], s=1, alpha=0.5, c="blue"
        )

        self.ax.set_xlabel(x_channel)
        self.ax.set_ylabel(y_channel)
        self.ax.set_title(f"{x_channel} vs {y_channel}")

        plt.show()

    def start_gate_drawing(self, gate_name=None):
        """Start interactive gate drawing"""
        if self.fig is None or self.ax is None:
            print("Please show data first using show_data()")
            return

        if gate_name is None:
            gate_name = f"Gate_{len(self.gates) + 1}"

        self.current_gate = gate_name
        print(f"Drawing gate: {gate_name}")
        print("Click to draw polygon vertices. Close the polygon to finish.")

        # Create the polygon selector
        self.selector = PolygonSelector(
            self.ax,
            self.on_polygon_complete,
            useblit=True,
            props=dict(color="red", linestyle="-", linewidth=2, alpha=0.5),
        )

        # Keep the plot window open
        plt.show(block=True)

    def on_polygon_complete(self, vertices):
        """Callback for when polygon drawing is complete"""
        # Create a new gate with the drawn vertices
        self.gates[self.current_gate] = FlowGate(
            self.current_gate, vertices, self.channels
        )

        print(f"Gate '{self.current_gate}' created with {len(vertices)} vertices")

        # Add the gate to the plot for visualization
        polygon = plt.Polygon(vertices, fill=True, alpha=0.3, color="red")
        self.ax.add_patch(polygon)
        self.ax.set_title(
            f"{self.channels[0]} vs {self.channels[1]} - Gate: {self.current_gate}"
        )

        plt.draw()

    def save_gates(self, filename):
        """Save all gates to a file"""
        with open(filename, "wb") as f:
            pickle.dump(self.gates, f)
        print(f"Saved {len(self.gates)} gates to {filename}")

    def load_gates(self, filename):
        """Load gates from a file"""
        with open(filename, "rb") as f:
            self.gates = pickle.load(f)
        if self.verbose:
            print(f"Loaded {len(self.gates)} gates from {filename}")
        return self.gates

    def apply_gate(self, gate_name, data=None):
        """Apply a gate to data and return the gated events"""
        if data is None:
            data = self.data

        if gate_name not in self.gates:
            print(f"Gate '{gate_name}' not found")
            return None

        gate = self.gates[gate_name]
        x_channel, y_channel = gate.channels

        # Check if the data has the required channels
        if x_channel not in data.columns or y_channel not in data.columns:
            print(f"Data missing required channels: {x_channel}, {y_channel}")
            return None

        # Extract the points
        points = data[[x_channel, y_channel]].values

        # Apply the gate
        mask = gate.contains_points(points)

        # Return the gated events
        return data[mask]

    def apply_all_gates(self, data=None):
        """Apply all gates to the data and add a 'gate_label' column"""
        if data is None:
            data = self.data

        # Create a new column filled with None
        data = data.copy()
        data["gate_label"] = None

        # Apply each gate in order
        for gate_name in self.gates:
            # Get the indices of events in this gate
            gate = self.gates[gate_name]
            x_channel, y_channel = gate.channels

            # Check if the data has the required channels
            if x_channel not in data.columns or y_channel not in data.columns:
                print(
                    f"Skipping gate '{gate_name}'. Data missing required channels: {x_channel}, {y_channel}"
                )
                continue

            # Get points
            points = data[[x_channel, y_channel]].values

            # Find points in the gate that don't already have a label
            mask = gate.contains_points(points) & (data["gate_label"].isna())

            # Label these points
            data.loc[mask, "gate_label"] = gate_name

        return data

    def fit_gaussian_to_gate(self, gate_name, data=None):
        """Fit a Gaussian mixture model to the gated events"""
        if data is None:
            data = self.data

        # Get the gated events
        gated_data = self.apply_gate(gate_name, data)

        # points = gated_data[[x_channel, y_channel]].values
        points = gated_data[self.column_names].values
        # remove any rows containing 0 or 255 for fitting
        points = points[(points != 0).all(axis=1) & (points != 255).all(axis=1)]

        if gated_data is None or len(points) < 10:
            print(f"No events found in gate '{gate_name}'")
            return {
                "mean": [-1] * len(self.column_names),
                "covariance": -1 * np.eye(len(self.column_names)),
                "count": 0,
            }

        # Fit a Gaussian model
        # gate = self.gates[gate_name]
        # x_channel, y_channel = gate.channels

        # set any NaN values to 0
        points = np.nan_to_num(points)
        gmm = GaussianMixture(n_components=1, covariance_type="full")
        try:
            gmm.fit(points)
        except Exception as e:
            print(f"Error fitting Gaussian to gate '{gate_name}': {e}")
            print(gated_data)
        # Extract parameters
        params = {
            "mean": gmm.means_[0],
            "covariance": gmm.covariances_[0],
            "count": len(gated_data),
        }

        return params

    def fit_gaussians_to_all_gates(self, data=None):
        """Fit Gaussian models to all gates"""
        if data is None:
            data = self.data

        results = {}

        for gate_name in self.gates:
            params = self.fit_gaussian_to_gate(gate_name, data)
            if params is not None:
                results[gate_name] = params

        return results

    def show_gates(self, x_channel=None, y_channel=None, data=None, sample_size=10000):
        """Show all gates on a scatter plot"""
        if data is None:
            data = self.data

        # Sample data if needed
        if sample_size and len(data) > sample_size:
            plot_data = data.sample(sample_size)
        else:
            plot_data = data

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # If channels not specified, use the first gate's channels
        if x_channel is None or y_channel is None:
            if len(self.gates) > 0:
                first_gate = next(iter(self.gates.values()))
                x_channel, y_channel = first_gate.channels
            else:
                print("No gates available and no channels specified")
                return

        # Plot the data
        ax.scatter(plot_data[x_channel], plot_data[y_channel], s=1, alpha=0.3, c="gray")

        # Plot each gate
        colors = plt.cm.tab10.colors
        for i, (gate_name, gate) in enumerate(self.gates.items()):
            # Skip gates for different channels
            if gate.channels != (x_channel, y_channel):
                continue

            # Plot the gate
            color = colors[i % len(colors)]
            polygon = plt.Polygon(
                gate.path_vertices, fill=True, alpha=0.3, color=color, label=gate_name
            )
            ax.add_patch(polygon)

            # Get the gated events for highlighting
            gated_data = self.apply_gate(gate_name, plot_data)
            if gated_data is not None and len(gated_data) > 0:
                ax.scatter(
                    gated_data[x_channel],
                    gated_data[y_channel],
                    s=2,
                    alpha=0.8,
                    c=color,
                )

        ax.set_xlabel(x_channel)
        ax.set_ylabel(y_channel)
        ax.set_title(f"{x_channel} vs {y_channel} - Gates")
        ax.legend()

        plt.show()


# Interactive interface for the gating system
def create_interactive_gating_interface():
    """Create an interactive Jupyter notebook interface for the gating system"""

    # Create the gating system
    fs = FlowGatingSystem()

    # File selection widget
    file_selector = widgets.Text(
        value="",
        placeholder="Enter path to CSV file",
        description="Data file:",
        disabled=False,
    )

    load_button = widgets.Button(
        description="Load Data",
        disabled=False,
        button_style="",
        tooltip="Load data from CSV file",
    )

    # Channel selection widgets
    channel_x = widgets.Dropdown(
        options=[],
        description="X Channel:",
        disabled=False,
    )

    channel_y = widgets.Dropdown(
        options=[],
        description="Y Channel:",
        disabled=False,
    )

    show_data_button = widgets.Button(
        description="Show Data",
        disabled=False,
        button_style="",
        tooltip="Show scatter plot",
    )

    # Gate controls
    gate_name_input = widgets.Text(
        value="",
        placeholder="Enter gate name",
        description="Gate name:",
        disabled=False,
    )

    draw_gate_button = widgets.Button(
        description="Draw Gate",
        disabled=False,
        button_style="",
        tooltip="Start drawing a gate",
    )

    # Save/load gate controls
    gate_file_input = widgets.Text(
        value="gates.pkl",
        placeholder="Enter file name",
        description="Gate file:",
        disabled=False,
    )

    save_gates_button = widgets.Button(
        description="Save Gates",
        disabled=False,
        button_style="",
        tooltip="Save gates to file",
    )

    load_gates_button = widgets.Button(
        description="Load Gates",
        disabled=False,
        button_style="",
        tooltip="Load gates from file",
    )

    show_gates_button = widgets.Button(
        description="Show Gates",
        disabled=False,
        button_style="",
        tooltip="Show all gates",
    )

    # Output widget
    output = widgets.Output()

    # Define button callbacks
    def on_load_button_clicked(b):
        with output:
            clear_output()
            if file_selector.value:
                data = fs.load_data(file_selector.value)
                if data is not None:
                    channel_x.options = list(data.columns)
                    channel_y.options = list(data.columns)
            else:
                print("Please enter a file path")

    def on_show_data_button_clicked(b):
        with output:
            clear_output()
            if fs.data is not None:
                fs.show_data(channel_x.value, channel_y.value)
            else:
                print("Please load data first")

    def on_draw_gate_button_clicked(b):
        with output:
            clear_output()
            if fs.data is not None:
                gate_name = gate_name_input.value if gate_name_input.value else None
                fs.start_gate_drawing(gate_name)
            else:
                print("Please load data first")

    def on_save_gates_button_clicked(b):
        with output:
            clear_output()
            if len(fs.gates) > 0:
                fs.save_gates(gate_file_input.value)
            else:
                print("No gates to save")

    def on_load_gates_button_clicked(b):
        with output:
            clear_output()
            try:
                fs.load_gates(gate_file_input.value)
            except Exception as e:
                print(f"Error loading gates: {e}")

    def on_show_gates_button_clicked(b):
        with output:
            clear_output()
            if fs.data is not None and len(fs.gates) > 0:
                fs.show_gates(channel_x.value, channel_y.value)
            else:
                print("Please load data and create or load gates first")

    # Connect callbacks to buttons
    load_button.on_click(on_load_button_clicked)
    show_data_button.on_click(on_show_data_button_clicked)
    draw_gate_button.on_click(on_draw_gate_button_clicked)
    save_gates_button.on_click(on_save_gates_button_clicked)
    load_gates_button.on_click(on_load_gates_button_clicked)
    show_gates_button.on_click(on_show_gates_button_clicked)

    # Layout the widgets
    file_box = widgets.HBox([file_selector, load_button])
    channel_box = widgets.HBox([channel_x, channel_y, show_data_button])
    gate_box = widgets.HBox([gate_name_input, draw_gate_button])
    gate_file_box = widgets.HBox(
        [gate_file_input, save_gates_button, load_gates_button, show_gates_button]
    )

    # Assemble the final layout
    layout = widgets.VBox([file_box, channel_box, gate_box, gate_file_box, output])

    return layout, fs


# Function to perform batch processing on multiple samples using saved gates
def process_samples(
    gating_system, sample_files, gate_file, output_folder="results", verbose=False
):
    """
    Process multiple samples using saved gates

    Parameters:
    -----------
    gating_system : FlowGatingSystem
        The gating system with loaded gates
    sample_files : list
        List of paths to sample CSV files
    gate_file : str
        Path to the saved gates file
    output_folder : str
        Folder to save results

    Returns:
    --------
    dict
        Dictionary of results for each sample
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the gates
    gating_system.load_gates(gate_file)

    # Process each sample
    results = {}

    for sample_file in sample_files:
        sample_name = os.path.basename(sample_file).split(".")[0]
        if verbose:
            print(f"Processing sample: {sample_name}")

        # Load the sample data
        gating_system.load_data(sample_file)

        # Apply gates and add labels
        labeled_data = gating_system.apply_all_gates()

        # Fit Gaussians to each gate
        gaussian_params = gating_system.fit_gaussians_to_all_gates()

        # # Save the labeled data
        # labeled_data.to_csv(f"{output_folder}/{sample_name}_labeled.csv", index=False)

        # # Save the Gaussian parameters
        # with open(f"{output_folder}/{sample_name}_gaussian_params.pkl", "wb") as f:
        #     pickle.dump(gaussian_params, f)

        # Store the results
        results[sample_name] = {
            "labeled_data": labeled_data,
            "gaussian_params": gaussian_params,
            "gates": list(gating_system.gates.keys()),
        }

        if verbose:
            print(
                f"Processed {sample_name}: {len(labeled_data)} events, {len(gaussian_params)} gates"
            )

    return results
