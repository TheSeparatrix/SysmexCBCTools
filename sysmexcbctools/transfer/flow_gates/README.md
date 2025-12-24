# Flow Cytometry Gates

This directory contains manually-defined flow cytometry gate definitions for Sysmex analyzer channels.

## ⚠️ Important Warning

**These gate definitions are manually derived and approximate.**

The gates in this directory were created through visual inspection of flow cytometry data and represent our best effort to identify cell populations (RBC, RET, PLT, etc.). However:

- **These gates should be considered approximate and are intended for research purposes**
- They have not been validated by expert hematologists
- They are not official Sysmex-provided gates
- They may not generalize well across different analyzers or patient populations

## Recommended for Future Improvements

For production or clinical use, consider:

1. **Official Sysmex gates**: Contact Sysmex for their official gating strategies
2. **Expert validation**: Have hematology experts review and validate the gates
3. **Larger datasets**: Derive gates from larger, more diverse patient populations
4. **Adaptive methods**: Use automated gating algorithms that adapt to analyzer characteristics
5. **Quality control**: Validate gates against known standards or reference samples

## File Formats

Gates are available in two formats:

### Pickle Format (Recommended)
- `{CHANNEL}_gates.pkl` - FlowGate objects with full metadata
- Contains: RET_gates.pkl, WDF_gates.pkl, WNR_gates.pkl, PLTF_gates.pkl
- Includes channel names and path vertices

### JSON Format (Alternative)
- `json_gates/{CHANNEL}_gates.json` - Simple coordinate lists
- Lightweight and human-readable
- Automatically converted to FlowGate objects when loaded

## Usage

The `FlowTransformer` automatically searches for and loads gates from this directory:

```python
from sysmexcbctools.transfer.sysmexalign import FlowTransformer

# Gates are automatically loaded if found
transformer = FlowTransformer(
    channel='RET',
    use_gate_sampling=True  # Default: True
)

# Or specify a custom gate file
transformer = FlowTransformer(
    channel='RET',
    gate_file='path/to/custom_gates.pkl'
)
```

## Gate Definitions

### RET Channel (Reticulocytes)
- **RBC**: Mature red blood cells (main population)
- **RET**: Reticulocytes (young red blood cells, ~1-2% of RBCs)
- **PLT**: Platelets (small cells)

### WDF Channel (White Cell Differential)
- Multiple white blood cell populations (neutrophils, lymphocytes, etc.)

### WNR Channel (White Nucleated RBC)
- Detection of nucleated red blood cells (rare)

### PLTF Channel (Platelet Fluorescence)
- Platelet populations with different RNA content

## Why Gates Matter for GMM Fitting

Without gate-aware sampling, GMM components collapse to high-density regions (e.g., RBC), completely ignoring rare but important populations (RET, PLT). Using these gates:

- Ensures rare populations are adequately represented during GMM fitting
- Prevents GMM collapse to majority populations
- Results in better cross-analyzer alignment for all cell types

See the notebook `examples/notebooks/07_transfer_flow_cytometry.ipynb` for diagnostic visualizations.
