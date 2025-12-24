# SysmexCBCTools

**A unified Python toolkit for processing and analysing Sysmex haematology analyser data.**

This repository consolidates multiple codes and functions developed throughout a PhD project, focusing on analysis of Sysmex complete blood count (CBC) data and general domain shift mitigation for tabular data. All modules feature scikit-learn-style APIs demonstrated in example Jupyter notebooks.

---

## Modules

### Data Module

Process XN_SAMPLE.csv files from raw Sysmex XN haematology analyser exports. Feature name lists were curated by Allerdien Visser (AUMC Amsterdam), with additional features contributed by Andrew Gibbs (UCL).

**API:** `XNSampleProcessor`

### Transfer Module

GMM-OT framework for aligning flow cytometry measurements between different Sysmex analyser systems using optimal transport. Transform raw flow cytometry data from one analyser to another using reference measurements (parallel samples, QC samples, or healthy population data).

**Method:** Delon, J. & Desolneux, A. (2020). "A Wasserstein-Type Distance in the Space of Gaussian Mixture Models". *SIAM Journal on Imaging Sciences*, 13(2), 936-970. doi: [10.1137/19M1301047](https://doi.org/10.1137/19M1301047)

**Dependencies:** Python Optimal Transport ([POT](https://github.com/PythonOT/POT)) package

**API:** `FlowTransformer`, `ImpedanceTransformer`, `XNSampleTransformer`

### Correction Module

GAM-based correction of tabular data against multiple categorical or continuous covariates (e.g., sample age, processing time, seasonal effects).

**Method:** Astle, W.J. et al. (2016). "The Allelic Landscape of Human Blood Cell Trait Variation and Links to Common Complex Disease". *Cell*, 167(5), 1415-1429.e19. doi: [10.1016/j.cell.2016.10.042](https://doi.org/10.1016/j.cell.2016.10.042)

**Dependencies:** [pyGAM](https://github.com/dswah/pyGAM) package for GAM model fitting

**API:** `GAMCorrector`

### Dis-AE 2 Module

Multi-domain shift mitigation for tabular data through domain-invariant representation learning. Unlike traditional domain adaptation methods that treat domains as single labels, Dis-AE 2 models multiple domain shifts simultaneously (e.g., "sample age AND time of year") and removes them independently.

**Base method:** Kreuter, D. et al. (2023). "Dis-AE: Multi-domain & Multi-task Generalisation on Real-World Clinical Data". arXiv:2306.09177. doi: [10.48550/arXiv.2306.09177](https://doi.org/10.48550/arXiv.2306.09177)

**Dis-AE 2 improvements over Dis-AE 1:**
- Domain-separated latent space with shared (domain-invariant) and private (reconstruction) features
- Alternating adversarial training instead of gradient reversal layers
- Improved training stability and performance
- Orthogonality loss encouraging clean separation between shared and private representations

**Implementation:** PyTorch

**API:** `DisAE`

**Note:** Dis-AE 2 architecture is not yet published. The implementation differs from the arXiv preprint cited above.

---

## Installation

```bash
# Install with all dependencies (recommended)
pip install -e ".[all]"

# Or install specific modules
pip install -e ".[transfer]"    # GMM-OT alignment
pip install -e ".[correction]"  # GAM-based correction
pip install -e ".[disae2]"      # Deep learning module

# For running example notebooks (includes jupyter, matplotlib, SciencePlots, phate)
pip install -e ".[notebooks]"

# For development (testing, linting, type checking)
pip install -e ".[dev]"
```

**PyTorch with CUDA** (for Dis-AE 2 module):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

---

## Basic Usage

```python
from sysmexcbctools import (
    XNSampleProcessor,      # Data cleaning
    FlowTransformer,        # Flow cytometry alignment
    GAMCorrector,           # Covariate correction
    DisAE                   # Domain-invariant learning
)

# See examples/notebooks/ for complete examples
```

---

## Example Notebooks

The `examples/notebooks/` directory contains Jupyter notebooks demonstrating all modules:

- **Data module** (Notebooks 04-06): Data cleaning, configuration, and advanced features
- **Transfer module** (Notebooks 07-10): Flow cytometry and impedance alignment workflows
- **Correction module** (Notebooks 11-14): GAM-based covariate correction and validation
- **Dis-AE 2 module** (Notebooks 01-03): Domain-invariant learning and multi-domain generalisation

**Note:** Most notebooks require Sysmex analyser data, which is not included in this repository due to patient privacy and ethics considerations.

---

## Documentation

- **Module API documentation:** See individual READMEs in `sysmexcbctools/{data,transfer,correction,disae2}/README.md`
- **Example workflows:** See `examples/notebooks/`
- **Developer guidance:** See `CLAUDE.md` for project structure and development guidelines

---

## Citation

If you use this software in your research, please cite it using the information in `CITATION.cff`. GitHub will automatically generate a citation in various formats from this file.

---

## Legal Disclaimer and Data Sources

This software is an independent, open-source tool developed for research and interoperability purposes. It analyses raw haematology data exported from Sysmex XN analysers, specifically:

- **Scatter (SCT) data**
- **XN_SAMPLE.csv data**
- **OutputData.csv data**

All data formats processed by this tool are directly exportable from any Sysmex XN haematology analyser through standard export functions available to clinical laboratories. This software was developed by analysing legitimately exported data files and does not involve reverse engineering, decompilation, or modification of any Sysmex proprietary software.

**Important Notes:**

- This project is **not affiliated with, endorsed by, or supported by** Sysmex Corporation or any of its subsidiaries
- This tool does **not** replicate, modify, or redistribute any Sysmex proprietary software
- Users are responsible for ensuring their use of this tool complies with their local regulations, institutional policies, and any applicable licence agreements
- The software is provided "as is" without warranty of any kind
- Users should validate all analytical results independently before use in clinical or research contexts

**Intended Use:** This tool is designed to facilitate research analysis of complete blood count data for users who have legitimate access to Sysmex XN analyser exports. It aims to improve interoperability and enable advanced analysis workflows in haematology research.

For questions about data export capabilities or licensing, please contact Sysmex Corporation directly.
