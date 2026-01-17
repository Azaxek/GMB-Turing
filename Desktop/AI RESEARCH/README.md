# Glioblastoma Turing Pattern Model

A reaction-diffusion model predicting the spatial distribution of multifocal glioblastoma lesions.

## Overview

This repository contains the code, data, and manuscript for our study demonstrating that multifocal glioblastoma follows predictable spatial patterns governed by Turing-type reaction-diffusion dynamics.

**Key Finding:** Satellite glioblastoma lesions appear at a characteristic spacing of ~2.8 cm, matching predictions from a VEGF-A/sFLT-1 activator-inhibitor model.

## Repository Structure

```
├── manuscript.tex          # Main LaTeX manuscript
├── references.bib          # Bibliography (50+ citations)
├── sn-jnl.cls             # Springer Nature template
├── figures/               # Generated figures (PNG, 600 DPI)
├── figures_tiff/          # Publication-ready TIFF figures
├── simulate_gbm.py        # Reaction-diffusion simulation
├── generate_extended_data.py  # Extended data figure generation
├── Supplementary_Information.tex  # SI document
└── sections/              # Modular LaTeX sections
```

## Running the Simulations

```bash
# Generate main figures
python simulate_gbm.py

# Generate extended data figures
python generate_extended_data.py

# Convert to TIFF for publication
python convert_to_tiff.py
```

## Compiling the Manuscript

```bash
pdflatex manuscript.tex
bibtex manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex
```

## Requirements

- Python 3.8+
- NumPy, Matplotlib, SciPy
- LaTeX distribution (MiKTeX or TeX Live)

## Citation

If you use this code or data, please cite:

```bibtex
@article{GBM_Turing_2026,
  title={Reaction-Diffusion Dynamics Explain the Characteristic Spacing of Multifocal Glioblastoma Lesions},
  author={[Authors]},
  journal={Nature Cancer},
  year={2026}
}
```

## License

MIT License
