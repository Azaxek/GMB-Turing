# GBM Turing Model

Reaction-diffusion simulation predicting spatial distribution of multifocal glioblastoma lesions.

## Overview

This code implements a Turing-type reaction-diffusion model for glioblastoma pattern formation, based on VEGF-A/sFLT-1 activator-inhibitor dynamics.

**Key Finding:** The model predicts satellite lesions at ~2.8 cm spacing, matching clinical observations.

## Files

- `simulate_gbm.py` - Main simulation generating Figures 1-4
- `generate_extended_data.py` - Extended data figures and graphical abstract
- `convert_to_tiff.py` - Convert figures to publication-ready TIFF format

## Usage

```bash
# Run main simulation
python simulate_gbm.py

# Generate extended figures
python generate_extended_data.py
```

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- SciPy
- tqdm
- Pillow

## License

MIT
