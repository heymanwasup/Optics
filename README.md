# Fourier Optics

This repository contains a compact Fourier optics imaging model for performance
analysis.

The code is split into two parts:

- `FieldSimulation`: generates object-plane complex electric-field matrices.
- `HopkinsImagingModel`: images those fields using a partially coherent
  Hopkins/Abbe-style source integration model.

## Quick Start

```bash
PYTHONPATH=src /Users/eroica/anaconda3/bin/python -m pytest
PYTHONPATH=src /Users/eroica/anaconda3/bin/python examples/run_hopkins_model.py
PYTHONPATH=src /Users/eroica/anaconda3/bin/python examples/run_line_grating_task.py
```

The EUV line-grating analysis notebook is available at
`notebooks/line_grating_analysis.ipynb`.

## Minimal Example

```python
from fourier_optics import FieldSimulation, HopkinsImagingModel

field = (
    FieldSimulation(shape=(256, 256), pixel_size=20e-9)
    .checkerboard(period=16, amplitude_high=1.0, amplitude_low=0.15)
    .with_phase_ramp(kx=0.3, ky=0.0)
    .field
)

model = HopkinsImagingModel(
    wavelength=193e-9,
    numerical_aperture=1.35,
    pixel_size=20e-9,
    source_sigma=0.7,
)

image = model.image_intensity(field)
```

`image` is a real-valued intensity matrix with the same shape as `field`.
