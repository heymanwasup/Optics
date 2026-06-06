from __future__ import annotations

import numpy as np

from fourier_optics import FieldSimulation, HopkinsImagingModel


def main() -> None:
    field = (
        FieldSimulation(shape=(256, 256), pixel_size=20e-9)
        .checkerboard(period=16, amplitude_high=1.0, amplitude_low=0.1)
        .with_phase_ramp(kx=0.05, ky=0.02)
        .field
    )
    assert field is not None

    model = HopkinsImagingModel(
        wavelength=193e-9,
        numerical_aperture=1.35,
        pixel_size=20e-9,
        source_sigma=0.7,
        source_samples=11,
    )
    image = model.image_intensity(field)

    print(f"field shape: {field.shape}, dtype: {field.dtype}")
    print(f"image shape: {image.shape}, dtype: {image.dtype}")
    print(
        "image statistics: "
        f"min={np.min(image):.6g}, max={np.max(image):.6g}, mean={np.mean(image):.6g}"
    )


if __name__ == "__main__":
    main()
