import numpy as np
import pytest

from fourier_optics import FieldSimulation, HopkinsImagingModel


def test_image_intensity_preserves_shape_and_is_real() -> None:
    field = FieldSimulation(shape=(32, 32), pixel_size=20e-9).checkerboard(period=4).field
    assert field is not None
    model = HopkinsImagingModel(
        wavelength=193e-9,
        numerical_aperture=1.35,
        pixel_size=20e-9,
        source_sigma=0.5,
        source_samples=5,
    )

    image = model.image_intensity(field)

    assert image.shape == field.shape
    assert image.dtype == np.float64
    assert np.all(image >= 0)


def test_rejects_non_2d_input() -> None:
    model = HopkinsImagingModel(
        wavelength=193e-9,
        numerical_aperture=1.35,
        pixel_size=20e-9,
    )

    with pytest.raises(ValueError, match="2D"):
        model.image_intensity(np.ones(8, dtype=np.complex128))
