import numpy as np
import pytest

from fourier_optics import (
    SimTrans,
    gaussian_defect_matrix,
    generate_example_photon_array,
    photon_defect_array,
)


def test_simtrans_expands_scalars_to_complex_grid() -> None:
    transmission = SimTrans(amplitude=0.5, phase=np.pi / 2, shape=(3, 4))

    assert transmission.shape == (3, 4)
    assert np.iscomplexobj(transmission)
    np.testing.assert_allclose(transmission, 0.5j)


def test_simtrans_aperture_masks_outside_region() -> None:
    aperture = np.array([[True, False], [False, True]])

    transmission = SimTrans(amplitude=1.0, phase=0.0, aperture=aperture)

    np.testing.assert_allclose(transmission, np.array([[1.0, 0.0], [0.0, 1.0]]))


def test_simtrans_aperture_shape_validation() -> None:
    with pytest.raises(ValueError, match="aperture shape"):
        SimTrans(shape=(2, 2), aperture=np.ones((3, 3)))


def test_photon_defect_array_places_defects_by_column() -> None:
    defs = {
        "def1": np.ones((2, 2)),
        "def2": np.full((2, 2), 2.0),
    }

    image = photon_defect_array(
        defs=defs,
        pixel_size=1.0,
        photon_def=10.0,
        photon_bkg=100.0,
        size_x=10.0,
        size_y=10.0,
        column_defects=["def1", "def2"],
        pitch=4.0,
        grid_shape=(2, 2),
    )

    assert image.shape == (10, 10)
    assert image[3, 3] == 110.0
    assert image[3, 7] == 120.0
    assert image[7, 3] == 110.0
    assert image[7, 7] == 120.0


def test_photon_defect_array_supports_single_defect_everywhere() -> None:
    image = photon_defect_array(
        defs={"same": np.ones((1, 1))},
        pixel_size=1.0,
        photon_def=5.0,
        photon_bkg=10.0,
        size_x=8.0,
        size_y=8.0,
        column_defects="same",
        pitch=2.0,
        grid_shape=(2, 3),
    )

    assert np.count_nonzero(image == 15.0) == 6


def test_gaussian_defect_matrix_peaks_at_amplitude() -> None:
    defect = gaussian_defect_matrix(fwhm=2.0, pixel_size=1.0, size_x=5.0)

    assert defect.shape == (5, 5)
    assert defect[2, 2] == pytest.approx(1.0)


def test_example_full_resolution_is_guarded_by_pixel_budget() -> None:
    with pytest.raises(MemoryError, match="exceeds max_pixels"):
        generate_example_photon_array()
