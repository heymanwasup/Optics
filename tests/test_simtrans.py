import numpy as np
import pytest

from fourier_optics import SimTrans


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
