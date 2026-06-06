import numpy as np
import pytest

from fourier_optics import FieldSimulation


def test_checkerboard_generates_complex_matrix() -> None:
    field = FieldSimulation(shape=(8, 8), pixel_size=1.0).checkerboard(period=2).field

    assert field is not None
    assert field.shape == (8, 8)
    assert np.iscomplexobj(field)
    assert set(np.unique(np.real(field))) == {0.0, 1.0}


def test_amplitude_phase_shape_validation() -> None:
    simulation = FieldSimulation(shape=(8, 8), pixel_size=1.0)

    with pytest.raises(ValueError, match="must match shape"):
        simulation.from_amplitude_phase(np.ones((4, 4)), np.zeros((4, 4)))
