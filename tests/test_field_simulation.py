import numpy as np
import pytest

from fourier_optics import (
    FieldSimulation,
    calculate_pupil_energy,
    pupil_energy_distribution,
    radial_pupil_energy_density,
)


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


def test_pupil_energy_distribution_matches_object_energy() -> None:
    field = np.ones((4, 4), dtype=np.complex128)

    pupil_energy = pupil_energy_distribution(field, pixelSize=1.0)

    assert pupil_energy.sum() == pytest.approx(np.sum(np.abs(field) ** 2))
    assert calculate_pupil_energy(field, 1.0, NAin=0.0, NAout=0.1, lambda_=1.0) == pytest.approx(16.0)


def test_pupil_energy_selects_na_annulus() -> None:
    cols = 8
    x = np.arange(cols)
    field = np.tile(np.exp(2j * np.pi * x / cols), (8, 1))

    captured = calculate_pupil_energy(field, 1.0, NAin=0.10, NAout=0.15, lambda_=1.0)

    assert captured == pytest.approx(np.sum(np.abs(field) ** 2))


def test_radial_energy_density_integrates_to_one_when_normalized() -> None:
    field = np.ones((8, 8), dtype=np.complex128)

    radius, density = radial_pupil_energy_density(field, 1.0, lambda_=1.0, bins=8)
    dr = radius[1] - radius[0]

    assert np.sum(density * dr) == pytest.approx(1.0)
