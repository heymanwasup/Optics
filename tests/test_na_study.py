import numpy as np
import pytest

from fourier_optics.na_study import (
    collected_na_energy,
    gaussian_width_from_sevd_height,
    make_demo_fdtd_cases,
    pupil_energy_distribution,
    radial_energy_density,
    sevd_from_gaussian_height_width,
)


def test_pupil_energy_matches_object_energy() -> None:
    field = np.ones((8, 8), dtype=np.complex128)

    pupil = pupil_energy_distribution(field, pixel_size_um=1.0, wavelength_um=1.0)

    assert pupil.energy.sum() == pytest.approx(np.sum(np.abs(field) ** 2))


def test_collected_na_energy_selects_known_frequency() -> None:
    cols = 16
    x = np.arange(cols)
    field = np.tile(np.exp(2j * np.pi * 2 * x / cols), (16, 1))

    energy = collected_na_energy(
        field,
        pixel_size_um=1.0,
        wavelength_um=1.0,
        na_inner=0.12,
        na_outer=0.13,
    )

    assert energy == pytest.approx(np.sum(np.abs(field) ** 2))


def test_radial_energy_density_normalizes_to_one() -> None:
    field = np.ones((16, 16), dtype=np.complex128)

    r_na, rho = radial_energy_density(field, 1.0, 1.0, bins=16, normalize=True)
    dr = r_na[1] - r_na[0]

    assert np.sum(rho * dr) == pytest.approx(1.0)


def test_sevd_width_round_trip() -> None:
    sevd = sevd_from_gaussian_height_width(2.5, 60.0)
    width = gaussian_width_from_sevd_height(sevd, 2.5)

    assert width == pytest.approx(60.0)


def test_make_demo_fdtd_cases_has_metadata() -> None:
    cases = make_demo_fdtd_cases(
        heights_nm=(1.0,),
        widths_nm=(40.0, 80.0),
        shape=(64, 64),
        pixel_size_um=0.002,
    )

    assert len(cases) == 2
    assert cases[0].field.shape == (64, 64)
    assert cases[0].sevd_nm is not None
    assert np.iscomplexobj(cases[0].field)
