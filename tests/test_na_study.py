import numpy as np
import pytest

from fourier_optics.na_study import (
    cases_from_fdtd_dict,
    collected_na_energy,
    collected_na_energy_from_dict,
    gaussian_width_from_sevd_height,
    make_demo_fdtd_dict,
    make_demo_fdtd_cases,
    plot_fdtd_dict_pupil_and_radial,
    pupil_energy_distribution,
    radial_energy_density,
    sevd,
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


def test_sevd_wrapper_uses_default_gaussian_formula() -> None:
    assert sevd(2.5, 60.0) == pytest.approx(sevd_from_gaussian_height_width(2.5, 60.0))


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


def test_cases_from_fdtd_dict_parses_key_metadata() -> None:
    data_fdtd = {
        "h1w40": np.ones((8, 8), dtype=np.complex128),
        "h2.5w60": np.ones((8, 8), dtype=np.complex128),
    }

    cases = cases_from_fdtd_dict(data_fdtd, ["h2.5w60", "h1w40"])

    assert cases[0].height_nm == pytest.approx(2.5)
    assert cases[0].width_nm == pytest.approx(60.0)
    assert cases[0].sevd_nm is not None
    assert cases[1].title == "h1w40"


def test_collected_na_energy_from_dict_matches_direct_call() -> None:
    data_fdtd = make_demo_fdtd_dict(
        heights_nm=(1.0,),
        widths_nm=(40.0,),
        shape=(64, 64),
        pixel_size_um=0.002,
    )

    from_dict = collected_na_energy_from_dict(
        data_fdtd,
        "h1w40",
        pixel_size_um=0.002,
        wavelength_um=0.0135,
        reference_field=1.0,
    )
    direct = collected_na_energy(
        data_fdtd["h1w40"],
        pixel_size_um=0.002,
        wavelength_um=0.0135,
        reference_field=1.0,
    )

    assert from_dict == pytest.approx(direct)


def test_plot_fdtd_dict_pupil_and_radial_supports_independent_na_ranges() -> None:
    import matplotlib.pyplot as plt

    data_fdtd = make_demo_fdtd_dict(
        heights_nm=(1.0,),
        widths_nm=(40.0, 80.0),
        shape=(64, 64),
        pixel_size_um=0.002,
    )

    fig_maps, fig_radial = plot_fdtd_dict_pupil_and_radial(
        data_fdtd,
        ["h1w40", "h1w80"],
        pixel_size_um=0.002,
        wavelength_um=0.0135,
        reference_field=1.0,
        pupil_min_na=0.05,
        pupil_max_na=0.45,
        radial_min_na=0.10,
        radial_max_na=0.28,
        radial_log_y=True,
    )

    assert fig_maps.axes[0].get_xlim() == pytest.approx((-0.45, 0.45))
    assert fig_radial.axes[0].get_xlim() == pytest.approx((0.10, 0.28))
    assert fig_radial.axes[0].get_yscale() == "log"
    plt.close(fig_maps)
    plt.close(fig_radial)
