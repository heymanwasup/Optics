from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


@dataclass(frozen=True)
class FDTDCase:
    """Container for one FDTD complex-amplitude result and its labels."""

    title: str
    field: ComplexArray
    height_nm: Optional[float] = None
    width_nm: Optional[float] = None
    sevd_nm: Optional[float] = None
    metadata: Optional[Mapping[str, object]] = None


@dataclass(frozen=True)
class PupilEnergy:
    """Pupil-plane energy sampled on NA coordinates."""

    energy: RealArray
    na_x: RealArray
    na_y: RealArray
    na_radius: RealArray


def sevd_from_gaussian_height_width(
    height_nm: float,
    width_nm: float,
    width_is_fwhm: bool = True,
) -> float:
    """Convert a circular Gaussian defect volume to equal-volume sphere diameter."""
    _validate_positive(height_nm, "height_nm")
    _validate_positive(width_nm, "width_nm")
    if width_is_fwhm:
        sigma_nm = width_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    else:
        sigma_nm = width_nm
    volume_nm3 = 2.0 * np.pi * height_nm * sigma_nm**2
    return float((6.0 * volume_nm3 / np.pi) ** (1.0 / 3.0))


def gaussian_width_from_sevd_height(
    sevd_nm: float,
    height_nm: float,
    width_is_fwhm: bool = True,
) -> float:
    """Return Gaussian width required for a target SEVD at fixed height."""
    _validate_positive(sevd_nm, "sevd_nm")
    _validate_positive(height_nm, "height_nm")
    volume_nm3 = np.pi * sevd_nm**3 / 6.0
    sigma_nm = np.sqrt(volume_nm3 / (2.0 * np.pi * height_nm))
    if width_is_fwhm:
        return float(2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_nm)
    return float(sigma_nm)


def pupil_na_grid(
    shape: Tuple[int, int],
    pixel_size_um: float,
    wavelength_um: float,
) -> Tuple[RealArray, RealArray, RealArray]:
    """Return shifted NAx, NAy, and radial NA grids for a sampled object field."""
    _validate_shape(shape)
    _validate_positive(pixel_size_um, "pixel_size_um")
    _validate_positive(wavelength_um, "wavelength_um")

    rows, cols = shape
    fy = np.fft.fftshift(np.fft.fftfreq(rows, d=pixel_size_um))
    fx = np.fft.fftshift(np.fft.fftfreq(cols, d=pixel_size_um))
    na_x, na_y = np.meshgrid(wavelength_um * fx, wavelength_um * fy)
    return (
        na_x.astype(np.float64),
        na_y.astype(np.float64),
        np.hypot(na_x, na_y).astype(np.float64),
    )


def pupil_energy_distribution(
    mat: ArrayLike,
    pixel_size_um: float,
    wavelength_um: float,
    reference_field: Optional[Union[complex, ArrayLike]] = None,
) -> PupilEnergy:
    """Return pupil energy and NA coordinates for one FDTD complex field."""
    field = _prepare_field(mat, reference_field)
    _validate_positive(pixel_size_um, "pixel_size_um")
    _validate_positive(wavelength_um, "wavelength_um")

    spectrum = np.fft.fftshift(np.fft.fft2(field))
    energy = (np.abs(spectrum) ** 2 * pixel_size_um**2 / field.size).astype(np.float64)
    na_x, na_y, na_radius = pupil_na_grid(field.shape, pixel_size_um, wavelength_um)
    return PupilEnergy(energy=energy, na_x=na_x, na_y=na_y, na_radius=na_radius)


def collected_na_energy(
    mat: ArrayLike,
    pixel_size_um: float,
    wavelength_um: float,
    na_inner: float = 0.10,
    na_outer: float = 0.28,
    reference_field: Optional[Union[complex, ArrayLike]] = None,
) -> float:
    """Return total pupil energy collected inside ``na_inner <= NA <= na_outer``."""
    _validate_na_range(na_inner, na_outer)
    pupil = pupil_energy_distribution(
        mat,
        pixel_size_um=pixel_size_um,
        wavelength_um=wavelength_um,
        reference_field=reference_field,
    )
    mask = (pupil.na_radius >= na_inner) & (pupil.na_radius <= na_outer)
    return float(pupil.energy[mask].sum())


def radial_energy_density(
    mat: ArrayLike,
    pixel_size_um: float,
    wavelength_um: float,
    bins: Optional[Union[int, ArrayLike]] = None,
    normalize: bool = True,
    reference_field: Optional[Union[complex, ArrayLike]] = None,
) -> Tuple[RealArray, RealArray]:
    """Return ``r`` and ``rho(r)`` where ``rho(r) * dr`` is energy in an NA shell."""
    pupil = pupil_energy_distribution(
        mat,
        pixel_size_um=pixel_size_um,
        wavelength_um=wavelength_um,
        reference_field=reference_field,
    )
    bin_edges = _radial_bin_edges(pupil.energy.shape, pixel_size_um, wavelength_um, bins)
    shell_energy, edges = np.histogram(
        pupil.na_radius.ravel(),
        bins=bin_edges,
        weights=pupil.energy.ravel(),
    )
    widths = np.diff(edges)
    density = shell_energy / widths
    if normalize:
        total = shell_energy.sum()
        if total > 0:
            density = density / total
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers.astype(np.float64), density.astype(np.float64)


def plot_cases_pupil_and_radial(
    cases: Sequence[FDTDCase],
    pixel_size_um: float,
    wavelength_um: float,
    na_inner: float = 0.10,
    na_outer: float = 0.28,
    reference_field: Optional[Union[complex, ArrayLike]] = None,
    max_na: Optional[float] = 0.8,
    bins: Optional[Union[int, ArrayLike]] = None,
    log_floor: float = 1e-14,
    cmap: str = "inferno",
) -> Tuple[object, object]:
    """Plot 2D pupil maps for multiple cases and one absolute radial-density overlay."""
    if len(cases) == 0:
        raise ValueError("cases must contain at least one FDTDCase")
    _validate_na_range(na_inner, na_outer)

    import matplotlib.pyplot as plt

    pupils = [
        pupil_energy_distribution(
            case.field,
            pixel_size_um=pixel_size_um,
            wavelength_um=wavelength_um,
            reference_field=reference_field,
        )
        for case in cases
    ]
    if max_na is None:
        max_na = max(float(np.nanmax(pupil.na_radius)) for pupil in pupils)
    _validate_positive(max_na, "max_na")

    cols = min(3, len(cases))
    rows = int(np.ceil(len(cases) / cols))
    fig_maps, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.9 * rows), squeeze=False)
    map_axes = axes.ravel()
    image = None

    for ax, case, pupil in zip(map_axes, cases, pupils):
        visible = pupil.na_radius <= max_na
        scale = float(np.nanmax(pupil.energy[visible])) if np.any(visible) else float(pupil.energy.max())
        scale = scale if scale > 0 else 1.0
        plot_data = np.full_like(pupil.energy, np.nan, dtype=np.float64)
        plot_data[visible] = np.log10(pupil.energy[visible] / scale + log_floor)
        image = ax.imshow(
            plot_data,
            origin="lower",
            extent=(
                float(pupil.na_x.min()),
                float(pupil.na_x.max()),
                float(pupil.na_y.min()),
                float(pupil.na_y.max()),
            ),
            cmap=cmap,
            vmin=np.log10(log_floor),
            vmax=0.0,
            interpolation="nearest",
        )
        ax.add_patch(plt.Circle((0.0, 0.0), na_inner, fill=False, color="#4cc9f0", linewidth=1.4))
        ax.add_patch(plt.Circle((0.0, 0.0), na_outer, fill=False, color="#f72585", linewidth=1.4))
        ax.set_xlim(-max_na, max_na)
        ax.set_ylim(-max_na, max_na)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(case.title)
        ax.set_xlabel("NAx")
        ax.set_ylabel("NAy")

    for ax in map_axes[len(cases) :]:
        ax.axis("off")
    if image is not None:
        fig_maps.colorbar(image, ax=map_axes[: len(cases)], shrink=0.86, label="log10 normalized pupil energy")
    fig_maps.suptitle("Pupil-plane energy distribution", y=1.02)

    fig_radial, ax = plt.subplots(figsize=(7.2, 4.6))
    for case in cases:
        r_na, rho = radial_energy_density(
            case.field,
            pixel_size_um=pixel_size_um,
            wavelength_um=wavelength_um,
            bins=bins,
            normalize=False,
            reference_field=reference_field,
        )
        ax.plot(r_na, rho, linewidth=1.9, label=case.title)
    ax.axvspan(na_inner, na_outer, color="0.82", alpha=0.45, label=f"PO NA {na_inner:.2f}-{na_outer:.2f}")
    ax.set_xlim(0.0, max_na)
    ax.set_xlabel("r (NA)")
    ax.set_ylabel("absolute radial energy density")
    ax.set_title("Absolute radial energy density comparison")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return fig_maps, fig_radial


def plot_sevd_energy_trends(
    cases: Sequence[FDTDCase],
    pixel_size_um: float,
    wavelength_um: float,
    target_sevd_nm: Optional[float] = None,
    na_inner: float = 0.10,
    na_outer: float = 0.28,
    reference_field: Optional[Union[complex, ArrayLike]] = None,
) -> Tuple[object, RealArray]:
    """Plot collected PO energy versus SEVD and compare heights at equal SEVD."""
    if len(cases) == 0:
        raise ValueError("cases must contain at least one FDTDCase")
    _validate_na_range(na_inner, na_outer)

    import matplotlib.pyplot as plt

    rows = []
    for case in cases:
        if case.sevd_nm is None:
            if case.height_nm is None or case.width_nm is None:
                raise ValueError("each case needs sevd_nm or both height_nm and width_nm")
            sevd_nm = sevd_from_gaussian_height_width(case.height_nm, case.width_nm)
        else:
            sevd_nm = float(case.sevd_nm)
        if case.height_nm is None:
            raise ValueError("each case needs height_nm for grouped trend plots")
        energy = collected_na_energy(
            case.field,
            pixel_size_um=pixel_size_um,
            wavelength_um=wavelength_um,
            na_inner=na_inner,
            na_outer=na_outer,
            reference_field=reference_field,
        )
        rows.append((float(case.height_nm), float(case.width_nm or np.nan), sevd_nm, energy))

    result = np.asarray(rows, dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7), constrained_layout=True)

    for height_nm in sorted(set(result[:, 0])):
        subset = result[result[:, 0] == height_nm]
        subset = subset[np.argsort(subset[:, 2])]
        axes[0].plot(subset[:, 2], subset[:, 3], marker="o", linewidth=1.9, label=f"h={height_nm:g} nm")
    axes[0].set_xlabel("SEVD (nm)")
    axes[0].set_ylabel(f"collected energy, NA {na_inner:.2f}-{na_outer:.2f}")
    axes[0].set_title("Collected energy increases with SEVD")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(title="height")

    if target_sevd_nm is None:
        target_sevd_nm = float(np.nanmedian(result[:, 2]))
    comparison = []
    for height_nm in sorted(set(result[:, 0])):
        subset = result[result[:, 0] == height_nm]
        subset = subset[np.argsort(subset[:, 2])]
        if subset.shape[0] < 2 or target_sevd_nm < subset[:, 2].min() or target_sevd_nm > subset[:, 2].max():
            comparison.append((height_nm, np.nan))
        else:
            comparison.append((height_nm, float(np.interp(target_sevd_nm, subset[:, 2], subset[:, 3]))))
    comp = np.asarray(comparison, dtype=np.float64)
    axes[1].bar([f"h={h:g}" for h in comp[:, 0]], comp[:, 1], color="#4c78a8")
    axes[1].set_ylabel(f"interpolated collected energy at SEVD={target_sevd_nm:.1f} nm")
    axes[1].set_title("Same-SEVD height comparison")
    axes[1].grid(True, axis="y", alpha=0.3)

    return fig, result


def make_demo_fdtd_cases(
    heights_nm: Sequence[float] = (1.0, 2.5, 4.0),
    widths_nm: Sequence[float] = (40.0, 60.0, 80.0, 110.0),
    shape: Tuple[int, int] = (512, 512),
    pixel_size_um: float = 0.001,
    wavelength_um: float = 0.0135,
    phase_scale: float = 4.0 * np.pi,
    profile: str = "gaussian",
    carrier_na: Optional[float] = 0.18,
    scatter_scale_per_nm: float = 0.03,
) -> list[FDTDCase]:
    """Generate deterministic fake FDTD complex fields for notebook demos."""
    _validate_shape(shape)
    _validate_positive(pixel_size_um, "pixel_size_um")
    _validate_positive(wavelength_um, "wavelength_um")

    rows, cols = shape
    pixel_size_nm = pixel_size_um * 1000.0
    y_nm = (np.arange(rows, dtype=np.float64) - rows / 2) * pixel_size_nm
    x_nm = (np.arange(cols, dtype=np.float64) - cols / 2) * pixel_size_nm
    xx_nm, yy_nm = np.meshgrid(x_nm, y_nm)
    radius2_nm = xx_nm * xx_nm + yy_nm * yy_nm

    cases: list[FDTDCase] = []
    for height_nm in heights_nm:
        for width_nm in widths_nm:
            if profile == "gaussian":
                sigma_nm = width_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
                height_map_nm = float(height_nm) * np.exp(-radius2_nm / (2.0 * sigma_nm**2))
            elif profile == "super_gaussian":
                order = 8.0
                sigma_nm = (width_nm / 2.0) / (2.0 * np.log(2.0)) ** (1.0 / order)
                height_map_nm = float(height_nm) * np.exp(-0.5 * (np.sqrt(radius2_nm) / sigma_nm) ** order)
            else:
                raise ValueError("profile must be 'gaussian' or 'super_gaussian'")

            if carrier_na is None:
                phase = phase_scale * height_map_nm / (wavelength_um * 1000.0)
                field = np.exp(1j * phase).astype(np.complex128)
            else:
                carrier_fx = carrier_na / wavelength_um
                carrier_phase = 2.0 * np.pi * carrier_fx * (xx_nm / 1000.0)
                scattered = scatter_scale_per_nm * height_map_nm * np.exp(1j * carrier_phase)
                field = (1.0 + scattered).astype(np.complex128)
            sevd_nm = sevd_from_gaussian_height_width(float(height_nm), float(width_nm))
            cases.append(
                FDTDCase(
                    title=f"h={height_nm:g} nm, w={width_nm:g} nm",
                    field=field,
                    height_nm=float(height_nm),
                    width_nm=float(width_nm),
                    sevd_nm=sevd_nm,
                    metadata={"profile": profile},
                )
            )
    return cases


def _prepare_field(mat: ArrayLike, reference_field: Optional[Union[complex, ArrayLike]]) -> ComplexArray:
    field = np.asarray(mat, dtype=np.complex128)
    if field.ndim != 2:
        raise ValueError("mat must be a 2D complex matrix")
    if reference_field is not None:
        reference = np.asarray(reference_field, dtype=np.complex128)
        field = field - reference
    return np.asarray(field, dtype=np.complex128)


def _radial_bin_edges(
    shape: Tuple[int, int],
    pixel_size_um: float,
    wavelength_um: float,
    bins: Optional[Union[int, ArrayLike]],
) -> RealArray:
    _validate_shape(shape)
    if bins is None:
        rows, cols = shape
        dna_y = wavelength_um / (rows * pixel_size_um)
        dna_x = wavelength_um / (cols * pixel_size_um)
        _, _, radius = pupil_na_grid(shape, pixel_size_um, wavelength_um)
        max_radius = float(np.nanmax(radius))
        bin_count = max(1, int(np.ceil(max_radius / min(dna_x, dna_y))))
        edges = np.linspace(0.0, max_radius, bin_count + 1)
    elif np.isscalar(bins):
        bin_count = int(bins)
        if bin_count <= 0:
            raise ValueError("bins must be positive")
        _, _, radius = pupil_na_grid(shape, pixel_size_um, wavelength_um)
        edges = np.linspace(0.0, float(np.nanmax(radius)), bin_count + 1)
    else:
        edges = np.asarray(bins, dtype=np.float64)

    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bins must define at least two bin edges")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("bin edges must be strictly increasing")
    return edges.astype(np.float64)


def _validate_shape(shape: Tuple[int, int]) -> None:
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("shape must be a pair of positive integers")


def _validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_na_range(na_inner: float, na_outer: float) -> None:
    if na_inner < 0:
        raise ValueError("na_inner must be non-negative")
    if na_outer < na_inner:
        raise ValueError("na_outer must be greater than or equal to na_inner")
