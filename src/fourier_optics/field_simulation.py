from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


def pupil_na_grid(
    shape: Tuple[int, int],
    pixelSize: float,
    lambda_: float,
) -> Tuple[RealArray, RealArray, RealArray]:
    """Return shifted pupil-plane NA coordinates for a sampled field."""
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("shape must be a pair of positive integers")
    if pixelSize <= 0:
        raise ValueError("pixelSize must be positive")
    if lambda_ <= 0:
        raise ValueError("lambda_ must be positive")

    rows, cols = shape
    fy = np.fft.fftshift(np.fft.fftfreq(rows, d=pixelSize))
    fx = np.fft.fftshift(np.fft.fftfreq(cols, d=pixelSize))
    na_x, na_y = np.meshgrid(lambda_ * fx, lambda_ * fy)
    return na_x, na_y, np.hypot(na_x, na_y)


def pupil_energy_distribution(mat: ComplexArray, pixelSize: float) -> RealArray:
    """Return shifted pupil-plane energy per sampled frequency pixel."""
    field = np.asarray(mat, dtype=np.complex128)
    if field.ndim != 2:
        raise ValueError("mat must be a 2D complex matrix")
    if pixelSize <= 0:
        raise ValueError("pixelSize must be positive")

    spectrum = np.fft.fftshift(np.fft.fft2(field))
    return (np.abs(spectrum) ** 2 * pixelSize**2 / field.size).astype(np.float64)


def calculate_pupil_energy(
    mat: ComplexArray,
    pixelSize: float,
    NAin: float,
    NAout: float,
    lambda_: float,
) -> float:
    """Calculate total sampled energy transmitted between inner and outer NA."""
    if NAin < 0:
        raise ValueError("NAin must be non-negative")
    if NAout < NAin:
        raise ValueError("NAout must be greater than or equal to NAin")

    energy = pupil_energy_distribution(mat, pixelSize)
    _, _, radius = pupil_na_grid(energy.shape, pixelSize, lambda_)
    mask = (radius >= NAin) & (radius <= NAout)
    return float(energy[mask].sum())


def plot_pupil_energy_distribution(
    mat: ComplexArray,
    pixelSize: float,
    NAin: float,
    NAout: float,
    lambda_: float,
    maxNA: Optional[float] = None,
    ax: Optional[object] = None,
    cmap: str = "magma",
) -> Tuple[object, object]:
    """Plot pupil energy distribution with inner and outer NA circles."""
    if NAin < 0:
        raise ValueError("NAin must be non-negative")
    if NAout < NAin:
        raise ValueError("NAout must be greater than or equal to NAin")

    import matplotlib.pyplot as plt

    energy = pupil_energy_distribution(mat, pixelSize)
    na_x, na_y, radius = pupil_na_grid(energy.shape, pixelSize, lambda_)
    if maxNA is None:
        maxNA = float(np.nanmax(radius))
    if maxNA <= 0:
        raise ValueError("maxNA must be positive")

    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 5.2))

    visible_energy = np.where(radius <= maxNA, energy, np.nan)
    im = ax.imshow(
        visible_energy,
        origin="lower",
        extent=(
            float(na_x.min()),
            float(na_x.max()),
            float(na_y.min()),
            float(na_y.max()),
        ),
        cmap=cmap,
        interpolation="nearest",
    )
    ax.add_patch(plt.Circle((0.0, 0.0), NAin, fill=False, color="#4cc9f0", linewidth=1.8))
    ax.add_patch(plt.Circle((0.0, 0.0), NAout, fill=False, color="#f72585", linewidth=1.8))
    ax.set_xlim(-maxNA, maxNA)
    ax.set_ylim(-maxNA, maxNA)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("NAx")
    ax.set_ylabel("NAy")
    ax.set_title("Pupil-plane energy distribution")
    ax.figure.colorbar(im, ax=ax, label="Energy per pupil sample")
    return ax, im


def radial_pupil_energy_density(
    mat: ComplexArray,
    pixelSize: float,
    lambda_: float,
    bins: Optional[Union[int, RealArray]] = None,
    normalize: bool = True,
) -> Tuple[RealArray, RealArray]:
    """Return rho(r) where rho(r) * dr is energy in the NA interval."""
    energy = pupil_energy_distribution(mat, pixelSize)
    _, _, radius = pupil_na_grid(energy.shape, pixelSize, lambda_)

    if bins is None:
        rows, cols = energy.shape
        dna_y = lambda_ / (rows * pixelSize)
        dna_x = lambda_ / (cols * pixelSize)
        max_radius = float(np.nanmax(radius))
        bin_count = max(1, int(np.ceil(max_radius / min(dna_x, dna_y))))
        bin_edges = np.linspace(0.0, max_radius, bin_count + 1)
    elif np.isscalar(bins):
        bin_edges = np.linspace(0.0, float(np.nanmax(radius)), int(bins) + 1)
    else:
        bin_edges = np.asarray(bins, dtype=np.float64)

    if bin_edges.ndim != 1 or bin_edges.size < 2:
        raise ValueError("bins must define at least two bin edges")
    if np.any(np.diff(bin_edges) <= 0):
        raise ValueError("bin edges must be strictly increasing")

    shell_energy, edges = np.histogram(radius.ravel(), bins=bin_edges, weights=energy.ravel())
    widths = np.diff(edges)
    density = shell_energy / widths
    if normalize:
        total = shell_energy.sum()
        if total > 0:
            density = density / total
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers.astype(np.float64), density.astype(np.float64)


def plot_radial_pupil_energy_density(
    mat: ComplexArray,
    pixelSize: float,
    lambda_: float,
    bins: Optional[Union[int, RealArray]] = None,
    normalize: bool = True,
    ax: Optional[object] = None,
) -> Tuple[object, RealArray, RealArray]:
    """Plot radial pupil energy density as a function of NA radius."""
    import matplotlib.pyplot as plt

    radius, density = radial_pupil_energy_density(
        mat,
        pixelSize,
        lambda_,
        bins=bins,
        normalize=normalize,
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(radius, density, color="#1d3557", linewidth=1.8)
    ax.set_xlabel("r (NA)")
    ax.set_ylabel("Normalized energy density" if normalize else "Energy density")
    ax.set_title("Radial pupil energy density")
    ax.grid(True, alpha=0.3)
    return ax, radius, density


@dataclass
class FieldSimulation:
    """Generate object-plane complex electric fields."""

    shape: Tuple[int, int]
    pixel_size: float
    field: Optional[ComplexArray] = None

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or min(self.shape) <= 0:
            raise ValueError("shape must be a pair of positive integers")
        if self.pixel_size <= 0:
            raise ValueError("pixel_size must be positive")
        if self.field is None:
            self.field = np.ones(self.shape, dtype=np.complex128)
        else:
            self.field = np.asarray(self.field, dtype=np.complex128)
            if self.field.shape != self.shape:
                raise ValueError("field shape must match shape")

    @property
    def coordinates(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        rows, cols = self.shape
        y = (np.arange(rows) - rows / 2) * self.pixel_size
        x = (np.arange(cols) - cols / 2) * self.pixel_size
        return np.meshgrid(x, y)

    def from_amplitude_phase(
        self,
        amplitude: NDArray[np.float64],
        phase: NDArray[np.float64],
    ) -> FieldSimulation:
        amplitude = np.asarray(amplitude, dtype=np.float64)
        phase = np.asarray(phase, dtype=np.float64)
        if amplitude.shape != self.shape or phase.shape != self.shape:
            raise ValueError("amplitude and phase must match shape")
        self.field = amplitude * np.exp(1j * phase)
        return self

    def rectangular_aperture(
        self,
        width: float,
        height: float,
        amplitude_inside: float = 1.0,
        amplitude_outside: float = 0.0,
    ) -> FieldSimulation:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        x, y = self.coordinates
        mask = (np.abs(x) <= width / 2) & (np.abs(y) <= height / 2)
        amplitude = np.where(mask, amplitude_inside, amplitude_outside)
        self.field = amplitude.astype(np.complex128)
        return self

    def circular_aperture(
        self,
        radius: float,
        amplitude_inside: float = 1.0,
        amplitude_outside: float = 0.0,
    ) -> FieldSimulation:
        if radius <= 0:
            raise ValueError("radius must be positive")
        x, y = self.coordinates
        mask = x * x + y * y <= radius * radius
        amplitude = np.where(mask, amplitude_inside, amplitude_outside)
        self.field = amplitude.astype(np.complex128)
        return self

    def checkerboard(
        self,
        period: int,
        amplitude_high: float = 1.0,
        amplitude_low: float = 0.0,
    ) -> FieldSimulation:
        if period <= 0:
            raise ValueError("period must be positive")
        rows, cols = np.indices(self.shape)
        mask = ((rows // period) + (cols // period)) % 2 == 0
        amplitude = np.where(mask, amplitude_high, amplitude_low)
        self.field = amplitude.astype(np.complex128)
        return self

    def with_phase_ramp(self, kx: float, ky: float) -> FieldSimulation:
        x, y = self.coordinates
        assert self.field is not None
        phase = 2 * np.pi * (kx * x / self.pixel_size + ky * y / self.pixel_size)
        self.field = self.field * np.exp(1j * phase)
        return self

    def with_random_phase(
        self,
        seed: Optional[int] = None,
        phase_span: float = 2 * np.pi,
    ) -> FieldSimulation:
        if phase_span < 0:
            raise ValueError("phase_span must be non-negative")
        assert self.field is not None
        rng = np.random.default_rng(seed)
        phase = rng.uniform(-phase_span / 2, phase_span / 2, size=self.shape)
        self.field = self.field * np.exp(1j * phase)
        return self
