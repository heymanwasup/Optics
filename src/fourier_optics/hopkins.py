from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


@dataclass(frozen=True)
class HopkinsImagingModel:
    """Partially coherent Fourier-optics imaging model.

    The implementation evaluates source points explicitly. Each point shifts the
    objective pupil in frequency space, forms a coherent aerial field, and adds
    the resulting intensity. This is equivalent to the Hopkins formulation for
    the sampled source and is intentionally straightforward for benchmarking.
    """

    wavelength: float
    numerical_aperture: float
    pixel_size: float
    source_sigma: float = 0.5
    source_samples: int = 9

    def __post_init__(self) -> None:
        if self.wavelength <= 0:
            raise ValueError("wavelength must be positive")
        if self.numerical_aperture <= 0:
            raise ValueError("numerical_aperture must be positive")
        if self.pixel_size <= 0:
            raise ValueError("pixel_size must be positive")
        if self.source_sigma < 0:
            raise ValueError("source_sigma must be non-negative")
        if self.source_samples <= 0:
            raise ValueError("source_samples must be positive")

    @property
    def cutoff_frequency(self) -> float:
        return self.numerical_aperture / self.wavelength

    def frequency_grid(self, shape: Tuple[int, int]) -> Tuple[RealArray, RealArray]:
        rows, cols = shape
        fy = np.fft.fftfreq(rows, d=self.pixel_size)
        fx = np.fft.fftfreq(cols, d=self.pixel_size)
        return np.meshgrid(fx, fy)

    def circular_pupil(
        self,
        shape: Tuple[int, int],
        shift_fx: float = 0.0,
        shift_fy: float = 0.0,
    ) -> ComplexArray:
        fx, fy = self.frequency_grid(shape)
        radius = np.hypot(fx - shift_fx, fy - shift_fy)
        return (radius <= self.cutoff_frequency).astype(np.complex128)

    def source_points(self) -> Tuple[RealArray, RealArray, RealArray]:
        if self.source_sigma == 0:
            return (
                np.array([0.0], dtype=np.float64),
                np.array([0.0], dtype=np.float64),
                np.array([1.0], dtype=np.float64),
            )

        axis = np.linspace(-self.source_sigma, self.source_sigma, self.source_samples)
        sx, sy = np.meshgrid(axis, axis)
        mask = sx * sx + sy * sy <= self.source_sigma * self.source_sigma
        sx = sx[mask]
        sy = sy[mask]
        weights = np.ones_like(sx, dtype=np.float64)
        weights /= weights.sum()
        return sx, sy, weights

    def image_intensity(self, object_field: ComplexArray) -> RealArray:
        field = np.asarray(object_field, dtype=np.complex128)
        if field.ndim != 2:
            raise ValueError("object_field must be a 2D complex matrix")

        spectrum = np.fft.fft2(field)
        sx, sy, weights = self.source_points()
        intensity = np.zeros(field.shape, dtype=np.float64)

        for source_x, source_y, weight in zip(sx, sy, weights):
            shift_fx = source_x * self.cutoff_frequency
            shift_fy = source_y * self.cutoff_frequency
            pupil = self.circular_pupil(field.shape, shift_fx=shift_fx, shift_fy=shift_fy)
            image_field = np.fft.ifft2(spectrum * pupil)
            intensity += weight * np.abs(image_field) ** 2

        return intensity
