from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


ComplexArray = NDArray[np.complex128]


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
