from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


ComplexArray = NDArray[np.complex128]
Shape = Tuple[int, int]


def SimTrans(
    amplitude: ArrayLike = 1.0,
    phase: ArrayLike = 0.0,
    shape: Optional[Shape] = None,
    aperture: Optional[ArrayLike] = None,
    outside_amplitude: Union[float, complex] = 0.0,
) -> ComplexArray:
    """Create a complex optical transmission function.

    The returned field is ``amplitude * exp(1j * phase)``. Inputs may be
    scalars or array-like values; if ``shape`` is provided, scalar inputs are
    expanded to that grid.
    """

    aperture_mask = None
    if aperture is not None:
        aperture_mask = np.asarray(aperture, dtype=bool)
        if shape is None:
            shape = aperture_mask.shape

    if shape is not None and (len(shape) != 2 or min(shape) <= 0):
        raise ValueError("shape must be a pair of positive integers")

    amplitude_array = np.asarray(amplitude, dtype=np.float64)
    phase_array = np.asarray(phase, dtype=np.float64)

    if shape is not None:
        amplitude_array = np.broadcast_to(amplitude_array, shape)
        phase_array = np.broadcast_to(phase_array, shape)
    else:
        amplitude_array, phase_array = np.broadcast_arrays(amplitude_array, phase_array)

    transmission = amplitude_array * np.exp(1j * phase_array)

    if aperture_mask is not None:
        if aperture_mask.shape != transmission.shape:
            raise ValueError("aperture shape must match transmission shape")
        outside = complex(outside_amplitude)
        transmission = np.where(aperture_mask, transmission, outside)

    return np.asarray(transmission, dtype=np.complex128)
