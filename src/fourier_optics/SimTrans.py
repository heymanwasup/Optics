from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
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


def gaussian_defect_matrix(
    fwhm: float,
    pixel_size: float,
    size_x: float,
    size_y: Optional[float] = None,
    amplitude: float = 1.0,
) -> FloatArray:
    """Create a normalized 2D Gaussian defect image.

    Physical units are caller-defined but must be consistent, for example all
    lengths in micrometers.
    """

    if fwhm <= 0:
        raise ValueError("fwhm must be positive")
    if pixel_size <= 0:
        raise ValueError("pixel_size must be positive")
    if size_x <= 0:
        raise ValueError("size_x must be positive")
    if size_y is None:
        size_y = size_x
    if size_y <= 0:
        raise ValueError("size_y must be positive")

    cols = _physical_size_to_pixels(size_x, pixel_size, "size_x")
    rows = _physical_size_to_pixels(size_y, pixel_size, "size_y")
    x = (np.arange(cols, dtype=np.float64) - (cols - 1) / 2) * pixel_size
    y = (np.arange(rows, dtype=np.float64) - (rows - 1) / 2) * pixel_size
    xx, yy = np.meshgrid(x, y)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return np.asarray(amplitude * np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma)), dtype=np.float64)


def gaussian_defect_library(
    fwhm_by_name: Mapping[str, float],
    pixel_size: float,
    defect_size_x: float,
    defect_size_y: Optional[float] = None,
) -> dict[str, FloatArray]:
    """Build a dictionary of Gaussian defect matrices keyed by defect name."""

    return {
        name: gaussian_defect_matrix(
            fwhm=fwhm,
            pixel_size=pixel_size,
            size_x=defect_size_x,
            size_y=defect_size_y,
        )
        for name, fwhm in fwhm_by_name.items()
    }


def photon_defect_array(
    defs: Mapping[str, ArrayLike],
    pixel_size: float,
    photon_def: float,
    photon_bkg: float,
    size_x: float,
    size_y: float,
    column_defects: Union[str, Sequence[str]],
    pitch: float,
    grid_shape: Shape = (10, 10),
    center: Optional[Tuple[float, float]] = None,
    max_pixels: Optional[int] = 100_000_000,
) -> FloatArray:
    """Generate a photon-count image containing a centered defect-point array.

    ``column_defects`` may be a sequence with one defect name per column, or a
    single defect name to fill every site with the same defect type.
    """

    _validate_positive(pixel_size, "pixel_size")
    _validate_positive(size_x, "size_x")
    _validate_positive(size_y, "size_y")
    _validate_positive(pitch, "pitch")
    rows, cols = _validate_grid_shape(grid_shape)

    canvas_cols = _physical_size_to_pixels(size_x, pixel_size, "size_x")
    canvas_rows = _physical_size_to_pixels(size_y, pixel_size, "size_y")
    _validate_pixel_budget(canvas_rows, canvas_cols, max_pixels)

    defect_names = _expand_column_defects(column_defects, cols)
    if center is None:
        center = (size_x / 2, size_y / 2)
    center_x, center_y = center

    canvas = np.full((canvas_rows, canvas_cols), photon_bkg, dtype=np.float64)
    x0 = center_x - (cols - 1) * pitch / 2
    y0 = center_y - (rows - 1) * pitch / 2

    for row_index in range(rows):
        y = y0 + row_index * pitch
        for col_index, defect_name in enumerate(defect_names):
            x = x0 + col_index * pitch
            patch = np.asarray(defs[defect_name], dtype=np.float64) * photon_def
            _add_patch_at_physical_center(canvas, patch, x, y, pixel_size)

    return canvas


def example_photon_array_config(preview_pixel_size: Optional[float] = None) -> dict[str, object]:
    """Return the requested 10-by-10 Gaussian-defect photon-array example."""

    pixel_size = 0.1 if preview_pixel_size is None else preview_pixel_size
    defect_size = 100.0
    defect_names = [f"def{i}" for i in range(1, 11)]
    defs = gaussian_defect_library(
        {name: float(index) for index, name in enumerate(defect_names, start=1)},
        pixel_size=pixel_size,
        defect_size_x=defect_size,
        defect_size_y=defect_size,
    )
    return {
        "defs": defs,
        "pixel_size": pixel_size,
        "photon_def": 1000.0,
        "photon_bkg": 1000.0,
        "size_x": 22_000.0,
        "size_y": 20_480.0,
        "column_defects": defect_names,
        "pitch": 1500.0,
        "grid_shape": (10, 10),
    }


def generate_example_photon_array(
    preview_pixel_size: Optional[float] = None,
    max_pixels: Optional[int] = 100_000_000,
) -> FloatArray:
    """Generate the requested example, optionally using a coarser preview grid."""

    config = example_photon_array_config(preview_pixel_size=preview_pixel_size)
    return photon_defect_array(max_pixels=max_pixels, **config)


def _validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _physical_size_to_pixels(size: float, pixel_size: float, name: str) -> int:
    pixels_float = size / pixel_size
    pixels = int(round(pixels_float))
    if not np.isclose(pixels_float, pixels, rtol=0, atol=1e-9):
        raise ValueError(f"{name} must be an integer multiple of pixel_size")
    if pixels <= 0:
        raise ValueError(f"{name} must contain at least one pixel")
    return pixels


def _validate_grid_shape(grid_shape: Shape) -> Shape:
    if len(grid_shape) != 2 or min(grid_shape) <= 0:
        raise ValueError("grid_shape must be a pair of positive integers")
    return int(grid_shape[0]), int(grid_shape[1])


def _validate_pixel_budget(rows: int, cols: int, max_pixels: Optional[int]) -> None:
    if max_pixels is None:
        return
    pixels = rows * cols
    if pixels > max_pixels:
        raise MemoryError(
            f"requested canvas has {pixels:,} pixels, which exceeds max_pixels={max_pixels:,}"
        )


def _expand_column_defects(column_defects: Union[str, Sequence[str]], columns: int) -> list[str]:
    if isinstance(column_defects, str):
        return [column_defects] * columns
    defect_names = list(column_defects)
    if len(defect_names) != columns:
        raise ValueError("column_defects must contain one defect name per grid column")
    return defect_names


def _add_patch_at_physical_center(
    canvas: FloatArray,
    patch: FloatArray,
    center_x: float,
    center_y: float,
    pixel_size: float,
) -> None:
    patch_rows, patch_cols = patch.shape
    center_col = int(round(center_x / pixel_size))
    center_row = int(round(center_y / pixel_size))
    row_start = center_row - patch_rows // 2
    col_start = center_col - patch_cols // 2
    row_end = row_start + patch_rows
    col_end = col_start + patch_cols

    canvas_row_start = max(row_start, 0)
    canvas_col_start = max(col_start, 0)
    canvas_row_end = min(row_end, canvas.shape[0])
    canvas_col_end = min(col_end, canvas.shape[1])
    if canvas_row_start >= canvas_row_end or canvas_col_start >= canvas_col_end:
        return

    patch_row_start = canvas_row_start - row_start
    patch_col_start = canvas_col_start - col_start
    patch_row_end = patch_row_start + (canvas_row_end - canvas_row_start)
    patch_col_end = patch_col_start + (canvas_col_end - canvas_col_start)
    canvas[canvas_row_start:canvas_row_end, canvas_col_start:canvas_col_end] += patch[
        patch_row_start:patch_row_end, patch_col_start:patch_col_end
    ]
