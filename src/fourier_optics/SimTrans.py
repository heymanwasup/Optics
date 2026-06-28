from __future__ import annotations

from pathlib import Path
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
    poisson_sample: bool = False,
    rng_seed: Optional[int] = None,
    edge_taper_width: float = 0.0,
) -> FloatArray:
    """Generate a photon-count image containing a centered defect-point array.

    ``column_defects`` may be a sequence with one defect name per column, or a
    single defect name to fill every site with the same defect type. Set
    ``poisson_sample`` to sample the final photon-count expectation image.
    ``edge_taper_width`` applies a raised-cosine taper to each defect patch so
    the embedded simulation window blends smoothly into the background.
    """

    _validate_positive(pixel_size, "pixel_size")
    _validate_positive(size_x, "size_x")
    _validate_positive(size_y, "size_y")
    _validate_positive(pitch, "pitch")
    _validate_nonnegative(photon_def, "photon_def")
    _validate_nonnegative(photon_bkg, "photon_bkg")
    _validate_nonnegative(edge_taper_width, "edge_taper_width")
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
            patch = _apply_edge_taper(patch, edge_taper_width, pixel_size)
            _add_patch_at_physical_center(canvas, patch, x, y, pixel_size)

    if poisson_sample:
        rng = np.random.default_rng(rng_seed)
        canvas = rng.poisson(canvas).astype(np.float64)

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
        "edge_taper_width": 15.0,
    }


def generate_example_photon_array(
    preview_pixel_size: Optional[float] = None,
    max_pixels: Optional[int] = 100_000_000,
    poisson_sample: bool = False,
    rng_seed: Optional[int] = None,
) -> FloatArray:
    """Generate the requested example, optionally using a coarser preview grid."""

    config = example_photon_array_config(preview_pixel_size=preview_pixel_size)
    return photon_defect_array(
        max_pixels=max_pixels,
        poisson_sample=poisson_sample,
        rng_seed=rng_seed,
        **config,
    )


def photon_defect_demo_params() -> dict[str, object]:
    """Return compact, visually clear demo parameters."""

    defect_names = [f"def{i}" for i in range(1, 11)]
    return {
        "pixel_size": 2.0,
        "size_x": 2200.0,
        "size_y": 2048.0,
        "grid_shape": (10, 10),
        "pitch": 160.0,
        "defect_size_x": 120.0,
        "defect_size_y": 120.0,
        "photon_def": 12000.0,
        "photon_bkg": 20.0,
        "poisson_sample": True,
        "rng_seed": 7,
        "edge_taper_width": 44.0,
        "column_defects": defect_names,
        "fwhm_by_name": {
            name: fwhm for name, fwhm in zip(defect_names, np.linspace(70.0, 8.0, 10))
        },
        "max_pixels": 2_000_000,
    }


def build_photon_defect_demo(params: Mapping[str, object]) -> dict[str, object]:
    """Build defect templates and the photon image from a parameter dictionary."""

    defs = gaussian_defect_library(
        params["fwhm_by_name"],  # type: ignore[arg-type]
        pixel_size=float(params["pixel_size"]),
        defect_size_x=float(params["defect_size_x"]),
        defect_size_y=float(params["defect_size_y"]),
    )
    photon_image = photon_defect_array(
        defs=defs,
        pixel_size=float(params["pixel_size"]),
        photon_def=float(params["photon_def"]),
        photon_bkg=float(params["photon_bkg"]),
        size_x=float(params["size_x"]),
        size_y=float(params["size_y"]),
        column_defects=params["column_defects"],  # type: ignore[arg-type]
        pitch=float(params["pitch"]),
        grid_shape=params["grid_shape"],  # type: ignore[arg-type]
        max_pixels=int(params["max_pixels"]),
        poisson_sample=bool(params["poisson_sample"]),
        rng_seed=int(params["rng_seed"]),
        edge_taper_width=float(params.get("edge_taper_width", 0.0)),
    )
    return {"defs": defs, "photon_image": photon_image, "params": dict(params)}


def save_photon_defect_demo_figures(
    photon_image: FloatArray,
    params: Mapping[str, object],
    output_dir: Union[str, Path],
    prefix: str = "photon_defect_array_demo",
) -> dict[str, Path]:
    """Save gray full-view, zoom-view, and profile figures for the demo."""

    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    paths = {
        "full": output_path / f"{prefix}_full_gray.png",
        "zoom": output_path / f"{prefix}_zoom_gray.png",
        "profile": output_path / f"{prefix}_profile.png",
    }

    pixel_size = float(params["pixel_size"])
    size_x = float(params["size_x"])
    size_y = float(params["size_y"])
    photon_bkg = float(params["photon_bkg"])
    photon_def = float(params["photon_def"])
    grid_shape = params["grid_shape"]  # type: ignore[assignment]
    pitch = float(params["pitch"])
    defect_size_x = float(params["defect_size_x"])
    defect_size_y = float(params["defect_size_y"])
    column_defects = list(params["column_defects"])  # type: ignore[arg-type]
    fwhm_by_name = params["fwhm_by_name"]  # type: ignore[assignment]

    extent = [0.0, size_x, 0.0, size_y]
    vmin = max(0.0, photon_bkg - 5.0)
    vmax = photon_bkg + photon_def
    zoom_extent = _defect_array_zoom_extent(
        size_x=size_x,
        size_y=size_y,
        grid_shape=grid_shape,  # type: ignore[arg-type]
        pitch=pitch,
        defect_size_x=defect_size_x,
        defect_size_y=defect_size_y,
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(
        photon_image,
        cmap="gray",
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title("Dark-field photon image with Poisson sampling")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    fig.colorbar(im, ax=ax, label="photons")
    fig.tight_layout()
    fig.savefig(paths["full"], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        photon_image,
        cmap="gray",
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlim(zoom_extent[0], zoom_extent[1])
    ax.set_ylim(zoom_extent[2], zoom_extent[3])
    ax.set_title("Zoom: FWHM decreases from left to right")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    cols = int(grid_shape[1])  # type: ignore[index]
    center_x = size_x / 2
    x_positions = center_x + (np.arange(cols) - (cols - 1) / 2) * pitch
    y_label = zoom_extent[3] - 35
    for name, x in zip(column_defects, x_positions):
        ax.text(
            x,
            y_label,
            f"{name}\n{fwhm_by_name[name]:.0f} um",
            color="white",
            ha="center",
            va="top",
            fontsize=7,
        )
    fig.colorbar(im, ax=ax, label="photons")
    fig.tight_layout()
    fig.savefig(paths["zoom"], dpi=180)
    plt.close(fig)

    center_row = photon_image.shape[0] // 2
    x_um = np.arange(photon_image.shape[1]) * pixel_size
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(x_um, photon_image[center_row], color="black", linewidth=1.2)
    ax.set_xlim(zoom_extent[0], zoom_extent[1])
    ax.set_xlabel("x (um)")
    ax.set_ylabel("photons")
    ax.set_title("Central row profile with Poisson noise")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths["profile"], dpi=180)
    plt.close(fig)

    return paths


def summarize_photon_image(photon_image: FloatArray, params: Mapping[str, object]) -> str:
    """Return a compact text summary for notebooks and logs."""

    return (
        f"photon_image shape: {photon_image.shape}\n"
        f"memory: {photon_image.nbytes / 1024**2:.2f} MiB\n"
        f"min / max photons: {photon_image.min():.1f} / {photon_image.max():.1f}\n"
        f"poisson_sample: {params['poisson_sample']}\n"
        f"edge_taper_width: {params.get('edge_taper_width', 0.0)} um"
    )


def _validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_nonnegative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _apply_edge_taper(patch: FloatArray, edge_taper_width: float, pixel_size: float) -> FloatArray:
    if edge_taper_width <= 0:
        return patch
    width_pixels = edge_taper_width / pixel_size
    if width_pixels <= 0:
        return patch
    weights = _raised_cosine_edge_window(patch.shape, width_pixels)
    return patch * weights


def _raised_cosine_edge_window(shape: Shape, width_pixels: float) -> FloatArray:
    rows, cols = shape
    y_distance = np.minimum(np.arange(rows), np.arange(rows)[::-1])
    x_distance = np.minimum(np.arange(cols), np.arange(cols)[::-1])
    y_weight = np.sin(0.5 * np.pi * np.clip(y_distance / width_pixels, 0.0, 1.0))
    x_weight = np.sin(0.5 * np.pi * np.clip(x_distance / width_pixels, 0.0, 1.0))
    return np.outer(y_weight, x_weight).astype(np.float64)


def _defect_array_zoom_extent(
    size_x: float,
    size_y: float,
    grid_shape: Shape,
    pitch: float,
    defect_size_x: float,
    defect_size_y: float,
) -> list[float]:
    rows, cols = grid_shape
    center_x = size_x / 2
    center_y = size_y / 2
    span_x = (cols - 1) * pitch + defect_size_x + 140.0
    span_y = (rows - 1) * pitch + defect_size_y + 140.0
    return [
        center_x - span_x / 2,
        center_x + span_x / 2,
        center_y - span_y / 2,
        center_y + span_y / 2,
    ]


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
