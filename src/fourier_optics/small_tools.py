from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import tifffile


PathLike = Union[str, Path]
DTypeLike = Optional[Union[str, np.dtype]]


def _convert_for_tif(image: np.ndarray, dtype: DTypeLike) -> np.ndarray:
    if dtype is None:
        return image

    target_dtype = np.dtype(dtype)
    if target_dtype.kind not in "uiuf":
        raise ValueError("dtype must be an integer or floating-point dtype")
    if image.dtype == target_dtype:
        return image

    if target_dtype.kind in "ui":
        info = np.iinfo(target_dtype)
        finite = np.isfinite(image)
        if not np.any(finite):
            return np.zeros(image.shape, dtype=target_dtype)

        finite_values = image[finite].astype(np.float64, copy=False)
        min_value = finite_values.min()
        max_value = finite_values.max()
        if min_value == max_value:
            return np.zeros(image.shape, dtype=target_dtype)

        scaled = (image.astype(np.float64, copy=False) - min_value) / (max_value - min_value)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
        scaled = np.clip(scaled, 0.0, 1.0)
        return np.rint(scaled * info.max).astype(target_dtype)

    return image.astype(target_dtype)


def array2d_to_tif(
    array: Any,
    output_path: PathLike,
    *,
    overwrite: bool = True,
    dtype: DTypeLike = "uint8",
) -> Path:
    """Save a 2D NumPy-compatible array as a grayscale TIFF file.

    By default the data is normalized to uint8, which produces a baseline TIFF
    that opens reliably in ImageJ and common OS image viewers. Pass dtype=None
    to preserve the input dtype for scientific data exchange.

    Parameters
    ----------
    array:
        Two-dimensional array-like data to write.
    output_path:
        Target TIFF path. Parent directories are created automatically.
    overwrite:
        If False, raise FileExistsError when the target file already exists.
    dtype:
        Output dtype. Integer dtypes are min-max normalized; dtype=None preserves
        the original array dtype.

    Returns
    -------
    pathlib.Path
        The written TIFF path.
    """
    image = np.asarray(array)
    if image.ndim != 2:
        raise ValueError(f"array must be 2D, got shape {image.shape}")

    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        _convert_for_tif(image, dtype),
        photometric="minisblack",
        compression=None,
    )
    return path
