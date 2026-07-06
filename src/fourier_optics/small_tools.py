from pathlib import Path
from typing import Any, Union

import numpy as np
import tifffile


PathLike = Union[str, Path]


def array2d_to_tif(array: Any, output_path: PathLike, *, overwrite: bool = True) -> Path:
    """Save a 2D NumPy-compatible array as a TIFF file.

    Parameters
    ----------
    array:
        Two-dimensional array-like data to write.
    output_path:
        Target TIFF path. Parent directories are created automatically.
    overwrite:
        If False, raise FileExistsError when the target file already exists.

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
    tifffile.imwrite(path, image)
    return path
