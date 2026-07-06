import numpy as np
import pytest
import tifffile

from fourier_optics import array2d_to_tif


def test_array2d_to_tif_writes_readable_tiff(tmp_path):
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = array2d_to_tif(array, tmp_path / "nested" / "image.tif")

    assert path.exists()
    np.testing.assert_array_equal(tifffile.imread(path), array)


def test_array2d_to_tif_rejects_non_2d_array(tmp_path):
    with pytest.raises(ValueError, match="array must be 2D"):
        array2d_to_tif(np.zeros((2, 3, 4)), tmp_path / "image.tif")


def test_array2d_to_tif_can_refuse_overwrite(tmp_path):
    path = tmp_path / "image.tif"
    array2d_to_tif(np.ones((2, 2)), path)

    with pytest.raises(FileExistsError):
        array2d_to_tif(np.zeros((2, 2)), path, overwrite=False)
