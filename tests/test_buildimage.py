import ast
from pathlib import Path

import numpy as np

from fourier_optics import PadingDefectsMatrix, build_def_lib, build_picture


def test_build_picture_uses_ten_rows_and_def_count_columns_by_default() -> None:
    defs = {
        "def1": np.ones((5, 5)) * 10.0,
        "def2": np.ones((5, 5)) * 20.0,
        "def3": np.ones((5, 5)) * 30.0,
    }

    image = build_picture(
        Defs_dict=defs,
        bg_photon_num=2.0,
        image_shape=(120, 140),
        def_matrix_step=10,
    )

    assert image.shape == (120, 140)
    assert image.min() == 2.0
    assert np.count_nonzero(image > 2.0) > 10 * len(defs)


def test_build_picture_smooths_patch_edges() -> None:
    image = build_picture(
        Defs_dict={"def1": np.ones((7, 7)) * 10.0},
        bg_photon_num=1.0,
        image_shape=(21, 21),
        def_matrix_shape=(1, 1),
        def_matrix_step=8,
    )

    assert image[7, 7] == 1.0
    assert image[10, 10] == 11.0
    assert 1.0 < image[9, 9] < 11.0


def test_build_def_lib_returns_shrinking_bright_defects() -> None:
    defs = build_def_lib(def_count=4, matrix_shape=(21, 21), fwhm_range=(10.0, 4.0))

    assert list(defs) == ["def1", "def2", "def3", "def4"]
    assert defs["def1"].shape == (21, 21)
    assert np.count_nonzero(defs["def1"] > 1000.0) > np.count_nonzero(defs["def4"] > 1000.0)


def test_buildimage_module_public_shape() -> None:
    module_path = Path(__file__).parents[1] / "src" / "fourier_optics" / "buildimage.py"
    tree = ast.parse(module_path.read_text())

    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]

    assert functions == ["build_def_lib", "build_picture", "plot_picture"]
    assert classes == ["PadingDefectsMatrix"]


def test_pading_defects_matrix_validates_column_count() -> None:
    try:
        PadingDefectsMatrix({"def1": np.ones((3, 3))}, 1.0, (20, 20), (10, 2))
    except ValueError as exc:
        assert "column count" in str(exc)
    else:
        raise AssertionError("expected column-count validation")
