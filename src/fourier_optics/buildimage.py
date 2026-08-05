from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
Shape = Tuple[int, int]


class PadingDefectsMatrix:
    """Embed defect matrices into a larger photon-count image."""

    def __init__(
        self,
        defs_dict: Mapping[str, ArrayLike],
        bg_photon_num: float,
        image_shape: Shape,
        def_matrix_shape: Optional[Shape] = None,
        def_matrix_step: int = 80,
    ) -> None:
        self.defs_dict = dict(defs_dict)
        self.bg_photon_num = float(bg_photon_num)
        self.image_shape = self._validate_shape(image_shape, "image_shape")
        if def_matrix_shape is None:
            def_matrix_shape = (10, len(self.defs_dict))
        self.def_matrix_shape = self._validate_shape(def_matrix_shape, "def_matrix_shape")
        self.def_matrix_step = int(def_matrix_step)
        if not self.defs_dict:
            raise ValueError("Defs_dict must contain at least one defect matrix")
        if self.bg_photon_num < 0:
            raise ValueError("bg_photon_num must be non-negative")
        if self.def_matrix_step <= 0:
            raise ValueError("def_matrix_step must be positive")
        if self.def_matrix_shape[1] != len(self.defs_dict):
            raise ValueError("def_matrix_shape column count must match len(Defs_dict)")

    def build(self) -> FloatArray:
        image = np.full(self.image_shape, self.bg_photon_num, dtype=np.float64)
        names = list(self.defs_dict)
        for row_index in range(self.def_matrix_shape[0]):
            for col_index, name in enumerate(names):
                center = self._site_center(row_index, col_index)
                patch = self._prepare_patch(self.defs_dict[name])
                self._paste_patch(image, patch, center)
        return image

    def build_random(
        self,
        def_val: ArrayLike,
        n_def_copy: int,
        n_imgs: int = 1,
        rng_seed: Optional[int] = None,
        max_attempts: int = 100_000,
    ) -> tuple[list[FloatArray], list[list[Shape]]]:
        if n_def_copy < 0:
            raise ValueError("n_def_copy must be non-negative")
        if n_imgs <= 0:
            raise ValueError("n_imgs must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

        patch = self._prepare_patch(def_val)
        patch_rows, patch_cols = patch.shape
        image_rows, image_cols = self.image_shape
        if patch_rows > image_rows or patch_cols > image_cols:
            raise ValueError("def_val must fit inside image_shape")

        rng = np.random.default_rng(rng_seed)
        images: list[FloatArray] = []
        coords_list: list[list[Shape]] = []
        for _ in range(n_imgs):
            image, centers = self._build_one_random_image(
                patch=patch,
                n_def_copy=n_def_copy,
                rng=rng,
                max_attempts=max_attempts,
            )
            images.append(image)
            coords_list.append(centers)
        return images, coords_list

    def _validate_shape(self, shape: Shape, name: str) -> Shape:
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError(f"{name} must be a pair of positive integers")
        return int(shape[0]), int(shape[1])

    def _site_center(self, row_index: int, col_index: int) -> Shape:
        rows, cols = self.def_matrix_shape
        image_rows, image_cols = self.image_shape
        center_row = (image_rows - 1) / 2 + (row_index - (rows - 1) / 2) * self.def_matrix_step
        center_col = (image_cols - 1) / 2 + (col_index - (cols - 1) / 2) * self.def_matrix_step
        return int(round(center_row)), int(round(center_col))

    def _prepare_patch(self, matrix: ArrayLike) -> FloatArray:
        patch = np.asarray(matrix, dtype=np.float64)
        if patch.ndim != 2:
            raise ValueError("each defect matrix must be 2D")
        patch = np.clip(patch, 0.0, None)
        return patch * self._edge_window(patch.shape)

    def _edge_window(self, shape: Shape) -> FloatArray:
        rows, cols = shape
        taper = max(1.0, min(rows, cols) * 0.35)
        y_distance = np.minimum(np.arange(rows), np.arange(rows)[::-1])
        x_distance = np.minimum(np.arange(cols), np.arange(cols)[::-1])
        y_weight = np.sin(0.5 * np.pi * np.clip(y_distance / taper, 0.0, 1.0))
        x_weight = np.sin(0.5 * np.pi * np.clip(x_distance / taper, 0.0, 1.0))
        return np.outer(y_weight, x_weight).astype(np.float64)

    def _paste_patch(self, image: FloatArray, patch: FloatArray, center: Shape) -> None:
        center_row, center_col = center
        patch_rows, patch_cols = patch.shape
        row_start = center_row - patch_rows // 2
        col_start = center_col - patch_cols // 2
        row_end = row_start + patch_rows
        col_end = col_start + patch_cols

        image_row_start = max(row_start, 0)
        image_col_start = max(col_start, 0)
        image_row_end = min(row_end, image.shape[0])
        image_col_end = min(col_end, image.shape[1])
        if image_row_start >= image_row_end or image_col_start >= image_col_end:
            return

        patch_row_start = image_row_start - row_start
        patch_col_start = image_col_start - col_start
        patch_row_end = patch_row_start + (image_row_end - image_row_start)
        patch_col_end = patch_col_start + (image_col_end - image_col_start)
        image[image_row_start:image_row_end, image_col_start:image_col_end] += patch[
            patch_row_start:patch_row_end, patch_col_start:patch_col_end
        ]

    def _boxes_overlap(
        self,
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> bool:
        first_row_start, first_row_end, first_col_start, first_col_end = first
        second_row_start, second_row_end, second_col_start, second_col_end = second
        return not (
            first_row_end <= second_row_start
            or second_row_end <= first_row_start
            or first_col_end <= second_col_start
            or second_col_end <= first_col_start
        )

    def _build_one_random_image(
        self,
        patch: FloatArray,
        n_def_copy: int,
        rng: np.random.Generator,
        max_attempts: int,
    ) -> tuple[FloatArray, list[Shape]]:
        patch_rows, patch_cols = patch.shape
        image_rows, image_cols = self.image_shape
        image = np.full(self.image_shape, self.bg_photon_num, dtype=np.float64)
        boxes: list[tuple[int, int, int, int]] = []
        centers: list[Shape] = []
        attempts = 0

        while len(centers) < n_def_copy and attempts < max_attempts:
            attempts += 1
            row_start = int(rng.integers(0, image_rows - patch_rows + 1))
            col_start = int(rng.integers(0, image_cols - patch_cols + 1))
            box = (row_start, row_start + patch_rows, col_start, col_start + patch_cols)
            if any(self._boxes_overlap(box, existing) for existing in boxes):
                continue

            center = (row_start + patch_rows // 2, col_start + patch_cols // 2)
            self._paste_patch(image, patch, center)
            boxes.append(box)
            centers.append(center)

        if len(centers) != n_def_copy:
            raise RuntimeError("could not place all defects without overlap")
        return image, centers


def build_def_lib(
    def_count: int = 10,
    matrix_shape: Shape = (61, 61),
    fwhm_range: Tuple[float, float] = (30.0, 5.0),
    peak_photon_num: float = 12000.0,
) -> dict[str, FloatArray]:
    """Build a demo defect library with left-to-right shrinking Gaussian spots."""

    if def_count <= 0:
        raise ValueError("def_count must be positive")
    if peak_photon_num < 0:
        raise ValueError("peak_photon_num must be non-negative")
    if len(matrix_shape) != 2 or min(matrix_shape) <= 0:
        raise ValueError("matrix_shape must be a pair of positive integers")
    rows, cols = int(matrix_shape[0]), int(matrix_shape[1])
    y = np.arange(rows, dtype=np.float64) - (rows - 1) / 2
    x = np.arange(cols, dtype=np.float64) - (cols - 1) / 2
    xx, yy = np.meshgrid(x, y)
    defects: dict[str, FloatArray] = {}
    for index, fwhm in enumerate(np.linspace(fwhm_range[0], fwhm_range[1], def_count), start=1):
        if fwhm <= 0:
            raise ValueError("fwhm_range values must be positive")
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        defects[f"def{index}"] = peak_photon_num * np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    return defects


def build_picture(
    Defs_dict: Mapping[str, ArrayLike],
    bg_photon_num: float,
    image_shape: Shape,
    def_matrix_shape: Optional[Shape] = None,
    def_matrix_step: int = 80,
) -> FloatArray:
    """Build an image_shape photon-count matrix from a defect dictionary."""

    return PadingDefectsMatrix(
        defs_dict=Defs_dict,
        bg_photon_num=bg_photon_num,
        image_shape=image_shape,
        def_matrix_shape=def_matrix_shape,
        def_matrix_step=def_matrix_step,
    ).build()


def build_picture_from_list(
    defs_list: Sequence[ArrayLike],
    bg_photon_num: float,
    image_shape: Shape,
    def_matrix_shape: Optional[Shape] = None,
    def_matrix_step: int = 80,
) -> FloatArray:
    """Build a photon-count image from a defect list."""

    defs_dict = {f"def{index}": value for index, value in enumerate(defs_list, start=1)}
    return build_picture(
        Defs_dict=defs_dict,
        bg_photon_num=bg_photon_num,
        image_shape=image_shape,
        def_matrix_shape=def_matrix_shape,
        def_matrix_step=def_matrix_step,
    )


def build_random_picture(
    def_val: ArrayLike,
    bg_photon_num: float,
    image_shape: Shape,
    n_def_copy: int,
    n_imgs: int = 1,
    rng_seed: Optional[int] = None,
    max_attempts: int = 100_000,
) -> tuple[list[FloatArray], list[list[Shape]]]:
    """Randomly place non-overlapping copies of one defect inside images."""

    builder = PadingDefectsMatrix(
        defs_dict={"def1": def_val},
        bg_photon_num=bg_photon_num,
        image_shape=image_shape,
        def_matrix_shape=(1, 1),
    )
    return builder.build_random(
        def_val=def_val,
        n_def_copy=n_def_copy,
        n_imgs=n_imgs,
        rng_seed=rng_seed,
        max_attempts=max_attempts,
    )


def plot_picture(
    image: ArrayLike,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Photon-count defect image",
    cmap: str = "gray",
) -> Optional[Path]:
    """Plot a photon-count image and optionally save it to disk."""

    import matplotlib.pyplot as plt

    image_array = np.asarray(image, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(image_array, cmap=cmap, origin="lower")
    ax.set_title(title)
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    fig.colorbar(im, ax=ax, label="photons")
    fig.tight_layout()
    if save_path is None:
        plt.show()
        return None
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
