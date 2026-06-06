from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fourier_optics import HopkinsImagingModel


def line_grating(
    shape: tuple[int, int],
    pixel_size: float,
    cd: float,
    pitch: float,
    line_amplitude: float = 1.0,
    space_amplitude: float = 0.0,
) -> np.ndarray:
    period_pixels = int(round(pitch / pixel_size))
    cd_pixels = int(round(cd / pixel_size))
    if not np.isclose(period_pixels * pixel_size, pitch):
        raise ValueError("pitch must be an integer number of pixels for this task")
    if not np.isclose(cd_pixels * pixel_size, cd):
        raise ValueError("CD must be an integer number of pixels for this task")

    _, cols = shape
    x_index = np.arange(cols)
    phase = np.mod(x_index + period_pixels // 2, period_pixels)
    row = np.where(phase < cd_pixels, line_amplitude, space_amplitude)
    amplitude = np.broadcast_to(row, shape)
    return amplitude.astype(np.complex128)


def contrast(image: np.ndarray) -> float:
    image_min = float(np.min(image))
    image_max = float(np.max(image))
    return (image_max - image_min) / (image_max + image_min)


def equal_radius_overlap_fraction(center_distance: float, radius: float) -> float:
    if center_distance >= 2 * radius:
        return 0.0
    if center_distance <= 0:
        return 1.0
    area = (
        2 * radius * radius * np.arccos(center_distance / (2 * radius))
        - 0.5 * center_distance * np.sqrt(4 * radius * radius - center_distance * center_distance)
    )
    return float(area / (np.pi * radius * radius))


def diffraction_order_amplitudes(max_order: int) -> tuple[np.ndarray, np.ndarray]:
    orders = np.arange(-max_order, max_order + 1)
    amplitudes = np.zeros_like(orders, dtype=np.float64)
    for index, order in enumerate(orders):
        if order == 0:
            amplitudes[index] = 0.5
        elif order % 2 != 0:
            amplitudes[index] = abs(1 / (np.pi * order))
    return orders, amplitudes


def save_image_result(
    image: np.ndarray,
    pixel_size: float,
    pitch: float,
    output_path: Path,
) -> None:
    center_row = image[image.shape[0] // 2]
    x_nm = (np.arange(image.shape[1]) - image.shape[1] / 2) * pixel_size * 1e9
    crop_half_width = int(round(2.5 * pitch / pixel_size))
    center = image.shape[1] // 2
    crop = image[:, center - crop_half_width : center + crop_half_width]
    crop_x = x_nm[center - crop_half_width : center + crop_half_width]

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)
    extent = [crop_x[0], crop_x[-1], -image.shape[0] * pixel_size * 0.5e9, image.shape[0] * pixel_size * 0.5e9]
    im = axes[0].imshow(crop, cmap="magma", aspect="auto", extent=extent, origin="lower")
    axes[0].set_title("Aerial image intensity, finite illumination NA")
    axes[0].set_xlabel("x (nm)")
    axes[0].set_ylabel("y (nm)")
    fig.colorbar(im, ax=axes[0], label="Intensity")

    axes[1].plot(x_nm, center_row, color="#22577a", linewidth=1.8)
    axes[1].set_xlim(-2.5 * pitch * 1e9, 2.5 * pitch * 1e9)
    axes[1].set_ylim(float(np.min(image)) - 0.02, float(np.max(image)) + 0.02)
    axes[1].set_title("Center-row intensity")
    axes[1].set_xlabel("x (nm)")
    axes[1].set_ylabel("Intensity")
    axes[1].grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_pupil_orders(
    wavelength: float,
    pitch: float,
    illumination_na: float,
    po_na: float,
    incident_angle_deg: float,
    output_path: Path,
) -> None:
    orders, amplitudes = diffraction_order_amplitudes(max_order=5)
    pupil_center_na = np.sin(np.deg2rad(incident_angle_deg))
    order_na_x = pupil_center_na + orders * wavelength / pitch
    intensities = amplitudes**2

    x_min = pupil_center_na - 0.36
    x_max = pupil_center_na + 0.36
    y_min = -0.22
    y_max = 0.22
    x_axis = np.linspace(x_min, x_max, 900)
    y_axis = np.linspace(y_min, y_max, 550)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    energy = np.zeros_like(x_grid)

    for center_x, order_energy in zip(order_na_x, intensities):
        if order_energy == 0:
            continue
        source_disk = (x_grid - center_x) ** 2 + y_grid**2 <= illumination_na**2
        energy[source_disk] += order_energy

    po_mask = (x_grid - pupil_center_na) ** 2 + y_grid**2 <= po_na**2
    captured_energy = np.where(po_mask, energy, np.nan)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(
        captured_energy,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        cmap="viridis",
        aspect="equal",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="Relative energy density")
    pupil = plt.Circle((pupil_center_na, 0), po_na, fill=False, color="#1d3557", linewidth=2.0)
    ax.add_patch(pupil)

    for center_x, order, order_energy in zip(order_na_x, orders, intensities):
        if order_energy == 0:
            continue
        source_disk_edge = plt.Circle(
            (center_x, 0),
            illumination_na,
            fill=False,
            color="#d62828" if abs(center_x - pupil_center_na) > po_na else "#2a9d8f",
            linewidth=1.2,
            alpha=0.75,
            linestyle="--",
        )
        ax.add_patch(source_disk_edge)

    ax.axhline(0, color="0.75", linewidth=1)
    ax.axvline(pupil_center_na, color="0.75", linewidth=1)

    for order, x, amp, energy in zip(orders, order_na_x, amplitudes, intensities):
        if energy == 0:
            continue
        ax.annotate(
            f"m={order}\nE={energy:.4f}",
            xy=(x, 0),
            xytext=(0, 18 if order % 2 == 0 else -36),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "0.6", "linewidth": 0.8},
        )

    incidence_frequency = np.sin(np.deg2rad(incident_angle_deg)) / wavelength
    ax.text(
        0.02,
        0.98,
        "Absolute NA pupil plane\n"
        "Color shows energy captured inside PO\n"
        "Dashed circles are illumination-NA diffraction disks\n"
        f"6 deg incident carrier = {incidence_frequency / 1e6:.3f} 1/um",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.9},
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("NAx")
    ax.set_ylabel("NAy")
    ax.set_title("Diffraction-order energy distribution on PO pupil")
    ax.grid(True, alpha=0.18)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_pupil_capture_breakdown(
    wavelength: float,
    pitch: float,
    illumination_na: float,
    po_na: float,
    incident_angle_deg: float,
    output_path: Path,
) -> None:
    orders = np.array([-1, 0, 1])
    amplitudes = np.array([1 / np.pi, 0.5, 1 / np.pi])
    intensities = amplitudes**2
    pupil_center_na = np.sin(np.deg2rad(incident_angle_deg))
    order_na_x = pupil_center_na + orders * wavelength / pitch

    x_min = pupil_center_na - 0.24
    x_max = pupil_center_na + 0.24
    y_min = -0.13
    y_max = 0.13
    x_axis = np.linspace(x_min, x_max, 1000)
    y_axis = np.linspace(y_min, y_max, 600)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    po_mask = (x_grid - pupil_center_na) ** 2 + y_grid**2 <= po_na**2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True, constrained_layout=True)
    for ax, order, center_x, order_energy in zip(axes, orders, order_na_x, intensities):
        source_disk = (x_grid - center_x) ** 2 + y_grid**2 <= illumination_na**2
        captured = np.where(po_mask & source_disk, order_energy, np.nan)
        ax.imshow(
            captured,
            extent=[x_min, x_max, y_min, y_max],
            origin="lower",
            cmap="viridis",
            aspect="equal",
            vmin=0.0,
            vmax=float(np.max(intensities)),
            interpolation="nearest",
        )
        ax.add_patch(plt.Circle((pupil_center_na, 0), po_na, fill=False, color="white", linewidth=2.0))
        ax.add_patch(plt.Circle((center_x, 0), illumination_na, fill=False, color="#ff595e", linewidth=1.4, linestyle="--"))
        ax.axvline(pupil_center_na, color="white", linewidth=0.9, alpha=0.7)
        ax.axhline(0, color="white", linewidth=0.9, alpha=0.7)
        overlap = float(np.count_nonzero(po_mask & source_disk) / np.count_nonzero(source_disk))
        ax.set_title(f"m={order}, E={order_energy:.4f}\ncaptured area={overlap:.2%}")
        ax.set_xlabel("NAx")
        ax.grid(True, color="white", alpha=0.15)

    axes[0].set_ylabel("NAy")
    fig.suptitle("Captured diffraction-order energy inside PO")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    wavelength = 13.5e-9
    cd = 52e-9
    pitch = 104e-9
    incident_angle_deg = 6.0
    illumination_na = 0.33 / 4
    po_na = illumination_na
    pixel_size = 1e-9
    shape = (256, 1040)

    field = line_grating(
        shape=shape,
        pixel_size=pixel_size,
        cd=cd,
        pitch=pitch,
        line_amplitude=1.0,
        space_amplitude=0.0,
    )

    recentered_model = HopkinsImagingModel(
        wavelength=wavelength,
        numerical_aperture=po_na,
        pixel_size=pixel_size,
        source_sigma=0.0,
        source_samples=1,
    )
    fixed_po_model = HopkinsImagingModel(
        wavelength=wavelength,
        numerical_aperture=po_na,
        pixel_size=pixel_size,
        source_sigma=illumination_na / po_na,
        source_samples=31,
    )

    recentered_image = recentered_model.image_intensity(field)
    fixed_po_image = fixed_po_model.image_intensity(field)

    grating_frequency = 1 / pitch
    cutoff_frequency = po_na / wavelength
    incidence_frequency = np.sin(np.deg2rad(incident_angle_deg)) / wavelength
    order_spacing_na = wavelength / pitch
    first_order_overlap = equal_radius_overlap_fraction(order_spacing_na, po_na)
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    image_output = output_dir / "line_grating_image.png"
    pupil_output = output_dir / "pupil_diffraction_orders.png"
    capture_output = output_dir / "pupil_capture_breakdown.png"
    save_image_result(fixed_po_image, pixel_size, pitch, image_output)
    save_pupil_orders(wavelength, pitch, illumination_na, po_na, incident_angle_deg, pupil_output)
    save_pupil_capture_breakdown(wavelength, pitch, illumination_na, po_na, incident_angle_deg, capture_output)

    print("Line grating imaging task")
    print(f"wavelength: {wavelength * 1e9:.3f} nm")
    print(f"CD: {cd * 1e9:.3f} nm")
    print(f"pitch: {pitch * 1e9:.3f} nm")
    print(f"incident angle: {incident_angle_deg:.3f} deg")
    print(f"illumination NA: {illumination_na:.6f}")
    print(f"PO NA: {po_na:.6f}")
    print(f"incident spatial frequency: {incidence_frequency / 1e6:.6f} 1/um")
    print(f"grating first-order frequency: {grating_frequency / 1e6:.6f} 1/um")
    print(f"PO cutoff frequency: {cutoff_frequency / 1e6:.6f} 1/um")
    print(f"diffraction order spacing in NA: {order_spacing_na:.6f}")
    print(f"partial-coherence capture limit NAillum + NApo: {illumination_na + po_na:.6f}")
    print(f"first-order source disk overlap with PO: {first_order_overlap:.6%}")
    print("")
    print("Diagnostic: single on-axis source after re-centering the 0th order")
    print(f"  min: {np.min(recentered_image):.9g}")
    print(f"  max: {np.max(recentered_image):.9g}")
    print(f"  mean: {np.mean(recentered_image):.9g}")
    print(f"  contrast: {contrast(recentered_image):.9g}")
    print("")
    print("EUV finite-illumination model")
    print(f"  source sigma: {illumination_na / po_na:.6f}")
    print(f"  min: {np.min(fixed_po_image):.9g}")
    print(f"  max: {np.max(fixed_po_image):.9g}")
    print(f"  mean: {np.mean(fixed_po_image):.9g}")
    print(f"  contrast: {contrast(fixed_po_image):.9g}")
    print("")
    print("Center-row samples, Case B:")
    print(np.array2string(fixed_po_image[shape[0] // 2, :32], precision=6))
    print("")
    print(f"Saved image result: {image_output}")
    print(f"Saved pupil orders: {pupil_output}")
    print(f"Saved capture breakdown: {capture_output}")


if __name__ == "__main__":
    main()
