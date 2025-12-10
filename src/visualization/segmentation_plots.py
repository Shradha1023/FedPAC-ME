"""
segmentation_plots.py

Utilities for visualizing MRI slices with segmentation overlays.
Supports transparency and contour-based mask visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


# -------------------------
# Basic mask overlay
# -------------------------
def overlay_mask(
    image,
    mask,
    slice_idx,
    alpha=0.4,
    image_cmap="gray",
    mask_cmap="jet",
    title="Image + Mask",
):
    """
    Overlay segmentation mask on a single MRI slice.

    Parameters:
        image : np.ndarray (H, W, D)
        mask  : np.ndarray (H, W, D)
        slice_idx : int
    """

    plt.figure(figsize=(6, 6))
    plt.imshow(image[:, :, slice_idx], cmap=image_cmap)
    plt.imshow(mask[:, :, slice_idx], cmap=mask_cmap, alpha=alpha)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# -------------------------
# Contour overlay
# -------------------------
def contour_overlay(
    image,
    mask,
    slice_idx,
    image_cmap="gray",
    contour_color="red",
    linewidths=1.5,
    title="Contour Overlay",
):
    """
    Overlay segmentation contours on MRI slice.
    """

    plt.figure(figsize=(6, 6))
    plt.imshow(image[:, :, slice_idx], cmap=image_cmap)
    plt.contour(mask[:, :, slice_idx], colors=contour_color, linewidths=linewidths)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# -------------------------
# Multi-class segmentation overlay
# -------------------------
def multiclass_overlay(
    image,
    mask,
    slice_idx,
    class_labels=(0, 1, 2, 4),
    class_names=None,
    image_cmap="gray",
    alpha=0.4,
    title="Multi-class Segmentation",
):
    """
    Overlay multi-class segmentation mask with legend.

    Expected BraTS labels: 0, 1, 2, 4
    """

    cmap = colors.ListedColormap(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(6, 6))
    plt.imshow(image[:, :, slice_idx], cmap=image_cmap)
    plt.imshow(mask[:, :, slice_idx], cmap=cmap, norm=norm, alpha=alpha)

    if class_names:
        legend_patches = [
            plt.Rectangle((0, 0), 1, 1, color=cmap(i), label=class_names[i])
            for i in range(len(class_names))
        ]
        plt.legend(handles=legend_patches, loc="lower left")

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# -------------------------
# Side-by-side comparison
# -------------------------
def compare_overlay(
    image,
    mask,
    slice_idx,
    alpha=0.4,
):
    """
    Show raw image and overlay side-by-side.
    """

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image[:, :, slice_idx], cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image[:, :, slice_idx], cmap="gray")
    plt.imshow(mask[:, :, slice_idx], cmap="jet", alpha=alpha)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
