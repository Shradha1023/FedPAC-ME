"""
visualize_slices.py

Visualization utilities for BraTS MRI volumes:
- Axial / coronal / sagittal views
- Single or multi-modality slice display
- Segmentation overlay
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate


# -------------------------
# Basic slice viewers
# -------------------------
def show_single_slice(volume, slice_idx, title="", cmap="gray"):
    """Show a single axial slice from a 3D volume"""
    plt.figure(figsize=(4, 4))
    plt.imshow(volume[:, :, slice_idx], cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_multimodal_slice(volumes, slice_idx, titles):
    """
    Show multiple modalities for the same slice
    volumes: list of 3D arrays
    """
    n = len(volumes)
    plt.figure(figsize=(4 * n, 4))

    for i, vol in enumerate(volumes):
        plt.subplot(1, n, i + 1)
        plt.imshow(vol[:, :, slice_idx], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# -------------------------
# Orthogonal views
# -------------------------
def show_orthogonal_views(volume, slice_idx):
    """Axial, Coronal, Sagittal views"""
    plt.figure(figsize=(12, 4))

    # Axial
    plt.subplot(1, 3, 1)
    plt.imshow(volume[:, :, slice_idx], cmap="gray")
    plt.title("Axial")
    plt.axis("off")

    # Coronal
    plt.subplot(1, 3, 2)
    plt.imshow(rotate(volume[:, slice_idx, :], 90, reshape=True), cmap="gray")
    plt.title("Coronal")
    plt.axis("off")

    # Sagittal
    plt.subplot(1, 3, 3)
    plt.imshow(rotate(volume[slice_idx, :, :], 90, reshape=True), cmap="gray")
    plt.title("Sagittal")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# -------------------------
# Overlay utilities
# -------------------------
def show_overlay(image, mask, slice_idx, alpha=0.4):
    """Overlay segmentation mask on image"""
    plt.figure(figsize=(5, 5))
    plt.imshow(image[:, :, slice_idx], cmap="gray")
    plt.imshow(mask[:, :, slice_idx], cmap="jet", alpha=alpha)
    plt.title("Overlay")
    plt.axis("off")
    plt.show()


# -------------------------
# Histogram
# -------------------------
def show_intensity_histogram(volume, slice_idx, title="Intensity Histogram"):
    """Plot pixel intensity histogram for a slice"""
    plt.figure(figsize=(5, 4))
    plt.hist(volume[:, :, slice_idx].ravel(), bins=50)
    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.show()
