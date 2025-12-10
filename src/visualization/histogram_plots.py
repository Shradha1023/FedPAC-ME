"""
histogram_plots.py

Utilities for plotting MRI intensity histograms
for different modalities and slices.
"""

import matplotlib.pyplot as plt


# -------------------------
# Single-modality histogram
# -------------------------
def plot_slice_histogram(
    volume,
    slice_idx,
    title="Intensity Histogram",
    bins=50,
    alpha=0.7,
):
    """
    Plot intensity histogram for a single slice of a modality.

    Parameters:
        volume : np.ndarray (H, W, D)
        slice_idx : int
    """

    plt.figure(figsize=(5, 4))
    plt.hist(volume[:, :, slice_idx].ravel(), bins=bins, alpha=alpha)
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# -------------------------
# Multimodal comparison
# -------------------------
def plot_multimodal_histograms(
    t1,
    t1ce,
    t2,
    flair,
    slice_idx,
    bins=50,
    alpha=0.5,
):
    """
    Plot overlaid intensity histograms for multiple modalities
    of the same slice.
    """

    modalities = {
        "T1": t1,
        "T1ce": t1ce,
        "T2": t2,
        "FLAIR": flair,
    }

    plt.figure(figsize=(7, 5))

    for name, vol in modalities.items():
        plt.hist(
            vol[:, :, slice_idx].ravel(),
            bins=bins,
            alpha=alpha,
            label=name,
        )

    plt.title("Intensity Histogram Comparison")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------------------------
# Whole-volume histogram
# -------------------------
def plot_volume_histogram(
    volume,
    title="Volume Intensity Histogram",
    bins=100,
):
    """
    Plot histogram over entire 3D volume
    """

    plt.figure(figsize=(6, 4))
    plt.hist(volume.ravel(), bins=bins)
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
