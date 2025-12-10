"""
plot_modalities.py

Utilities for side-by-side visualization of MRI modalities
(T1, T1ce, T2, FLAIR)
"""

import matplotlib.pyplot as plt


# -------------------------
# Side-by-side plotting
# -------------------------
def plot_modalities(
    t1,
    t1ce,
    t2,
    flair,
    slice_idx,
    titles=("T1", "T1ce", "T2", "FLAIR"),
    cmap="gray",
    figsize=(16, 4),
):
    """
    Plot T1, T1ce, T2, and FLAIR modalities side-by-side for a given slice.

    Parameters:
        t1, t1ce, t2, flair : np.ndarray
            3D MRI volumes with shape (H, W, D)
        slice_idx : int
            Slice index to visualize
    """

    volumes = [t1, t1ce, t2, flair]

    plt.figure(figsize=figsize)
    for i, (vol, title) in enumerate(zip(volumes, titles)):
        plt.subplot(1, 4, i + 1)
        plt.imshow(vol[:, :, slice_idx], cmap=cmap)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# -------------------------
# Optional: averaged view
# -------------------------
def plot_average_modality(t1, t1ce, t2, flair, slice_idx, cmap="gray"):
    """
    Plot averaged modality image for comparison
    """
    avg = (t1[:, :, slice_idx] +
           t1ce[:, :, slice_idx] +
           t2[:, :, slice_idx] +
           flair[:, :, slice_idx]) / 4.0

    plt.figure(figsize=(4, 4))
    plt.imshow(avg, cmap=cmap)
    plt.title("Average of Modalities")
    plt.axis("off")
    plt.show()
