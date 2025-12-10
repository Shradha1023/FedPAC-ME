"""
preprocess.py

Preprocessing utilities for BraTS2020 dataset:
- Load NIfTI MRI modalities
- Min-Max normalization
- Segmentation file cleanup
- Optional visualization for sanity check
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# -----------------------
# Configuration
# -----------------------
DATASET_DIR = Path("data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
modalities = ["t1", "t1ce", "t2", "flair"]


# -----------------------
# Utility functions
# -----------------------
def normalize_volume(volume):
    """Min-Max normalize a 3D MRI volume"""
    scaler = MinMaxScaler()
    shape = volume.shape
    volume = scaler.fit_transform(volume.reshape(-1, shape[-1])).reshape(shape)
    return volume


def rename_segmentation(case_path):
    """Fix inconsistent segmentation filename if needed"""
    old_name = case_path / "W39_1998.09.19_Segm.nii"
    new_name = case_path / f"{case_path.name}_seg.nii"
    if old_name.exists() and not new_name.exists():
        old_name.rename(new_name)
        print(f"[INFO] Renamed segmentation file in {case_path.name}")


def load_case(case_path):
    """Load and preprocess all modalities for one patient"""
    images = []

    for mod in modalities:
        img_path = case_path / f"{case_path.name}_{mod}.nii"
        img = nib.load(img_path).get_fdata()
        img = normalize_volume(img)
        images.append(img)

    seg_path = case_path / f"{case_path.name}_seg.nii"
    seg = nib.load(seg_path).get_fdata()

    images = np.stack(images, axis=-1)  # (H, W, D, C)
    return images, seg


def visualize_sample(images, segmentation, slice_idx=95):
    """Visual sanity check"""
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    titles = ["T1", "T1ce", "T2", "FLAIR", "Segmentation"]
    for i in range(4):
        axes[i].imshow(images[:, :, slice_idx, i], cmap="gray")
        axes[i].set_title(titles[i])
        axes[i].axis("off")

    axes[4].imshow(segmentation[:, :, slice_idx], cmap="jet")
    axes[4].set_title("Mask")
    axes[4].axis("off")

    plt.tight_layout()
    plt.show()


# -----------------------
# Main preprocessing loop
# -----------------------
def preprocess_dataset(visualize=False):
    cases = sorted(DATASET_DIR.glob("BraTS20_Training_*"))
    print(f"[INFO] Found {len(cases)} cases")

    for case in cases:
        rename_segmentation(case)
        images, seg = load_case(case)

        if visualize:
            visualize_sample(images, seg)
            break   # visualize only one case

    print("[DONE] Preprocessing completed")


if __name__ == "__main__":
    preprocess_dataset(visualize=True)
