"""
dataset.py

PyTorch Dataset for BraTS MRI data.
"""

import os
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset


class BrainTumorDataset(Dataset):
    def __init__(self, data_folder, slice_mode="middle"):
        self.data_folder = data_folder
        self.slice_mode = slice_mode

        self.patient_dirs = [
            os.path.join(data_folder, p)
            for p in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, p))
        ]

    def __len__(self):
        return len(self.patient_dirs)

    def _load_nifti(self, path):
        return nib.load(path).get_fdata()

    def __getitem__(self, idx):
        patient_path = self.patient_dirs[idx]

        files = os.listdir(patient_path)
        flair_path = [f for f in files if "flair" in f][0]
        t1ce_path  = [f for f in files if "t1ce" in f][0]
        seg_files  = [f for f in files if "seg" in f]

        flair = self._load_nifti(os.path.join(patient_path, flair_path))
        t1ce  = self._load_nifti(os.path.join(patient_path, t1ce_path))
        seg   = self._load_nifti(os.path.join(patient_path, seg_files[0])) if seg_files else np.zeros_like(flair)

        z = flair.shape[-1] // 2
        flair = flair[:, :, z]
        t1ce  = t1ce[:, :, z]
        seg   = seg[:, :, z]

        image = np.stack([flair, t1ce], axis=0)
        mask  = (seg > 0).astype(np.float32)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
