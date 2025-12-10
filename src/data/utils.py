"""
utils.py

General utility functions for:
- Path handling
- File & directory checks
- Listing BraTS patient folders
"""

from pathlib import Path
import os


# -----------------------
# Path utilities
# -----------------------
def resolve_path(path):
    """
    Resolve absolute path (works for local and Colab environments)
    """
    return Path(path).expanduser().resolve()


def ensure_dir(path):
    """
    Create directory if it does not exist
    """
    path = resolve_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# -----------------------
# File & directory checks
# -----------------------
def file_exists(path):
    """Check if a file exists"""
    return resolve_path(path).is_file()


def dir_exists(path):
    """Check if a directory exists"""
    return resolve_path(path).is_dir()


def validate_files(file_list):
    """
    Check multiple files and return missing ones
    """
    missing = [f for f in file_list if not file_exists(f)]
    return missing


# -----------------------
# BraTS helpers
# -----------------------
def list_patient_dirs(dataset_dir, prefix="BraTS20_Training"):
    """
    List all BraTS patient directories
    """
    dataset_dir = resolve_path(dataset_dir)
    return sorted(
        [p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    )


def check_modalities(patient_dir, modalities):
    """
    Verify all required modality files exist for a patient
    """
    patient_dir = resolve_path(patient_dir)
    missing = []

    for mod in modalities:
        file_path = patient_dir / f"{patient_dir.name}_{mod}.nii"
        if not file_path.exists():
            missing.append(file_path.name)

    return missing


# -----------------------
# Environment helpers
# -----------------------
def is_colab():
    """Detect if running inside Google Colab"""
    try:
        import google.colab  # noqa
        return True
    except ImportError:
        return False
