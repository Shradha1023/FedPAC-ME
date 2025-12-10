"""
download.py

Utilities for:
- Downloading the BraTS2020 dataset (Kaggle)
- Extracting ZIP files
- Creating required folder structure

Requirements:
- kaggle
- unzip utility (Linux/macOS) or zipfile (Python)
"""

import os
import zipfile
import subprocess
from pathlib import Path


# -------------------------
# Configuration
# -------------------------
DATASET_NAME = "awsaf49/brats20-dataset-training-validation"
OUTPUT_DIR = Path("data")
ZIP_NAME = "brats20-dataset-training-validation.zip"


# -------------------------
# Helper functions
# -------------------------
def create_folders():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Dataset directory created at: {OUTPUT_DIR}")


def download_from_kaggle():
    """
    Downloads dataset using Kaggle CLI.
    Make sure KAGGLE_USERNAME and KAGGLE_KEY are set as environment variables.
    """
    print("[INFO] Downloading dataset from Kaggle...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET_NAME],
        check=True
    )
    print("[INFO] Download complete")


def extract_zip():
    zip_path = Path(ZIP_NAME)
    if not zip_path.exists():
        raise FileNotFoundError(f"{ZIP_NAME} not found")

    print("[INFO] Extracting ZIP file...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(OUTPUT_DIR)

    print(f"[INFO] Dataset extracted to {OUTPUT_DIR}")


# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":
    create_folders()
    download_from_kaggle()
    extract_zip()
    print("[DONE] Dataset is ready.")
