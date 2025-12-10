"""
split_clients.py

Split BraTS dataset into multiple client folders
for federated learning.
"""

import os
import shutil


def split_dataset_into_clients(
    dataset_path,
    num_clients=10,
):
    patient_folders = sorted([
        f for f in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, f))
    ])

    folders_per_client = len(patient_folders) // num_clients

    for i in range(num_clients):
        client_folder = os.path.join(dataset_path, f"client_{i+1}")
        os.makedirs(client_folder, exist_ok=True)

        start_idx = i * folders_per_client
        end_idx = start_idx + folders_per_client

        for folder in patient_folders[start_idx:end_idx]:
            src = os.path.join(dataset_path, folder)
            dst = os.path.join(client_folder, folder)
            shutil.move(src, dst)

        print(f"Client {i+1}: {folders_per_client} patients assigned")

    print("✅ Dataset successfully split into clients")


if __name__ == "__main__":
    BASE_PATH = "/content/dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    split_dataset_into_clients(BASE_PATH)
