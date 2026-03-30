"""
split_clients.py

Split BraTS dataset into multiple client folders
for federated learning.
"""

import os
import shutil
import numpy as np

# List all patient folders
patient_folders = sorted(os.listdir(dataset_path))

num_clients = 10
alpha = 0.5  

# Create client directories
client_dirs = []
for i in range(num_clients):
    client_folder = os.path.join(dataset_path, f"client_{i+1}")
    os.makedirs(client_folder, exist_ok=True)
    client_dirs.append(client_folder)

# Shuffle data
np.random.shuffle(patient_folders)

# Generate Dirichlet distribution for all samples
dirichlet_dist = np.random.dirichlet([alpha] * num_clients, len(patient_folders))

# Assign each folder to a client
client_data_count = [0] * num_clients

for idx, folder in enumerate(patient_folders):
    src_path = os.path.join(dataset_path, folder)

    # Sample client index based on Dirichlet probabilities
    client_idx = np.argmax(dirichlet_dist[idx])

    dest_path = os.path.join(client_dirs[client_idx], folder)
    shutil.move(src_path, dest_path)

    client_data_count[client_idx] += 1

# Print distribution
for i in range(num_clients):
    print(f"Client {i+1}: {client_data_count[i]} folders")

print("Dataset successfully divided into non-IID clients using Dirichlet distribution.")
