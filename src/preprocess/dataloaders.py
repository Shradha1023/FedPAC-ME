"""
dataloaders.py

Create client-wise and validation DataLoaders.
"""

import os
from torch.utils.data import DataLoader
from dataset import BrainTumorDataset


def create_client_loaders(
    base_path,
    num_clients=10,
    batch_size=4,
):
    client_loaders = {}

    for i in range(1, num_clients + 1):
        client_path = os.path.join(base_path, f"client_{i}")
        dataset = BrainTumorDataset(client_path)
        client_loaders[f"client_{i}"] = DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

    return client_loaders


def create_validation_loader(
    val_path,
    batch_size=4,
):
    dataset = BrainTumorDataset(val_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
