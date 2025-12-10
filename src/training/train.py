"""
train.py

Main training script for client-wise training
of the Attention-Contrastive MoE model.
"""

import os
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F

from models.attention_contrastive import AttentionContrastiveModel
from models.losses import PerspectiveAwareContrastiveLoss
from models.augmentations import MultiPerspectiveAugment
from data.dataloaders import create_client_loaders

# --------------------------------------------------
# Configuration
# --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_CLIENTS = 10
BATCH_SIZE = 4
CHECKPOINT_DIR = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --------------------------------------------------
# Load data
# --------------------------------------------------
TRAIN_PATH = "/content/dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
client_loaders = create_client_loaders(
    TRAIN_PATH, num_clients=NUM_CLIENTS, batch_size=BATCH_SIZE
)

# --------------------------------------------------
# Shared modules
# --------------------------------------------------
criterion = PerspectiveAwareContrastiveLoss()
augmentor = MultiPerspectiveAugment()

# --------------------------------------------------
# Client-wise Training
# --------------------------------------------------
local_models = {}
client_metrics = {}

for client_id, dataloader in client_loaders.items():
    print(f"\n🚀 Training {client_id}")

    model = AttentionContrastiveModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()

    epoch_losses = []

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0

        for images, _ in dataloader:
            images = images.to(DEVICE)
            optimizer.zero_grad()

            # Multi-perspective views
            pcv, ssv, gav = augmentor.apply(model, images)

            # Forward
            z = F.normalize(model(images), dim=1)
            z_pcv = F.normalize(model(pcv), dim=1)
            z_ssv = F.normalize(model(ssv), dim=1)
            z_gav = F.normalize(model(gav), dim=1)

            loss = criterion(z, z_pcv, z_ssv, z_gav)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)

        print(
            f"[{client_id}] Epoch {epoch+1}/{NUM_EPOCHS} "
            f"- Loss: {avg_loss:.4f}"
        )

    # --------------------------------------------------
    # Save client model
    # --------------------------------------------------
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{client_id}.pth")
    torch.save(model.state_dict(), ckpt_path)

    local_models[client_id] = copy.deepcopy(model.state_dict())
    client_metrics[client_id] = epoch_losses

    print(f"✅ {client_id} model saved")

print("\n🎯 Training completed for all clients")
