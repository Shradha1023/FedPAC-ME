import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ==============================
# 📁 PATHS
# ==============================
train_base_path = "/content/federated_dataset"   # your generated dataset
num_clients = 10

# ==============================
# 🧠 CLASS MAPPING
# ==============================
classes = ["ADI","DEB","LYM","MUC","MUS","NOR","STR","TUM"]
class_to_idx = {cls: i for i, cls in enumerate(classes)}

# ==============================
# 🔄 TRANSFORMS
# ==============================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# ==============================
# 📦 DATASET CLASS (UPDATED)
# ==============================
class GastricCancerDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.transform = transform
        self.samples = []

        for cls in classes:
            cls_path = os.path.join(data_folder, cls)
            if not os.path.exists(cls_path):
                continue

            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                self.samples.append((img_path, class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label

# ==============================
# 🔗 CREATE CLIENT DATALOADERS
# ==============================
client_dataloaders = {}

for i in range(1, num_clients + 1):
    client_path = os.path.join(train_base_path, f"client_{i}", "train")

    dataset = GastricCancerDataset(client_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    client_dataloaders[f"client_{i}"] = dataloader

print("Client train dataloaders created ✅")

# ==============================
# 📊 VALIDATION LOADERS 
# ==============================
val_loaders = {}

for i in range(1, num_clients + 1):
    val_path = os.path.join(train_base_path, f"client_{i}", "val")

    dataset = GastricCancerDataset(val_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    val_loaders[f"client_{i}"] = dataloader

print("Validation dataloaders created ✅")

