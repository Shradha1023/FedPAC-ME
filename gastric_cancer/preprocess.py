import os
import shutil
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ==============================
# 🔧 CONFIG
# ==============================
np.random.seed(42)

dataset = "/content/dataset/HMU-GC-HE-30K/all_image"
output_path = "/content/federated_dataset"

classes = ["ADI","DEB","LYM","MUC","MUS","NOR","STR","TUM"]
num_clients = 10
alpha = 0.5
min_samples = 50

train_ratio = 0.6
val_ratio = 0.2  # test will be remaining

# ==============================
# 📦 LOAD DATA PATHS
# ==============================
data = {cls: [] for cls in classes}

for cls in classes:
    cls_path = os.path.join(dataset, cls)
    for img in os.listdir(cls_path):
        data[cls].append(os.path.join(cls_path, img))

# ==============================
# 📁 CREATE FOLDER STRUCTURE
# ==============================
splits = ["train", "val", "test"]

for i in range(num_clients):
    for split in splits:
        for cls in classes:
            os.makedirs(
                os.path.join(output_path, f"client_{i+1}", split, cls),
                exist_ok=True
            )

# ==============================
# 🎯 DIRICHLET NON-IID SPLIT (WITH MIN SAMPLES)
# ==============================
client_data = {i: {cls: [] for cls in classes} for i in range(num_clients)}

for cls in classes:
    np.random.shuffle(data[cls])

    while True:
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(data[cls])).astype(int)

        # Fix rounding issue
        while proportions.sum() < len(data[cls]):
            proportions[np.argmax(proportions)] += 1

        if np.all(proportions > min_samples):
            break

    start = 0
    for i in range(num_clients):
        end = start + proportions[i]
        client_data[i][cls].extend(data[cls][start:end])
        start = end

# ==============================
# 🔀 TRAIN / VAL / TEST SPLIT
# ==============================
split_data = {
    i: {"train": {cls: [] for cls in classes},
        "val": {cls: [] for cls in classes},
        "test": {cls: [] for cls in classes}}
    for i in range(num_clients)
}

for client_id in range(num_clients):
    for cls in classes:
        imgs = client_data[client_id][cls]

        if len(imgs) < 3:
            # Edge case handling
            split_data[client_id]["train"][cls] = imgs
            continue

        train, temp = train_test_split(
            imgs, test_size=(1 - train_ratio), random_state=42
        )

        val, test = train_test_split(
            temp, test_size=0.5, random_state=42
        )

        split_data[client_id]["train"][cls] = train
        split_data[client_id]["val"][cls] = val
        split_data[client_id]["test"][cls] = test

# ==============================
# 📂 COPY FILES TO FINAL STRUCTURE
# ==============================
for client_id in range(num_clients):
    print(f"\nProcessing client_{client_id+1}")

    for split in splits:
        for cls in classes:
            for img_path in tqdm(split_data[client_id][split][cls]):
                filename = os.path.basename(img_path)
                dest_path = os.path.join(
                    output_path,
                    f"client_{client_id+1}",
                    split,
                    cls,
                    filename
                )

                shutil.copy(img_path, dest_path)
                # use shutil.move() if you want to save space

# ==============================
# 📊 FINAL DISTRIBUTION CHECK
# ==============================
for i in range(num_clients):
    print(f"\nClient {i+1}")

    for split in splits:
        print(f"  {split.upper()}:")
        for cls in classes:
            path = os.path.join(output_path, f"client_{i+1}", split, cls)
            print(f"    {cls}: {len(os.listdir(path))}")

