import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

dataset_path = "/content/dataset/HMU-GC-HE-30K"
image_path = os.path.join(dataset_path, "all_image")

class_counts = {}

for cls in os.listdir(image_path):
    cls_path = os.path.join(image_path, cls)
    if os.path.isdir(cls_path):
        class_counts[cls] = len(os.listdir(cls_path))

# Convert to DataFrame
df_counts = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])
df_counts = df_counts.sort_values(by="Count", ascending=False)

print(df_counts)

plt.figure(figsize=(8,5))
sns.barplot(x="Class", y="Count", data=df_counts)
plt.title("Class Distribution")
plt.xticks(rotation=45)
plt.show()

import random

def show_samples(class_name, n=5):
    cls_path = os.path.join(image_path, class_name)
    images = random.sample(os.listdir(cls_path), n)

    plt.figure(figsize=(15,3))
    for i, img_name in enumerate(images):
        img = Image.open(os.path.join(cls_path, img_name))
        plt.subplot(1, n, i+1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis('off')
    plt.show()

for cls in ["ADI","DEB","LYM","MUC","MUS","NOR","STR", "TUM"]:
    show_samples(cls)

sizes = []

for cls in os.listdir(image_path):
    cls_path = os.path.join(image_path, cls)
    for img_name in os.listdir(cls_path)[:50]:  # sample 50 per class
        img = Image.open(os.path.join(cls_path, img_name))
        sizes.append(img.size)

print(set(sizes))

path = "/content/dataset"

csv_path = os.path.join(path, "HMU-GC-Clinical.csv")
df = pd.read_csv(csv_path)

print(df.head())
print(df.info())
print(df.describe())

print(df.columns)
