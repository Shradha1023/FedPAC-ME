import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import random

# ==============================
# 🔹 Multi-Head Self-Attention
# ==============================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (batch, embed_dim) → convert to sequence
        x = x.unsqueeze(1)  # (batch, 1, embed_dim)
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(x + attn_output)
        return x.squeeze(1)  # back to (batch, embed_dim)

# ==============================
# 🔹 Mixture of Experts
# ==============================
class MixtureOfExperts(nn.Module):
    def __init__(self, in_dim, num_experts=4, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for _ in range(num_experts)
        ])

        self.gate = nn.Linear(in_dim, num_experts)

    def forward(self, x):
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=-1)

        topk_vals, topk_idx = gate_weights.topk(self.top_k, dim=-1)

        outputs = torch.zeros_like(x)

        for k in range(self.top_k):
            expert_indices = topk_idx[:, k]
            expert_weights = topk_vals[:, k].unsqueeze(-1)

            expert_outputs = torch.zeros_like(x)

            for expert_id in range(self.num_experts):
                mask = (expert_indices == expert_id)
                if mask.sum() > 0:
                    selected_inputs = x[mask]
                    processed = self.experts[expert_id](selected_inputs)
                    expert_outputs[mask] = processed

            outputs += expert_weights * expert_outputs

        return outputs

# ==============================
# 🔹 Multi-Perspective Augmentation
# ==============================
class MultiPerspectiveAugment:
    def __init__(self, epsilon=0.01, shift_pixels=5):
        self.epsilon = epsilon
        self.shift_pixels = shift_pixels

    def perturbed_contrastive_view(self, image):
        noise = torch.randn_like(image) * 0.02
        return image + noise

    def spatially_shifted_view(self, image):
        dx = random.randint(-self.shift_pixels, self.shift_pixels)
        dy = random.randint(-self.shift_pixels, self.shift_pixels)
        return torch.roll(image, shifts=(dx, dy), dims=(2, 3))

    def adversarial_view(self, model, image):
        image = image.clone().detach().requires_grad_(True)

        features, logits = model(image)
        loss = logits.norm(p=2)  # stable proxy loss

        loss.backward()
        perturbation = self.epsilon * image.grad.sign()

        return (image + perturbation).detach()

    def apply(self, model, image):
        pcv = self.perturbed_contrastive_view(image)
        ssv = self.spatially_shifted_view(image)
        gav = self.adversarial_view(model, image)
        return pcv, ssv, gav

# ==============================
# 🔹 MAIN MODEL
# ==============================
class AttentionContrastiveModel(nn.Module):
    def __init__(self, feature_dim=128, num_classes=8, num_experts=4, top_k=2):
        super().__init__()

        self.encoder = models.densenet121(pretrained=True)

        # ❗ FIX: use RGB (3 channels)
        self.encoder.features.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.attention = MultiHeadSelfAttention(embed_dim=1024)
        self.moe = MixtureOfExperts(1024, num_experts=num_experts, top_k=top_k)

        # 🔹 Projection head (contrastive)
        self.projector = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, feature_dim)
        )

        # 🔹 Classification head
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.encoder.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        x = self.attention(x)
        x = self.moe(x)

        features = self.projector(x)
        logits = self.classifier(x)

        return features, logits

# ==============================
# 🔹 Contrastive Loss
# ==============================
class PerspectiveAwareContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, original, pcv, ssv, gav):
        loss_pcv = (1 - self.cosine_similarity(original, pcv)) / self.temperature
        loss_ssv = (1 - self.cosine_similarity(original, ssv)) / self.temperature
        loss_gav = (1 - self.cosine_similarity(original, gav)) / self.temperature

        return loss_pcv.mean() + loss_ssv.mean() + loss_gav.mean()

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
from sklearn.metrics import precision_score, recall_score, f1_score

# ==============================
# ⚙️ DEVICE
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# ⚙️ TRAINING CONFIG
# ==============================
num_epochs = 5
learning_rate = 0.0007
lambda_contrast = 0.5

# ==============================
# 🔹 LOSSES + AUGMENTATION
# ==============================
contrastive_criterion = PerspectiveAwareContrastiveLoss()
classification_criterion = torch.nn.CrossEntropyLoss()
augmentor = MultiPerspectiveAugment()

# ==============================
# 📊 METRICS
# ==============================
def compute_metrics(logits, labels):
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    acc = np.mean(preds == labels)
    prec = precision_score(labels, preds, average='macro', zero_division=0)
    rec = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)

    return acc, prec, rec, f1

# ==============================
# 🧠 TRAINING LOOP
# ==============================
local_models = []
all_client_metrics = {}

for client, dataloader in client_dataloaders.items():
    print(f"\n🚀 Training {client}")

    # Use lighter model config
    model = AttentionContrastiveModel(num_experts=2, top_k=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    client_losses = []
    client_accuracies = []
    client_precisions = []
    client_recalls = []
    client_f1_scores = []

    for epoch in range(num_epochs):
        total_loss = 0
        acc_list, prec_list, rec_list, f1_list = [], [], [], []

        for batch_idx, (images, labels) in enumerate(dataloader):
            print(f"{client} | Epoch {epoch+1} | Batch {batch_idx}")

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # =========================
            # 🔹 FORWARD 
            # =========================
            features, logits = model(images)

            # =========================
            # 🔹 LIGHTWEIGHT AUGMENTATIONS
            # =========================
            pcv = augmentor.perturbed_contrastive_view(images)
            ssv = augmentor.spatially_shifted_view(images)

            # ❌ Removed adversarial (too slow)
            gav = images.clone()

            # =========================
            # 🔹 FORWARD AUGMENTED
            # =========================
            f_pcv, _ = model(pcv)
            f_ssv, _ = model(ssv)
            f_gav, _ = model(gav)

            # Normalize
            features = F.normalize(features, dim=1)
            f_pcv = F.normalize(f_pcv, dim=1)
            f_ssv = F.normalize(f_ssv, dim=1)
            f_gav = F.normalize(f_gav, dim=1)

            # =========================
            # 🔹 LOSSES
            # =========================
            contrastive_loss = contrastive_criterion(features, f_pcv, f_ssv, f_gav)
            classification_loss = classification_criterion(logits, labels)

            loss = classification_loss + lambda_contrast * contrastive_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # =========================
            # 🔹 METRICS
            # =========================
            acc, prec, rec, f1 = compute_metrics(logits, labels)

            acc_list.append(acc)
            prec_list.append(prec)
            rec_list.append(rec)
            f1_list.append(f1)


        # =========================
        # 📊 EPOCH SUMMARY
        # =========================
        avg_loss = total_loss / len(acc_list)
        avg_acc = np.mean(acc_list)
        avg_prec = np.mean(prec_list)
        avg_rec = np.mean(rec_list)
        avg_f1 = np.mean(f1_list)

        print(f"\n📊 {client} Epoch [{epoch+1}/{num_epochs}]")
        print(f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | "
              f"Prec: {avg_prec:.4f} | Rec: {avg_rec:.4f} | F1: {avg_f1:.4f}")

        client_losses.append(avg_loss)
        client_accuracies.append(avg_acc)
        client_precisions.append(avg_prec)
        client_recalls.append(avg_rec)
        client_f1_scores.append(avg_f1)

    # Save metrics
    all_client_metrics[client] = {
        "losses": client_losses,
        "accuracies": client_accuracies,
        "precisions": client_precisions,
        "recalls": client_recalls,
        "f1_scores": client_f1_scores,
    }

    # Save model
    local_models.append(copy.deepcopy(model))

print("\n✅ Training completed for all clients")
