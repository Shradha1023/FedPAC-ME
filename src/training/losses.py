# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Dice Loss
# -------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2 * torch.sum(probs * targets) + self.smooth
        den = torch.sum(probs) + torch.sum(targets) + self.smooth
        return 1 - num / den

# -------------------------------
# Cross-Entropy Loss
# -------------------------------
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return self.ce(logits, targets)

# -------------------------------
# Contrastive Loss
# -------------------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# -------------------------------
# PACL Loss (Perspective-Aware Contrastive Learning)
# -------------------------------
class PACLLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(PACLLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: tensor of shape [batch_size, feature_dim]
        labels: tensor of shape [batch_size]
        """
        batch_size = features.size(0)
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        exp_sim = torch.exp(similarity_matrix) * (~torch.eye(batch_size, dtype=bool, device=features.device))
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        loss = - (mask * log_prob).sum() / mask.sum()
        return loss
