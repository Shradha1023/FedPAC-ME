"""
loss.py

Perspective-Aware Contrastive Loss (PACL)
Aligned with FedPAC-ME methodology.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerspectiveAwareContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def _info_nce(self, anchor, positive):
        """
        anchor, positive: (B, D)
        """
        B = anchor.size(0)

        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)

        # Similarity matrix (B x B)
        logits = torch.matmul(anchor, positive.T) / self.temperature

        labels = torch.arange(B, device=anchor.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    def forward(self, original, pcv, ssv, gav):
        """
        Multi-perspective InfoNCE loss.
        """
        loss_pcv = self._info_nce(original, pcv)
        loss_ssv = self._info_nce(original, ssv)
        loss_gav = self._info_nce(original, gav)

        return loss_pcv + loss_ssv + loss_gav
