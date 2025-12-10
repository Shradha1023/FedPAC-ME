"""
loss.py

Perspective-Aware Contrastive Loss (PACL) for multi-perspective MRI feature learning.
"""

import torch
import torch.nn as nn

class PerspectiveAwareContrastiveLoss(nn.Module):
    """
    Computes contrastive loss between different perspectives of the same image:
    - perturbed contrastive view (PCV)
    - spatially shifted view (SSV)
    - adversarially generated view (GAV)
    """
    def __init__(self, temperature=0.5):
        super(PerspectiveAwareContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, original, pcv, ssv, gav):
        """
        Args:
            original: Original feature embeddings (batch_size, feature_dim)
            pcv: Perturbed contrastive view embeddings
            ssv: Spatially shifted view embeddings
            gav: Adversarially generated view embeddings

        Returns:
            Scalar contrastive loss
        """
        loss_pcv = (1 - self.cosine_similarity(original, pcv)) / self.temperature
        loss_ssv = (1 - self.cosine_similarity(original, ssv)) / self.temperature
        loss_gav = (1 - self.cosine_similarity(original, gav)) / self.temperature

        return loss_pcv.mean() + loss_ssv.mean() + loss_gav.mean()
