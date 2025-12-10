"""
attention_contrastive_model.py

Attention-based contrastive feature extractor with Mixture-of-Experts (MoE) projection head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.attention import MultiHeadSelfAttention
from models.moe import MixtureOfExperts

class AttentionContrastiveModel(nn.Module):
    """
    Combines a DenseNet encoder, multi-head self-attention, 
    Mixture-of-Experts (MoE) layer, and projection head for feature extraction.
    """
    def __init__(self, feature_dim=128, num_experts=4, top_k=2):
        super().__init__()
        # Encoder backbone (DenseNet121)
        self.encoder = models.densenet121(pretrained=True)
        # Modify input layer to accept 2-channel input (FLAIR + T1ce)
        self.encoder.features.conv0 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Attention and Mixture-of-Experts modules
        self.attention = MultiHeadSelfAttention(embed_dim=1024)
        self.moe = MixtureOfExperts(1024, num_experts=num_experts, top_k=top_k)

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        # Extract features from encoder
        x = self.encoder.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        # Apply attention and Mixture-of-Experts
        x = self.attention(x)
        x = self.moe(x)

        # Project to feature_dim
        x = self.projector(x)
        return x
