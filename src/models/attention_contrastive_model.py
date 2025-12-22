"""
attention_contrastive_model.py

Attention-based contrastive feature extractor aligned with FedPAC-ME methodology.
Applies channel-wise MHSA over global feature vectors before MoE routing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.attention import MultiHeadSelfAttention
from models.moe import MixtureOfExperts


class AttentionContrastiveModel(nn.Module):
    def __init__(self, embed_dim=1024, feature_dim=128,
                 num_experts=4, top_k=2):
        super().__init__()

        # DenseNet backbone (feature extractor ONLY)
        backbone = models.densenet121(pretrained=True)
        backbone.features.conv0 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.encoder = backbone.features

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Channel-wise MHSA (methodology-aligned)
        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=4
        )

        # Mixture of Experts
        self.moe = MixtureOfExperts(embed_dim, num_experts, top_k)

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.LayerNorm(512),
            nn.Linear(512, feature_dim)
        )

    def encode(self, x):
        """
        x: (B, C, H, W)
        """
        x = self.encoder(x)                    # (B, 1024, H', W')
        x = self.gap(x).flatten(1)             # (B, 1024)
        return x

    def forward(self, x):
        # Feature extraction
        x = self.encode(x)                     # X_flat

        # MHSA refinement (channel-wise)
        x = self.attention(x)                  # X_attn

        # Mixture of Experts routing
        x = self.moe(x)                        # X_moe

        # Projection for contrastive learning
        z = self.projector(x)
        return F.normalize(z, dim=1)
