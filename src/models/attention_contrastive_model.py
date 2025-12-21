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
    def __init__(self, embed_dim=1024, feature_dim=128, num_experts=4, top_k=2):
        super().__init__()
        # Encoder backbone (DenseNet121)
        self.encoder = models.densenet121(pretrained=True)
        # Modify input layer to accept 2-channel input (FLAIR + T1ce)
        self.encoder.features.conv0 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Attention and Mixture-of-Experts modules
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.moe = MixtureOfExperts(embed_dim, num_experts, top_k)

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, feature_dim)
        )

    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)   # (B,1024)
        return x

    def forward(self, x):
        x = self.encode(x)

        # MHSA (Seq len = 1)
        x_seq = x.unsqueeze(1)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        x = self.norm(x + attn_out.squeeze(1))

        x = self.moe(x)
        z = self.projector(x)
        return F.normalize(z, dim=1)
