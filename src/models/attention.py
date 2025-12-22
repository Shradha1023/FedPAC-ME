"""
attention.py

Multi-Head Self-Attention module used in FedPAC-ME.
Applies channel-wise self-attention over global feature embeddings
(B x D) before Mixture-of-Experts routing.
"""

import torch
import torch.nn as nn
import math


class MultiHeadSelfAttention(nn.Module):
    """
    Channel-wise Multi-Head Self-Attention with residual connection
    and layer normalization.

    Args:
        embed_dim (int): Feature dimension (D)
        num_heads (int): Number of attention heads (H)
    """

    def __init__(self, embed_dim=1024, num_heads=4):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, D) — global feature vector after GAP
        """
        B, D = x.shape

        # Linear projections
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # Split into heads: (B, H, head_dim)
        q = q.view(B, self.num_heads, self.head_dim)
        k = k.view(B, self.num_heads, self.head_dim)
        v = v.view(B, self.num_heads, self.head_dim)

        # Scaled dot-product attention (channel-wise)
        attn = (q * k) / math.sqrt(self.head_dim)   # (B, H, head_dim)
        attn = torch.softmax(attn, dim=-1)

        # Weighted values
        out = attn * v                              # (B, H, head_dim)

        # Merge heads
        out = out.view(B, D)
        out = self.out(out)

        # Residual connection + LayerNorm
        return self.norm(x + out)
