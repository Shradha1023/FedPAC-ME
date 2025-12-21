"""
attention.py

Multi-Head Self-Attention module used in FedPAC-ME.
Applies self-attention over feature embeddings to model
channel-wise dependencies before Mixture-of-Experts routing.
"""

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention layer with residual connection
    and layer normalization.

    Args:
        embed_dim (int): Feature embedding dimension
        num_heads (int): Number of attention heads
    """

    def __init__(self, embed_dim=1024, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, D)
        B, D = x.shape

        qkv = self.qkv(x)                     # (B, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, self.num_heads, self.head_dim)
        k = k.view(B, self.num_heads, self.head_dim)
        v = v.view(B, self.num_heads, self.head_dim)

        attn = torch.softmax(
            (q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim),
            dim=-1
        )

        out = (attn @ v).reshape(B, D)
        out = self.out(out)

        return self.norm(x + out)
