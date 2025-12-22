"""
attention.py

Multi-Head Self-Attention module used in FedPAC-ME.
Applies self-attention over feature embeddings to model
channel-wise dependencies before Mixture-of-Experts routing.
"""
import torch
import torch.nn as nn
import math


class MultiHeadSelfAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention with residual connection
    and layer normalization.

    Args:
        embed_dim (int): Embedding dimension (D)
        num_heads (int): Number of attention heads (H)
    """

    def __init__(self, embed_dim=1024, num_heads=4):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape

        qkv = self.qkv(x)                          # (B, N, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, H, N, head_dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)         # (B, H, N, N)

        # Weighted sum
        out = attn @ v                             # (B, H, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        out = self.out(out)

        # Residual + LayerNorm
        return self.norm(x + out)

