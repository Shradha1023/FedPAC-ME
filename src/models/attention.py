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

    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Tensor: Attention-refined features with same shape as input
        """
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(x + attn_output)  # Residual connection
        return x
