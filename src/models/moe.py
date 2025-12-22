"""
moe.py

Mixture of Experts (MoE) with soft top-k gating.
Aligned with FedPAC-ME methodology.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixtureOfExperts(nn.Module):
    def __init__(self, in_dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ) for _ in range(num_experts)
        ])

        self.gate = nn.Linear(in_dim, num_experts)

    def forward(self, x):
        """
        x: (B, D)
        """
        B, D = x.shape

        # Gating logits
        logits = self.gate(x)                    # (B, E)

        # Top-k gating (before softmax)
        topk_logits, topk_idx = logits.topk(self.top_k, dim=-1)

        # Normalize top-k gates
        topk_weights = F.softmax(topk_logits, dim=-1)  # (B, K)

        # Compute all expert outputs
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )  # (B, E, D)

        # Select top-k expert outputs
        selected = torch.gather(
            expert_outputs,
            dim=1,
            index=topk_idx.unsqueeze(-1).expand(-1, -1, D)
        )  # (B, K, D)

        # Weighted sum
        out = torch.sum(selected * topk_weights.unsqueeze(-1), dim=1)

        return out
