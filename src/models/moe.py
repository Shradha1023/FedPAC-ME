"""
moe.py

Mixture of Experts (MoE) layer with top-k gating mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MixtureOfExperts(nn.Module):
    """
    Implements an improved Mixture-of-Experts layer.
    """
    def __init__(self, in_dim, num_experts=4, top_k=2):
        super(MixtureOfExperts, self).__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(in_dim, num_experts)

    def forward(self, x):
        """
        x: tensor of shape (batch_size, in_dim)
        """
        batch_size = x.size(0)

        # Compute gating weights
        gate_logits = self.gate(x)  # (batch_size, num_experts)
        gate_weights = F.softmax(gate_logits, dim=-1)

        # Get top-k expert indices and weights
        topk_vals, topk_idx = gate_weights.topk(self.top_k, dim=-1)  # (batch_size, top_k)
        outputs = torch.zeros_like(x)

        for k in range(self.top_k):
            expert_indices = topk_idx[:, k]
            expert_weights = topk_vals[:, k].unsqueeze(-1)
            expert_outputs = torch.zeros_like(x)

            for expert_id in range(self.num_experts):
                mask = (expert_indices == expert_id)
                if mask.sum() > 0:
                    selected_inputs = x[mask]
                    processed = self.experts[expert_id](selected_inputs)
                    expert_outputs[mask] = processed

            outputs += expert_weights * expert_outputs

        return outputs
