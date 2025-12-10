"""
aggregation.py

Handles:
- Weighted aggregation
- Aggregation with expert weighting (for ME)
- Perspective-aware proto-aggregation (if used)
"""

import numpy as np

def weighted_aggregate(weights_list, weights=None):
    """
    Weighted aggregation of client weights
    """
    if weights is None:
        weights = [1/len(weights_list)] * len(weights_list)

    agg_weights = sum(w * weights[i] for i, w in enumerate(weights))
    return agg_weights

def expert_aggregate(client_weights, expert_scores):
    """
    Aggregation with expert weighting
    """
    total_score = sum(expert_scores)
    normalized_scores = [s / total_score for s in expert_scores]
    return weighted_aggregate(client_weights, normalized_scores)
