# metrics.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

# -------------------------------
# Dice Coefficient
# -------------------------------
def dice_coefficient(preds, targets, smooth=1.0):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    num = 2 * torch.sum(preds * targets) + smooth
    den = torch.sum(preds) + torch.sum(targets) + smooth
    return (num / den).item()

# -------------------------------
# Precision, Recall, F1-score, IoU
# -------------------------------
def compute_metrics(preds, targets):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).int()
    targets = targets.int()
    
    precision = precision_score(targets.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), zero_division=0)
    recall = recall_score(targets.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), zero_division=0)
    f1 = f1_score(targets.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), zero_division=0)
    iou = jaccard_score(targets.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), zero_division=0)
    dice = dice_coefficient(preds, targets)
    
    return precision, recall, f1, iou, dice

# -------------------------------
# Client-wise Metrics Visualization
# -------------------------------
def plot_client_metrics(all_client_metrics):
    metrics_to_plot = {
        "Accuracy": "accuracies",
        "Loss": "losses",
        "Precision": "precisions",
        "Recall": "recalls",
        "F1 Score": "f1_scores",
        "Dice Coefficient": "dice_coefficients",
        "Confidence": "confidences"
    }

    # Plot all metrics for all clients
    plt.figure(figsize=(18, 12))
    rows = 3
    cols = 3
    for i, (metric_name, key) in enumerate(metrics_to_plot.items(), start=1):
        plt.subplot(rows, cols, i)
        for client_id, metrics in all_client_metrics.items():
            values = metrics.get(key, [])
            epochs = list(range(1, len(values) + 1))
            plt.plot(epochs, values, label=client_id)
        plt.title(f"{metric_name} Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.legend(fontsize="small")
        plt.grid(True)

    plt.tight_layout()
    plt.suptitle("Client-wise Performance Metrics Over Epochs", fontsize=16, y=1.02)
    plt.show()

# -------------------------------
# Create client metrics tables
# -------------------------------
def print_client_metrics_table(all_client_metrics):
    for client_id, metrics in all_client_metrics.items():
        num_epochs = len(metrics['losses'])
        data = {
            "Epoch": list(range(1, num_epochs + 1)),
            "Loss": metrics["losses"],
            "Accuracy": metrics["accuracies"],
            "Precision": metrics["precisions"],
            "Recall": metrics["recalls"],
            "F1-Score": metrics["f1_scores"],
            "Confidence": met
