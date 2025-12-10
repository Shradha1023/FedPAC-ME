"""
fedavg.py

Core algorithms:
- Weight aggregation
- FedAvg update rule
- Model evaluation with metrics (Accuracy, Precision, Recall, F1-score, Confidence, Dice Coefficient)
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import copy

# -----------------------------
# Federated Averaging (FedAvg)
# -----------------------------
def fed_avg(models):
    """
    Perform Federated Averaging on a list of local models.

    Args:
        models (list): List of PyTorch model instances (local models from clients).

    Returns:
        dict: State dict of the global model after aggregation.
    """
    global_dict = models[0].state_dict()

    # Initialize all weights to zero
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key]).float()

    # Sum up all model weights
    for model in models:
        local_dict = model.state_dict()
        for key in global_dict.keys():
            global_dict[key] += local_dict[key].float() / len(models)

    return global_dict


# -----------------------------
# Negative Sample Generation
# -----------------------------
def get_negative_samples(batch_size, dataset):
    """
    Generate negative samples from the dataset.

    Args:
        batch_size (int): Number of negative samples.
        dataset (Dataset): PyTorch dataset object.

    Returns:
        torch.Tensor: Batch of negative samples (images only).
    """
    indices = torch.randint(0, len(dataset), (batch_size,))
    negative_samples = torch.stack([dataset[i][0] for i in indices])
    return negative_samples


# -----------------------------
# Metric Computation
# -----------------------------
def compute_metrics(z_i, z_j, z_j_neg, threshold=0.99):
    """
    Compute Accuracy, Precision, Recall, F1-score, Confidence, and Dice Coefficient
    for positive and negative pairs using cosine similarity.

    Args:
        z_i (torch.Tensor): Anchor features.
        z_j (torch.Tensor): Positive features.
        z_j_neg (torch.Tensor): Negative features.
        threshold (float): Cosine similarity threshold for positive prediction.

    Returns:
        Tuple: accuracy, precision, recall, f1, confidence, dice_coeff, total_samples
    """
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    z_j_neg = F.normalize(z_j_neg, dim=1)

    # Compute cosine similarities
    similarities_pos = torch.sum(z_i * z_j, dim=1)
    similarities_neg = torch.sum(z_i * z_j_neg, dim=1)

    # Predictions
    predictions_pos = (similarities_pos > threshold).int().cpu().numpy()
    predictions_neg = (similarities_neg > threshold).int().cpu().numpy()

    # Ground truth
    targets_pos = np.ones_like(predictions_pos)
    targets_neg = np.zeros_like(predictions_neg)

    # Concatenate
    predictions = np.concatenate((predictions_pos, predictions_neg))
    targets = np.concatenate((targets_pos, targets_neg))

    # Metrics
    accuracy = np.mean(predictions == targets)
    precision = precision_score(targets, predictions, zero_division=0)
    recall = recall_score(targets, predictions, zero_division=0)
    f1 = f1_score(targets, predictions, zero_division=0)
    confidence = similarities_pos.mean().item()
    dice_coeff = (2 * np.sum(predictions * targets)) / (np.sum(predictions) + np.sum(targets) + 1e-8)

    return accuracy, precision, recall, f1, confidence, dice_coeff, len(targets)


# -----------------------------
# Model Evaluation
# -----------------------------
def evaluate_model(model, dataloader, criterion, device, threshold=0.99):
    """
    Evaluate a model on a dataloader and compute validation metrics.

    Args:
        model (torch.nn.Module): PyTorch model.
        dataloader (DataLoader): Validation dataloader.
        criterion (callable): Loss function.
        device (torch.device): Computation device (CPU/GPU).
        threshold (float): Cosine similarity threshold for metrics.

    Returns:
        Tuple: avg_loss, avg_accuracy, avg_precision, avg_recall, avg_f1, avg_confidence, avg_dice_coeff
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_precision, all_recall, all_f1, all_confidence, all_dice_coeff = [], [], [], [], []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            negative_images = get_negative_samples(len(images), dataloader.dataset).to(device)

            # Extract features
            z_i = model(images)
            z_j = model(images)
            z_j_neg = model(negative_images)

            # Dummy global values for criterion if needed
            ssv = z_i
            gav = z_j.mean(dim=0, keepdim=True).repeat(z_j.size(0), 1)

            loss = criterion(z_i, z_j, ssv, gav)
            total_loss += loss.item()

            metrics = compute_metrics(z_i, z_j, z_j_neg, threshold)
            accuracy, precision, recall, f1, confidence, dice_coeff, total = metrics

            total_correct += accuracy * total
            total_samples += total
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)
            all_confidence.append(confidence)
            all_dice_coeff.append(dice_coeff)

    # Average metrics
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_correct / total_samples
    avg_precision = np.mean(all_precision)
    avg_recall = np.mean(all_recall)
    avg_f1 = np.mean(all_f1)
    avg_confidence = np.mean(all_confidence)
    avg_dice_coeff = np.mean(all_dice_coeff)

    print(f"Validation -> Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}, Confidence: {avg_confidence:.4f}, Dice: {avg_dice_coeff:.4f}")
    return avg_loss, avg_accuracy, avg_precision, avg_recall, avg_f1, avg_confidence, avg_dice_coeff
