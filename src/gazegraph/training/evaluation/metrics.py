import numpy as np
import torch
from sklearn.metrics import average_precision_score


def accuracy(pred, targets):
    acc = (pred == targets).sum().item()
    return acc


def top_k_accuracy(predictions, targets, k=5):
    """Calculate top-k accuracy for multi-class classification.

    Args:
        predictions: Predicted logits or probabilities [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        k: Number of top predictions to consider

    Returns:
        Top-k accuracy as a float
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Get top-k predictions
    top_k_preds = np.argsort(predictions, axis=1)[:, -k:]

    # Check if true label is in top-k predictions
    correct = 0
    for i, target in enumerate(targets):
        if target in top_k_preds[i]:
            correct += 1

    return correct / len(targets)


def confusion_matrix(pred, targets):
    tp = (pred * targets).sum().item()
    fn = ((1 - pred) * targets).sum().item()
    fp = (pred * (1 - targets)).sum().item()
    tn = ((1 - pred) * (1 - targets)).sum().item()

    return np.array([[tp, fp], [fn, tn]])


def mAP(all_predictions, all_targets):
    """Calculate mean Average Precision for multi-label classification.

    Args:
        all_predictions: Predicted scores [batch_size, num_classes]
        all_targets: Ground truth binary labels [batch_size, num_classes]

    Returns:
        Mean Average Precision as a float
    """
    num_classes = all_targets.shape[1]
    ap_per_class = []

    for c in range(num_classes):
        if np.sum(all_targets[:, c]) > 0:  # Ignore classes without positive samples
            ap = average_precision_score(all_targets[:, c], all_predictions[:, c])
            ap_per_class.append(ap)

    # Compute mean Average Precision (mAP)
    return np.mean(ap_per_class) if ap_per_class else 0.0
