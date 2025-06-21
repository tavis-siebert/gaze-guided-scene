import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix as sklearn_confusion_matrix,
)


def accuracy(pred, targets):
    """Calculate accuracy for predictions."""
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
    """Calculate confusion matrix for binary classification."""
    tp = (pred * targets).sum().item()
    fn = ((1 - pred) * targets).sum().item()
    fp = (pred * (1 - targets)).sum().item()
    tn = ((1 - pred) * (1 - targets)).sum().item()

    return np.array([[tp, fp], [fn, tn]])


def multiclass_confusion_matrix(predictions, targets, num_classes):
    """Calculate confusion matrix for multi-class classification using sklearn.

    Args:
        predictions: Predicted class indices [batch_size]
        targets: Ground truth labels [batch_size]
        num_classes: Number of classes

    Returns:
        Confusion matrix as numpy array [num_classes, num_classes]
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    return sklearn_confusion_matrix(targets, predictions, labels=range(num_classes))


def compute_per_class_metrics(predictions, targets, num_classes):
    """Compute per-class precision, recall, F1, and support metrics.

    Args:
        predictions: Predicted class indices [batch_size]
        targets: Ground truth labels [batch_size]
        num_classes: Number of classes

    Returns:
        Dictionary with per-class metrics
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Initialize per-class metrics
    class_tp = np.zeros(num_classes)
    class_fp = np.zeros(num_classes)
    class_fn = np.zeros(num_classes)
    class_support = np.zeros(num_classes)

    # Update per-class metrics
    for i in range(len(targets)):
        true_class = targets[i]
        pred_class = predictions[i]

        class_support[true_class] += 1

        if true_class == pred_class:
            class_tp[true_class] += 1
        else:
            class_fn[true_class] += 1
            class_fp[pred_class] += 1

    # Calculate per-class metrics
    class_metrics = {}
    for c in range(num_classes):
        if class_support[c] > 0:
            recall = class_tp[c] / class_support[c] if class_support[c] > 0 else 0
            precision = (
                class_tp[c] / (class_tp[c] + class_fp[c])
                if (class_tp[c] + class_fp[c]) > 0
                else 0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            class_metrics[c] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": class_support[c],
            }

    return class_metrics


def compute_overall_metrics(all_outputs, all_targets, all_predictions, num_classes):
    """Compute overall accuracy and top-k accuracy metrics.

    Args:
        all_outputs: List of output tensors from model
        all_targets: List of target tensors
        all_predictions: List of prediction tensors
        num_classes: Number of classes

    Returns:
        Tuple of (accuracy, top5_accuracy, per_class_metrics, prediction_distribution)
    """
    # Flatten arrays
    all_targets_flat = np.concatenate(all_targets)
    all_outputs_flat = np.vstack(all_outputs)
    all_predictions_flat = np.concatenate(all_predictions)

    # Calculate accuracy
    total_accuracy = (all_predictions_flat == all_targets_flat).sum() / len(
        all_targets_flat
    )

    # Calculate top-5 accuracy
    top5_acc = top_k_accuracy(all_outputs_flat, all_targets_flat, k=5)

    # Calculate per-class metrics
    class_metrics = compute_per_class_metrics(
        all_predictions_flat, all_targets_flat, num_classes
    )

    # Calculate prediction distribution
    pred_counts = np.bincount(all_predictions_flat, minlength=num_classes)

    return total_accuracy, top5_acc, class_metrics, pred_counts


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
