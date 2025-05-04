
import numpy as np
from sklearn.metrics import average_precision_score

def accuracy(pred, targets):
    acc = (pred == targets).sum().item()
    return acc

def confusion_matrix(pred, targets):
    tp = (pred * targets).sum().item()
    fn = ((1 - pred) * targets).sum().item()
    fp = (pred * (1 - targets)).sum().item()
    tn = ((1 - pred) * (1 - targets)).sum().item()

    return np.array([[tp, fp], [fn, tn]])

def mAP(all_predictions, all_targets):
    num_classes = all_targets.shape[1]
    ap_per_class = []

    for c in range(num_classes):
        if np.sum(all_targets[:, c]) > 0:  # Ignore classes without positive samples
            ap = average_precision_score(all_targets[:, c], all_predictions[:, c])
            ap_per_class.append(ap)

    # Compute mean Average Precision (mAP)
    mAP = np.mean(ap_per_class)

    return mAP