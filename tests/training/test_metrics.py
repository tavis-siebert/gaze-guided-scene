"""
Tests for the training metrics module.
"""

import numpy as np
import torch

from gazegraph.training.evaluation.metrics import (
    accuracy,
    top_k_accuracy,
    compute_per_class_metrics,
    compute_overall_metrics,
    mAP,
)


class TestMetrics:
    """Test class for training metrics functions."""

    def test_accuracy_basic(self):
        """Test basic accuracy computation."""
        pred = torch.tensor([0, 1, 2, 1])
        targets = torch.tensor([0, 1, 2, 2])
        acc = accuracy(pred, targets)
        assert acc == 3  # 3 out of 4 correct

    def test_top_k_accuracy_torch(self):
        """Test top-k accuracy with torch tensors."""
        # Create predictions where class 2 has highest score, class 1 second highest
        predictions = torch.tensor(
            [
                [0.1, 0.3, 0.6],  # top-1: 2 ✓
                [
                    0.4,
                    0.2,
                    0.4,
                ],  # top-1: 0 ✓ (torch.topk picks first occurrence for ties)
                [0.2, 0.7, 0.1],  # top-1: 1 ✓
            ]
        )
        targets = torch.tensor([2, 0, 1])

        # All predictions are actually correct
        top1_acc = top_k_accuracy(predictions, targets, k=1)
        assert abs(top1_acc - 1.0) < 0.01

        # Top-2 accuracy: all should be correct within top-2
        top2_acc = top_k_accuracy(predictions, targets, k=2)
        assert abs(top2_acc - 1.0) < 0.01

    def test_top_k_accuracy_numpy(self):
        """Test top-k accuracy with numpy arrays."""
        predictions = np.array(
            [
                [0.1, 0.3, 0.6],  # top-1: 2 ✓
                [
                    0.4,
                    0.2,
                    0.4,
                ],  # top-1: 0 ✓ (numpy.argsort picks last occurrence for ties)
                [0.2, 0.7, 0.1],  # top-1: 1 ✓
            ]
        )
        targets = np.array([2, 0, 1])

        # Test with a case where not all are correct
        predictions_mixed = np.array(
            [
                [0.1, 0.3, 0.6],  # top-1: 2 ✓
                [0.4, 0.2, 0.4],  # top-1: 2 ✗ (target is 0)
                [0.2, 0.7, 0.1],  # top-1: 1 ✓
            ]
        )

        top1_acc = top_k_accuracy(predictions_mixed, targets, k=1)
        assert abs(top1_acc - 2 / 3) < 0.01  # 2 out of 3 correct

        top2_acc = top_k_accuracy(predictions_mixed, targets, k=2)
        assert abs(top2_acc - 1.0) < 0.01  # all correct within top-2

    def test_compute_per_class_metrics(self):
        """Test per-class metrics computation."""
        # predictions: [0, 1, 2, 1, 0, 2]
        # targets:     [0, 1, 2, 2, 0, 1]
        predictions = np.array([0, 1, 2, 1, 0, 2])
        targets = np.array([0, 1, 2, 2, 0, 1])
        num_classes = 3

        metrics = compute_per_class_metrics(predictions, targets, num_classes)

        # Class 0: targets=[0,0], predictions=[0,0] -> TP=2, FP=0, FN=0
        assert metrics[0]["precision"] == 1.0
        assert metrics[0]["recall"] == 1.0
        assert metrics[0]["f1"] == 1.0
        assert metrics[0]["support"] == 2

        # Class 1: targets=[1,1], predictions=[1,1] -> TP=1, FP=1, FN=1
        # Precision = TP/(TP+FP) = 1/(1+1) = 0.5
        # Recall = TP/(TP+FN) = 1/(1+1) = 0.5
        assert metrics[1]["precision"] == 0.5
        assert metrics[1]["recall"] == 0.5
        assert abs(metrics[1]["f1"] - 0.5) < 0.01

        # Class 2: targets=[2,2], predictions=[2,2] -> TP=1, FP=1, FN=1
        # Precision = TP/(TP+FP) = 1/(1+1) = 0.5
        # Recall = TP/(TP+FN) = 1/(1+1) = 0.5
        assert metrics[2]["precision"] == 0.5
        assert metrics[2]["recall"] == 0.5
        assert abs(metrics[2]["f1"] - 0.5) < 0.01

    def test_compute_overall_metrics(self):
        """Test overall metrics computation."""
        # Create simple test data
        all_outputs = [
            np.array([[0.1, 0.9], [0.8, 0.2]]),  # batch 1
            np.array([[0.3, 0.7]]),  # batch 2
        ]
        all_targets = [
            np.array([1, 0]),  # batch 1
            np.array([1]),  # batch 2
        ]
        all_predictions = [
            np.array([1, 0]),  # batch 1
            np.array([1]),  # batch 2
        ]
        num_classes = 2

        accuracy_val, top5_acc, class_metrics, pred_dist = compute_overall_metrics(
            all_outputs, all_targets, all_predictions, num_classes
        )

        # All predictions are correct
        assert accuracy_val == 1.0
        assert top5_acc == 1.0  # With only 2 classes, top-5 is always 1.0
        assert len(class_metrics) == 2
        assert pred_dist[0] == 1  # One prediction for class 0
        assert pred_dist[1] == 2  # Two predictions for class 1

    def test_mAP_multilabel(self):
        """Test mAP computation for multi-label classification."""
        # Create multi-label test data
        predictions = np.array(
            [
                [0.9, 0.1, 0.8],  # High confidence for classes 0 and 2
                [0.2, 0.9, 0.1],  # High confidence for class 1
                [0.8, 0.3, 0.9],  # High confidence for classes 0 and 2
            ]
        )
        targets = np.array(
            [
                [1, 0, 1],  # True labels: classes 0 and 2
                [0, 1, 0],  # True labels: class 1
                [1, 0, 1],  # True labels: classes 0 and 2
            ]
        )

        map_score = mAP(predictions, targets)

        # Should be high since predictions match targets well
        assert map_score > 0.8
        assert map_score <= 1.0

    def test_mAP_no_positive_samples(self):
        """Test mAP when some classes have no positive samples."""
        predictions = np.array(
            [
                [0.9, 0.1],
                [0.2, 0.9],
            ]
        )
        targets = np.array(
            [
                [1, 0],  # Only class 0 has positive samples
                [1, 0],
            ]
        )

        map_score = mAP(predictions, targets)

        # Should handle missing classes gracefully
        assert 0.0 <= map_score <= 1.0

    def test_mAP_empty_targets(self):
        """Test mAP with no positive samples at all."""
        predictions = np.array(
            [
                [0.9, 0.1],
                [0.2, 0.9],
            ]
        )
        targets = np.array(
            [
                [0, 0],
                [0, 0],
            ]
        )

        map_score = mAP(predictions, targets)

        # Should return 0.0 when no positive samples
        assert map_score == 0.0
