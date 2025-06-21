"""
Tests for object detection configuration parameters.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from gazegraph.graph.object_detection import ObjectDetector, Detection
from gazegraph.config.config_utils import DotDict


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = DotDict(
        {
            "models": {
                "yolo_world": {
                    "backend": "onnx",
                    "onnx": {"conf_threshold": 0.15, "iou_threshold": 0.5},
                }
            },
            "graph": {
                "fixated_object_detection": {
                    "min_fixation_frame_threshold": 4,
                    "min_fixation_frame_ratio": 0.5,
                    "bbox_margin": 10,
                    "weights": {
                        "duration": 1.0,
                        "bbox_stability": 1.0,
                        "gaze_proximity": 1.0,
                        "confidence": 1.0,
                    },
                    "thresholds": {
                        "bbox_stability": 0.5,
                        "gaze_proximity": 0.3,
                        "confidence": 0.6,
                    },
                }
            },
            "processing": {"n_cores": 2},
        }
    )
    return config


@pytest.fixture
def mock_detector(mock_config):
    """Create a mock detector with mocked model."""
    with patch("gazegraph.graph.object_detection.YOLOWorldModel") as mock_model_class:
        # Create a mock model instance
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        # Setup mock model behavior
        mock_model.predict.return_value = []

        # Create detector with mock model
        detector = ObjectDetector(
            model_path=Path("path/to/model"),
            classes=["cup", "bowl", "spoon"],
            config=mock_config,
        )

        # Return the detector with mocked model
        return detector


class TestObjectDetectorConfig:
    """Test that object detector correctly uses configuration parameters."""

    def test_min_fixation_frame_threshold(self, mock_detector):
        """Test that min_fixation_frame_threshold is correctly applied."""
        # Set up test data with 3 frames of fixation (below threshold of 4)
        mock_detector.all_detections = [
            Detection(
                bbox=(10, 10, 20, 20),
                class_name="cup",
                score=0.8,
                class_id=0,
                frame_idx=0,
                is_fixated=True,
            ),
            Detection(
                bbox=(12, 12, 20, 20),
                class_name="cup",
                score=0.8,
                class_id=0,
                frame_idx=1,
                is_fixated=True,
            ),
            Detection(
                bbox=(14, 14, 20, 20),
                class_name="cup",
                score=0.8,
                class_id=0,
                frame_idx=2,
                is_fixated=True,
            ),
        ]

        # Set up other required state
        mock_detector.gaze_points = [(15, 15, 0), (15, 15, 1), (15, 15, 2)]

        # Mock the helper methods to return good scores
        with (
            patch.object(ObjectDetector, "_compute_mean_iou", return_value=0.8),
            patch.object(ObjectDetector, "_compute_geometric_mean", return_value=0.8),
            patch.object(
                ObjectDetector, "_compute_mean_gaze_distance", return_value=0.1
            ),
        ):
            # Compute scores
            scores = mock_detector._compute_fixation_scores()

            # Should be empty because min_fixation_frame_threshold is 4
            assert len(scores) == 0

    def test_min_fixation_frame_ratio(self, mock_detector):
        """Test that min_fixation_frame_ratio is correctly applied."""
        # Set up test data with 4 frames total, but only 1 frame with cup fixation (ratio 0.25, below threshold of 0.5)
        mock_detector.all_detections = [
            # Cup fixated in 1 frame
            Detection(
                bbox=(10, 10, 20, 20),
                class_name="cup",
                score=0.8,
                class_id=0,
                frame_idx=0,
                is_fixated=True,
            ),
            # Bowl fixated in 3 frames
            Detection(
                bbox=(50, 50, 20, 20),
                class_name="bowl",
                score=0.8,
                class_id=1,
                frame_idx=1,
                is_fixated=True,
            ),
            Detection(
                bbox=(52, 52, 20, 20),
                class_name="bowl",
                score=0.8,
                class_id=1,
                frame_idx=2,
                is_fixated=True,
            ),
            Detection(
                bbox=(54, 54, 20, 20),
                class_name="bowl",
                score=0.8,
                class_id=1,
                frame_idx=3,
                is_fixated=True,
            ),
        ]

        # Set up other required state
        mock_detector.gaze_points = [(15, 15, 0), (55, 55, 1), (55, 55, 2), (55, 55, 3)]

        # Override the min_fixation_frame_threshold to ensure we're only testing ratio
        mock_detector.min_fixation_frame_threshold = 1

        # Mock the helper methods to return good scores
        with (
            patch.object(ObjectDetector, "_compute_mean_iou", return_value=0.8),
            patch.object(ObjectDetector, "_compute_geometric_mean", return_value=0.8),
            patch.object(
                ObjectDetector, "_compute_mean_gaze_distance", return_value=0.1
            ),
        ):
            # Compute scores
            scores = mock_detector._compute_fixation_scores()

            # Should only contain bowl (ratio 0.75, above threshold of 0.5)
            # Cup should be filtered out (ratio 0.25, below threshold of 0.5)
            assert len(scores) == 1
            assert "bowl" in scores
            assert "cup" not in scores

    def test_confidence_threshold(self, mock_detector):
        """Test that confidence_threshold is correctly applied."""
        # Set up test data with two objects, one with high confidence, one with low
        mock_detector.all_detections = [
            # Cup with high confidence (0.9)
            Detection(
                bbox=(10, 10, 20, 20),
                class_name="cup",
                score=0.9,
                class_id=0,
                frame_idx=0,
                is_fixated=True,
            ),
            Detection(
                bbox=(12, 12, 20, 20),
                class_name="cup",
                score=0.9,
                class_id=0,
                frame_idx=1,
                is_fixated=True,
            ),
            Detection(
                bbox=(14, 14, 20, 20),
                class_name="cup",
                score=0.9,
                class_id=0,
                frame_idx=2,
                is_fixated=True,
            ),
            Detection(
                bbox=(16, 16, 20, 20),
                class_name="cup",
                score=0.9,
                class_id=0,
                frame_idx=3,
                is_fixated=True,
            ),
            # Bowl with low confidence (0.5)
            Detection(
                bbox=(50, 50, 20, 20),
                class_name="bowl",
                score=0.5,
                class_id=1,
                frame_idx=0,
                is_fixated=True,
            ),
            Detection(
                bbox=(52, 52, 20, 20),
                class_name="bowl",
                score=0.5,
                class_id=1,
                frame_idx=1,
                is_fixated=True,
            ),
            Detection(
                bbox=(54, 54, 20, 20),
                class_name="bowl",
                score=0.5,
                class_id=1,
                frame_idx=2,
                is_fixated=True,
            ),
            Detection(
                bbox=(56, 56, 20, 20),
                class_name="bowl",
                score=0.5,
                class_id=1,
                frame_idx=3,
                is_fixated=True,
            ),
        ]

        # Set up other required state
        mock_detector.gaze_points = [(15, 15, 0), (15, 15, 1), (15, 15, 2), (15, 15, 3)]

        # Override thresholds to ensure we're only testing confidence
        mock_detector.min_fixation_frame_threshold = 1
        mock_detector.min_fixation_frame_ratio = 0.1
        mock_detector.bbox_stability_threshold = 0.0
        mock_detector.gaze_proximity_threshold = 0.0

        # Use real geometric mean but mock other methods
        with (
            patch.object(ObjectDetector, "_compute_mean_iou", return_value=0.8),
            patch.object(
                ObjectDetector, "_compute_mean_gaze_distance", return_value=0.1
            ),
        ):
            # Compute scores
            scores = mock_detector._compute_fixation_scores()

            # Should only contain cup (confidence 0.9, above threshold of 0.6)
            # Bowl should be filtered out (confidence 0.5, below threshold of 0.6)
            assert len(scores) == 1
            assert "cup" in scores
            assert "bowl" not in scores

    def test_bbox_stability_threshold(self, mock_detector):
        """Test that bbox_stability_threshold is correctly applied."""
        # Set up test data with two objects, one with stable bboxes, one with unstable
        mock_detector.all_detections = [
            # Cup with stable bboxes
            Detection(
                bbox=(10, 10, 20, 20),
                class_name="cup",
                score=0.8,
                class_id=0,
                frame_idx=0,
                is_fixated=True,
            ),
            Detection(
                bbox=(11, 11, 20, 20),
                class_name="cup",
                score=0.8,
                class_id=0,
                frame_idx=1,
                is_fixated=True,
            ),
            Detection(
                bbox=(12, 12, 20, 20),
                class_name="cup",
                score=0.8,
                class_id=0,
                frame_idx=2,
                is_fixated=True,
            ),
            Detection(
                bbox=(13, 13, 20, 20),
                class_name="cup",
                score=0.8,
                class_id=0,
                frame_idx=3,
                is_fixated=True,
            ),
            # Bowl with unstable bboxes
            Detection(
                bbox=(10, 10, 20, 20),
                class_name="bowl",
                score=0.8,
                class_id=1,
                frame_idx=0,
                is_fixated=True,
            ),
            Detection(
                bbox=(30, 30, 20, 20),
                class_name="bowl",
                score=0.8,
                class_id=1,
                frame_idx=1,
                is_fixated=True,
            ),
            Detection(
                bbox=(50, 50, 20, 20),
                class_name="bowl",
                score=0.8,
                class_id=1,
                frame_idx=2,
                is_fixated=True,
            ),
            Detection(
                bbox=(70, 70, 20, 20),
                class_name="bowl",
                score=0.8,
                class_id=1,
                frame_idx=3,
                is_fixated=True,
            ),
        ]

        # Set up other required state
        mock_detector.gaze_points = [(15, 15, 0), (15, 15, 1), (15, 15, 2), (15, 15, 3)]

        # Override thresholds to ensure we're only testing stability
        mock_detector.min_fixation_frame_threshold = 1
        mock_detector.min_fixation_frame_ratio = 0.1
        mock_detector.confidence_threshold = 0.0
        mock_detector.gaze_proximity_threshold = 0.0

        # Use real _compute_mean_iou but mock other methods
        with (
            patch.object(ObjectDetector, "_compute_geometric_mean", return_value=0.8),
            patch.object(
                ObjectDetector, "_compute_mean_gaze_distance", return_value=0.1
            ),
        ):
            # Compute scores
            scores = mock_detector._compute_fixation_scores()

            # Should only contain cup (stable bboxes)
            # Bowl should be filtered out (unstable bboxes)
            assert len(scores) == 1
            assert "cup" in scores
            assert "bowl" not in scores

    def test_gaze_proximity_threshold(self, mock_detector):
        """Test that gaze_proximity_threshold is correctly applied."""
        # Set up test data with two objects, one close to gaze, one far from gaze
        mock_detector.all_detections = [
            # Cup close to gaze
            Detection(
                bbox=(10, 10, 20, 20),
                class_name="cup",
                score=0.8,
                class_id=0,
                frame_idx=0,
                is_fixated=True,
            ),
            Detection(
                bbox=(11, 11, 20, 20),
                class_name="cup",
                score=0.8,
                class_id=0,
                frame_idx=1,
                is_fixated=True,
            ),
            Detection(
                bbox=(12, 12, 20, 20),
                class_name="cup",
                score=0.8,
                class_id=0,
                frame_idx=2,
                is_fixated=True,
            ),
            Detection(
                bbox=(13, 13, 20, 20),
                class_name="cup",
                score=0.8,
                class_id=0,
                frame_idx=3,
                is_fixated=True,
            ),
            # Bowl far from gaze
            Detection(
                bbox=(100, 100, 20, 20),
                class_name="bowl",
                score=0.8,
                class_id=1,
                frame_idx=0,
                is_fixated=True,
            ),
            Detection(
                bbox=(101, 101, 20, 20),
                class_name="bowl",
                score=0.8,
                class_id=1,
                frame_idx=1,
                is_fixated=True,
            ),
            Detection(
                bbox=(102, 102, 20, 20),
                class_name="bowl",
                score=0.8,
                class_id=1,
                frame_idx=2,
                is_fixated=True,
            ),
            Detection(
                bbox=(103, 103, 20, 20),
                class_name="bowl",
                score=0.8,
                class_id=1,
                frame_idx=3,
                is_fixated=True,
            ),
        ]

        # Set up gaze points close to cup
        mock_detector.gaze_points = [(20, 20, 0), (20, 20, 1), (20, 20, 2), (20, 20, 3)]

        # Override thresholds to ensure we're only testing gaze proximity
        mock_detector.min_fixation_frame_threshold = 1
        mock_detector.min_fixation_frame_ratio = 0.1
        mock_detector.confidence_threshold = 0.0
        mock_detector.bbox_stability_threshold = 0.0

        # Define custom side effect for _compute_mean_gaze_distance
        def mock_gaze_distance(bboxes, gaze_points):
            # Return small distance for cup (close to gaze)
            if bboxes[0][0] < 50:
                return 1.0  # Close to gaze, proximity = 1/(1+1) = 0.5
            # Return large distance for bowl (far from gaze)
            else:
                return 10.0  # Far from gaze, proximity = 1/(1+10) = 0.09

        # Mock methods
        with (
            patch.object(ObjectDetector, "_compute_mean_iou", return_value=0.8),
            patch.object(ObjectDetector, "_compute_geometric_mean", return_value=0.8),
            patch.object(
                ObjectDetector,
                "_compute_mean_gaze_distance",
                side_effect=mock_gaze_distance,
            ),
        ):
            # Compute scores
            scores = mock_detector._compute_fixation_scores()

            # Should only contain cup (close to gaze, proximity 0.5 > threshold 0.3)
            # Bowl should be filtered out (far from gaze, proximity 0.09 < threshold 0.3)
            assert len(scores) == 1
            assert "cup" in scores
            assert "bowl" not in scores

    def test_weights_affect_final_score(self, mock_detector):
        """Test that weights correctly affect the final fixation score."""
        # Set up test data with two objects
        mock_detector.all_detections = [
            # Cup
            Detection(
                bbox=(10, 10, 20, 20),
                class_name="cup",
                score=0.8,
                class_id=0,
                frame_idx=0,
                is_fixated=True,
            ),
            Detection(
                bbox=(11, 11, 20, 20),
                class_name="cup",
                score=0.8,
                class_id=0,
                frame_idx=1,
                is_fixated=True,
            ),
            # Bowl
            Detection(
                bbox=(50, 50, 20, 20),
                class_name="bowl",
                score=0.8,
                class_id=1,
                frame_idx=0,
                is_fixated=True,
            ),
            Detection(
                bbox=(51, 51, 20, 20),
                class_name="bowl",
                score=0.8,
                class_id=1,
                frame_idx=1,
                is_fixated=True,
            ),
        ]

        # Set up other required state
        mock_detector.gaze_points = [(15, 15, 0), (15, 15, 1)]

        # Set all thresholds to 0 to ensure both objects pass
        mock_detector.min_fixation_frame_threshold = 1
        mock_detector.min_fixation_frame_ratio = 0.1
        mock_detector.confidence_threshold = 0.0
        mock_detector.bbox_stability_threshold = 0.0
        mock_detector.gaze_proximity_threshold = 0.0

        # Test with default weights (all 1.0)
        with (
            patch.object(ObjectDetector, "_compute_mean_iou", return_value=0.8),
            patch.object(ObjectDetector, "_compute_geometric_mean", return_value=0.8),
            patch.object(
                ObjectDetector, "_compute_mean_gaze_distance", return_value=0.1
            ),
        ):
            # Compute scores with default weights
            scores_default = mock_detector._compute_fixation_scores()

            # Change weights to emphasize stability
            mock_detector.bbox_stability_weight = 2.0
            mock_detector.gaze_proximity_weight = 0.5
            mock_detector.duration_weight = 0.5
            mock_detector.confidence_weight = 0.5

            # Compute scores with modified weights
            scores_modified = mock_detector._compute_fixation_scores()

            # Both objects should be in both score sets
            assert "cup" in scores_default
            assert "bowl" in scores_default
            assert "cup" in scores_modified
            assert "bowl" in scores_modified

            # Final scores should be different due to weight changes
            assert (
                scores_default["cup"]["final_score"]
                != scores_modified["cup"]["final_score"]
            )
            assert (
                scores_default["bowl"]["final_score"]
                != scores_modified["bowl"]["final_score"]
            )
