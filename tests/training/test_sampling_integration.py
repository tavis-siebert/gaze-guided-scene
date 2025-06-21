"""
Tests for sampling integration with action recognition.
"""

import pytest
from unittest.mock import Mock, patch
from gazegraph.training.dataset.sampling import get_samples
from gazegraph.graph.checkpoint_manager import GraphCheckpoint
from gazegraph.datasets.egtea_gaze.video_metadata import VideoMetadata


@pytest.fixture
def mock_metadata():
    """Create mock metadata."""
    metadata = Mock(spec=VideoMetadata)
    metadata.get_action_frame_range.return_value = (0, 500)
    return metadata


@pytest.fixture
def mock_checkpoints():
    """Create mock checkpoints."""
    checkpoints = []
    for frame in [100, 200, 300, 400]:
        checkpoint = Mock(spec=GraphCheckpoint)
        checkpoint.frame_number = frame
        checkpoint.get_future_action_labels.return_value = {"future_actions": [1, 2]}
        checkpoints.append(checkpoint)
    return checkpoints


class TestSamplingIntegration:
    """Test cases for sampling integration."""

    def test_get_samples_action_recognition_mode(self, mock_checkpoints, mock_metadata):
        """Test get_samples with action recognition task mode."""
        with patch(
            "gazegraph.training.dataset.sampling.get_action_recognition_samples"
        ) as mock_ar_samples:
            mock_ar_samples.return_value = [
                (mock_checkpoints[0], {"action_recognition": 1})
            ]

            samples = get_samples(
                checkpoints=mock_checkpoints,
                video_name="test_video",
                strategy="all",
                samples_per_video=1,
                allow_duplicates=False,
                oversampling=False,
                metadata=mock_metadata,
                task_mode="action_recognition",
                action_completion_ratio=1.0,
                min_nodes_threshold=1,
                visit_lookback_frames=0,
            )

            # Verify action recognition sampling was called
            mock_ar_samples.assert_called_once_with(
                checkpoints=mock_checkpoints,
                video_name="test_video",
                samples_per_action=1,
                metadata=mock_metadata,
                action_completion_ratio=1.0,
                min_nodes_threshold=1,
                visit_lookback_frames=0,
            )

            assert len(samples) == 1
            assert samples[0][1]["action_recognition"] == 1

    def test_get_samples_future_actions_mode(self, mock_checkpoints, mock_metadata):
        """Test get_samples with future actions task mode (default behavior)."""
        samples = get_samples(
            checkpoints=mock_checkpoints,
            video_name="test_video",
            strategy="all",
            samples_per_video=0,
            allow_duplicates=False,
            oversampling=False,
            metadata=mock_metadata,
            task_mode="future_actions",
        )

        # Should return samples with future action labels
        assert len(samples) == 4
        for checkpoint, labels in samples:
            assert "future_actions" in labels

    def test_get_samples_empty_checkpoints(self, mock_metadata):
        """Test get_samples with empty checkpoints."""
        samples = get_samples(
            checkpoints=[],
            video_name="test_video",
            strategy="all",
            samples_per_video=1,
            allow_duplicates=False,
            oversampling=False,
            metadata=mock_metadata,
            task_mode="action_recognition",
        )

        assert samples == []

    def test_get_samples_parameter_forwarding(self, mock_checkpoints, mock_metadata):
        """Test that parameters are correctly forwarded to action recognition sampling."""
        with patch(
            "gazegraph.training.dataset.sampling.get_action_recognition_samples"
        ) as mock_ar_samples:
            mock_ar_samples.return_value = []

            custom_params = {
                "action_completion_ratio": 0.8,
                "min_nodes_threshold": 2,
                "visit_lookback_frames": 30,
                "custom_param": "test_value",
            }

            get_samples(
                checkpoints=mock_checkpoints,
                video_name="test_video",
                strategy="uniform",
                samples_per_video=2,
                allow_duplicates=True,
                oversampling=True,
                metadata=mock_metadata,
                task_mode="action_recognition",
                **custom_params,
            )

            # Verify all parameters were forwarded
            mock_ar_samples.assert_called_once_with(
                checkpoints=mock_checkpoints,
                video_name="test_video",
                samples_per_action=2,
                metadata=mock_metadata,
                **custom_params,
            )
