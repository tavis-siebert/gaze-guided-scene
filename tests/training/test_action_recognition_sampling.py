"""
Tests for action recognition sampling functionality.
"""

import pytest
from unittest.mock import Mock, patch
from gazegraph.training.dataset.action_recognition_sampling import (
    ActionRecognitionSampler,
    get_action_recognition_samples,
)
from gazegraph.graph.checkpoint_manager import GraphCheckpoint
from gazegraph.datasets.egtea_gaze.action_record import ActionRecord
from gazegraph.datasets.egtea_gaze.video_metadata import VideoMetadata


@pytest.fixture
def mock_metadata():
    """Create mock metadata with test action records."""
    metadata = Mock(spec=VideoMetadata)

    # Create mock action records directly
    action_records = []
    for i, (start, end, action_idx) in enumerate([(100, 200, 1), (300, 400, 2)]):
        record = Mock(spec=ActionRecord)
        record.action_idx = action_idx
        record.start_frame = start
        record.end_frame = end
        action_records.append(record)

    metadata.get_records_for_video.return_value = action_records
    return metadata


@pytest.fixture
def mock_checkpoints():
    """Create mock checkpoints for testing."""
    checkpoints = []

    for frame in [50, 150, 250, 350, 450]:
        # Create mock nodes with visits
        nodes = {
            -1: {"type": "root"},  # Root node
            0: {"visits": [[frame - 20, frame + 10]], "label": "plate"},
            1: {"visits": [[frame - 10, frame + 20]], "label": "hand"},
        }

        # Create mock edges
        edges = [
            {"source_id": -1, "target_id": 0},
            {"source_id": -1, "target_id": 1},
            {"source_id": 0, "target_id": 1},
        ]

        # Create mock adjacency
        adjacency = {
            -1: [0, 1],
            0: [1],
            1: [],
        }

        checkpoint = GraphCheckpoint(
            nodes=nodes,
            edges=edges,
            adjacency=adjacency,
            frame_number=frame,
            non_black_frame_count=frame,
            video_name="test_video",
            object_label_to_id={"plate": 0, "hand": 1},
            video_length=500,
        )
        checkpoints.append(checkpoint)

    return checkpoints


class TestActionRecognitionSampler:
    """Test cases for ActionRecognitionSampler."""

    def test_init(self, mock_metadata):
        """Test sampler initialization."""
        sampler = ActionRecognitionSampler(mock_metadata)
        assert sampler.metadata == mock_metadata

    def test_get_action_recognition_samples_empty_checkpoints(self, mock_metadata):
        """Test sampling with empty checkpoints list."""
        sampler = ActionRecognitionSampler(mock_metadata)
        samples = sampler.get_action_recognition_samples([], "test_video")
        assert samples == []

    def test_get_action_recognition_samples_no_actions(self, mock_checkpoints):
        """Test sampling when no action records are found."""
        metadata = Mock(spec=VideoMetadata)
        metadata.get_records_for_video.return_value = []

        sampler = ActionRecognitionSampler(metadata)
        samples = sampler.get_action_recognition_samples(mock_checkpoints, "test_video")
        assert samples == []

    def test_get_action_recognition_samples_default_params(
        self, mock_metadata, mock_checkpoints
    ):
        """Test sampling with default parameters."""
        sampler = ActionRecognitionSampler(mock_metadata)
        samples = sampler.get_action_recognition_samples(mock_checkpoints, "test_video")

        # Should get one sample per action (2 actions)
        assert len(samples) == 2

        # Check sample structure
        for checkpoint, labels in samples:
            assert isinstance(checkpoint, GraphCheckpoint)
            assert "action_recognition" in labels
            assert labels["action_recognition"] in [1, 2]

    def test_get_action_recognition_samples_custom_params(
        self, mock_metadata, mock_checkpoints
    ):
        """Test sampling with custom parameters."""
        sampler = ActionRecognitionSampler(mock_metadata)
        samples = sampler.get_action_recognition_samples(
            mock_checkpoints,
            "test_video",
            action_completion_ratio=0.5,
            min_nodes_threshold=1,
            visit_lookback_frames=30,
        )

        assert len(samples) == 2

    def test_find_best_checkpoint(self, mock_metadata, mock_checkpoints):
        """Test finding best checkpoint for target frame."""
        sampler = ActionRecognitionSampler(mock_metadata)

        # Test exact match
        best = sampler._find_best_checkpoint(mock_checkpoints, 150)
        assert best is not None
        assert best.frame_number == 150

        # Test closest before target
        best = sampler._find_best_checkpoint(mock_checkpoints, 175)
        assert best is not None
        assert best.frame_number == 150

        # Test no checkpoint before target
        best = sampler._find_best_checkpoint(mock_checkpoints, 25)
        assert best is None

    def test_filter_checkpoint_for_action_no_lookback(
        self, mock_metadata, mock_checkpoints
    ):
        """Test checkpoint filtering without lookback window."""
        sampler = ActionRecognitionSampler(mock_metadata)

        # Create action that overlaps with checkpoint visits
        action_record = Mock(spec=ActionRecord)
        action_record.action_idx = 1
        action_record.start_frame = 140
        action_record.end_frame = 160

        checkpoint = mock_checkpoints[1]  # Frame 150
        filtered = sampler._filter_checkpoint_for_action(
            checkpoint, action_record, min_nodes_threshold=1, visit_lookback_frames=0
        )

        assert filtered is not None
        assert len(filtered.nodes) >= 2  # Root + at least one object node

    def test_filter_checkpoint_for_action_with_lookback(
        self, mock_metadata, mock_checkpoints
    ):
        """Test checkpoint filtering with lookback window."""
        sampler = ActionRecognitionSampler(mock_metadata)

        # Create action that doesn't overlap without lookback
        action_record = Mock(spec=ActionRecord)
        action_record.action_idx = 1
        action_record.start_frame = 180
        action_record.end_frame = 200

        checkpoint = mock_checkpoints[1]  # Frame 150

        # Without lookback - should fail
        filtered = sampler._filter_checkpoint_for_action(
            checkpoint, action_record, min_nodes_threshold=1, visit_lookback_frames=0
        )
        assert filtered is None

        # With lookback - should succeed
        filtered = sampler._filter_checkpoint_for_action(
            checkpoint, action_record, min_nodes_threshold=1, visit_lookback_frames=50
        )
        assert filtered is not None

    def test_filter_checkpoint_min_nodes_threshold(
        self, mock_metadata, mock_checkpoints
    ):
        """Test minimum nodes threshold filtering."""
        sampler = ActionRecognitionSampler(mock_metadata)

        action_record = Mock(spec=ActionRecord)
        action_record.action_idx = 1
        action_record.start_frame = 140
        action_record.end_frame = 160

        checkpoint = mock_checkpoints[1]

        # High threshold should fail
        filtered = sampler._filter_checkpoint_for_action(
            checkpoint, action_record, min_nodes_threshold=10, visit_lookback_frames=0
        )
        assert filtered is None

        # Low threshold should succeed
        filtered = sampler._filter_checkpoint_for_action(
            checkpoint, action_record, min_nodes_threshold=1, visit_lookback_frames=0
        )
        assert filtered is not None


def test_get_action_recognition_samples_function(mock_metadata, mock_checkpoints):
    """Test the main entry point function."""
    samples = get_action_recognition_samples(
        checkpoints=mock_checkpoints,
        video_name="test_video",
        samples_per_action=1,
        metadata=mock_metadata,
    )

    assert len(samples) == 2

    for checkpoint, labels in samples:
        assert isinstance(checkpoint, GraphCheckpoint)
        assert "action_recognition" in labels


def test_get_action_recognition_samples_with_kwargs(mock_metadata, mock_checkpoints):
    """Test entry point function with additional kwargs."""
    samples = get_action_recognition_samples(
        checkpoints=mock_checkpoints,
        video_name="test_video",
        samples_per_action=1,
        metadata=mock_metadata,
        action_completion_ratio=0.8,
        min_nodes_threshold=1,
        visit_lookback_frames=20,
    )

    assert len(samples) == 2
