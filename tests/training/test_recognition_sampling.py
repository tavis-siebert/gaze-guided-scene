"""
Tests for the unified recognition sampling functionality.
"""

import pytest
from unittest.mock import Mock

from gazegraph.graph.checkpoint_manager import GraphCheckpoint
from gazegraph.datasets.egtea_gaze.video_metadata import VideoMetadata
from gazegraph.datasets.egtea_gaze.action_record import ActionRecord
from gazegraph.training.dataset.recognition_sampling import (
    BaseRecognitionSampler,
    ActionRecognitionSampler,
    ObjectRecognitionSampler,
    get_action_recognition_samples,
    get_object_recognition_samples,
)


@pytest.fixture
def mock_metadata():
    """Create mock VideoMetadata with action records."""
    metadata = Mock(spec=VideoMetadata)

    # Create mock action records
    action_record_1 = Mock(spec=ActionRecord)
    action_record_1.action_idx = 1
    action_record_1.noun_id = 5
    action_record_1.start_frame = 100
    action_record_1.end_frame = 200

    action_record_2 = Mock(spec=ActionRecord)
    action_record_2.action_idx = 2
    action_record_2.noun_id = 8
    action_record_2.start_frame = 300
    action_record_2.end_frame = 400

    # Action record without valid action_idx (should be skipped)
    action_record_3 = Mock(spec=ActionRecord)
    action_record_3.action_idx = None
    action_record_3.noun_id = 10
    action_record_3.start_frame = 300
    action_record_3.end_frame = 320

    metadata.get_records_for_video.return_value = [
        action_record_1,
        action_record_2,
        action_record_3,
    ]

    return metadata


@pytest.fixture
def mock_checkpoints():
    """Create mock checkpoints for testing."""
    checkpoints = []

    for frame in [50, 150, 250, 350, 450]:
        # Create mock nodes with visits
        nodes = {
            -1: {"type": "root"},  # Root node
            0: {"visits": [[frame - 20, frame + 10]], "label": "object1"},
            1: {"visits": [[frame - 10, frame + 20]], "label": "object2"},
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
            object_label_to_id={"object1": 0, "object2": 1},
            video_length=500,
        )
        checkpoints.append(checkpoint)

    return checkpoints


class TestBaseRecognitionSampler:
    """Test cases for BaseRecognitionSampler."""

    def test_init(self, mock_metadata):
        """Test sampler initialization."""
        sampler = BaseRecognitionSampler(mock_metadata)
        assert sampler.metadata == mock_metadata

    def test_get_recognition_samples_empty_checkpoints(self, mock_metadata):
        """Test sampling with empty checkpoints list."""
        sampler = BaseRecognitionSampler(mock_metadata)
        samples = sampler.get_recognition_samples(
            [], "test_video", label_extractor=lambda r: r.action_idx
        )
        assert samples == []

    def test_get_recognition_samples_no_actions(self, mock_checkpoints):
        """Test sampling when no action records are found."""
        metadata = Mock(spec=VideoMetadata)
        metadata.get_records_for_video.return_value = []

        sampler = BaseRecognitionSampler(metadata)
        samples = sampler.get_recognition_samples(
            mock_checkpoints, "test_video", label_extractor=lambda r: r.action_idx
        )
        assert samples == []

    def test_get_recognition_samples_with_label_extractor(
        self, mock_metadata, mock_checkpoints
    ):
        """Test sampling with custom label extractor."""
        sampler = BaseRecognitionSampler(mock_metadata)

        # Test with action_idx extractor
        samples = sampler.get_recognition_samples(
            mock_checkpoints,
            "test_video",
            label_extractor=lambda r: r.action_idx,
            label_key="action_recognition",
        )

        # Should get two samples (action_record_3 has None action_idx and is skipped)
        assert len(samples) == 2

        # Check sample structure
        for checkpoint, labels in samples:
            assert isinstance(checkpoint, GraphCheckpoint)
            assert "action_recognition" in labels
            assert labels["action_recognition"] in [1, 2]

    def test_find_best_checkpoint(self, mock_metadata, mock_checkpoints):
        """Test finding best checkpoint for target frame."""
        sampler = BaseRecognitionSampler(mock_metadata)

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

    def test_filter_checkpoint_for_action_basic(self, mock_metadata, mock_checkpoints):
        """Test basic checkpoint filtering."""
        sampler = BaseRecognitionSampler(mock_metadata)

        # Create action that overlaps with checkpoint visits
        action_record = Mock(spec=ActionRecord)
        action_record.start_frame = 140
        action_record.end_frame = 160

        checkpoint = mock_checkpoints[1]  # Frame 150
        filtered = sampler._filter_checkpoint_for_action(
            checkpoint, action_record, min_nodes_threshold=1, visit_lookback_frames=0
        )

        assert filtered is not None
        assert len(filtered.nodes) >= 2  # Root + at least one object node

    def test_filter_checkpoint_min_nodes_threshold(
        self, mock_metadata, mock_checkpoints
    ):
        """Test minimum nodes threshold filtering."""
        sampler = BaseRecognitionSampler(mock_metadata)

        action_record = Mock(spec=ActionRecord)
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


class TestActionRecognitionSampler:
    """Test cases for ActionRecognitionSampler."""

    def test_get_action_recognition_samples(self, mock_metadata, mock_checkpoints):
        """Test action recognition sampling."""
        sampler = ActionRecognitionSampler(mock_metadata)
        samples = sampler.get_action_recognition_samples(mock_checkpoints, "test_video")

        # Should get two samples (one for each valid action)
        assert len(samples) == 2

        # Check sample structure
        for checkpoint, labels in samples:
            assert isinstance(checkpoint, GraphCheckpoint)
            assert "action_recognition" in labels
            assert labels["action_recognition"] in [1, 2]


class TestObjectRecognitionSampler:
    """Test cases for ObjectRecognitionSampler."""

    def test_get_object_recognition_samples(self, mock_metadata, mock_checkpoints):
        """Test object recognition sampling."""
        sampler = ObjectRecognitionSampler(mock_metadata)
        samples = sampler.get_object_recognition_samples(mock_checkpoints, "test_video")

        # Should get two samples (one for each valid noun)
        assert len(samples) == 2

        # Check sample structure
        for checkpoint, labels in samples:
            assert isinstance(checkpoint, GraphCheckpoint)
            assert "object_recognition" in labels
            assert labels["object_recognition"] in [5, 8]


def test_get_action_recognition_samples_function(mock_metadata, mock_checkpoints):
    """Test the main entry point function for action recognition."""
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


def test_get_object_recognition_samples_function(mock_metadata, mock_checkpoints):
    """Test the main entry point function for object recognition."""
    samples = get_object_recognition_samples(
        checkpoints=mock_checkpoints,
        video_name="test_video",
        samples_per_action=1,
        metadata=mock_metadata,
    )

    assert len(samples) == 2

    for checkpoint, labels in samples:
        assert isinstance(checkpoint, GraphCheckpoint)
        assert "object_recognition" in labels
