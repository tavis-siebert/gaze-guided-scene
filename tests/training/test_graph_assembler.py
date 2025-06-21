import pytest
import torch
from unittest.mock import MagicMock, patch

from gazegraph.datasets.egtea_gaze.action_record import ActionRecord
from gazegraph.training.dataset.graph_assembler import (
    ObjectGraph,
    create_graph_assembler,
)
from gazegraph.training.dataset.node_features import NodeFeatureExtractor
from gazegraph.config.config_utils import DotDict


class DummyNodeEmbeddings:
    def __init__(self, emb_dim=4):
        self.emb_dim = emb_dim
        self._cache = {}

    def get_action_embedding(self, action_idx):
        # Deterministic dummy embedding
        if action_idx not in self._cache:
            self._cache[action_idx] = torch.ones(self.emb_dim) * (action_idx + 1)
        return self._cache[action_idx]


@pytest.fixture
def mock_records(monkeypatch):
    monkeypatch.setattr(ActionRecord, "_ensure_initialized", lambda *a, **k: None)
    records = [
        ActionRecord(["vid", "10", "20", "1", "3"]),  # idx=0
        ActionRecord(["vid", "30", "40", "2", "1"]),  # idx=1
        ActionRecord(["vid", "50", "60", "3", "2"]),  # idx=2
    ]
    ActionRecord._action_to_idx = {(1, 3): 0, (2, 1): 1, (3, 2): 2}
    for r in records:
        r._is_initialized = True
    return records


@pytest.fixture
def patch_records_for_video(mock_records, monkeypatch):
    monkeypatch.setattr(
        ActionRecord,
        "get_records_for_video",
        lambda v: mock_records if v == "vid" else [],
    )
    return mock_records


def test_get_past_action_records(patch_records_for_video):
    # Only first two records are past for frame_number 40
    past = ActionRecord.get_past_action_records("vid", 40)
    assert len(past) == 2
    assert past[0].end_frame == 20 and past[1].end_frame == 40
    # All for frame_number >= 60
    assert len(ActionRecord.get_past_action_records("vid", 60)) == 3
    # None for frame_number < 20
    assert not ActionRecord.get_past_action_records("vid", 5)


def test_get_future_action_records(patch_records_for_video):
    # Only last record is future for frame_number 40
    fut = ActionRecord.get_future_action_records("vid", 40)
    assert len(fut) == 1 and fut[0].start_frame == 50
    # All for frame_number < 10
    fut = ActionRecord.get_future_action_records("vid", 5)
    assert len(fut) == 3
    # None for frame_number >= 60
    assert not ActionRecord.get_future_action_records("vid", 60)


# Create a DotDict-compatible config for testing
def create_test_config():
    # DotDict requires a dictionary for initialization
    config = DotDict(
        {
            # Add minimal required attributes for testing
            "training": {"num_classes": 10},
            "directories": {"graphs": "/tmp/graphs", "traces": "/tmp/traces"},
            "dataset": {
                "egtea": {"raw_videos": "/tmp/videos"},
                "embeddings": {
                    "object_label_embedding_path": "/tmp/embeddings/object_labels",
                    "roi_embedding_path": "/tmp/embeddings/roi",
                    "action_embedding_path": "/tmp/embeddings/actions",
                },
            },
        }
    )
    return config


@patch("gazegraph.training.dataset.graph_assembler.get_node_feature_extractor")
def test_create_graph_assembler_object_graph(mock_get_node_feature_extractor):
    # Setup mock
    mock_extractor = MagicMock(spec=NodeFeatureExtractor)
    mock_get_node_feature_extractor.return_value = mock_extractor

    # Test object-graph creation
    config = create_test_config()
    assembler = create_graph_assembler(
        graph_type="object-graph",
        config=config,
        device="cpu",
        object_node_feature="one-hot",
    )

    # Verify correct type and initialization
    assert isinstance(assembler, ObjectGraph)
    assert assembler.config == config
    assert assembler.device == "cpu"
    assert assembler.object_node_feature == "one-hot"
    assert assembler.node_feature_extractor == mock_extractor

    # Verify the node feature extractor was created with correct params
    mock_get_node_feature_extractor.assert_called_once_with(
        "one-hot", device="cpu", config=config
    )


def test_create_graph_assembler_invalid_type():
    # Test invalid graph type
    config = create_test_config()

    # We need to use monkeypatch to bypass the type checking for this test
    # since we're intentionally passing an invalid graph type
    with patch(
        "gazegraph.training.dataset.graph_assembler.create_graph_assembler"
    ) as mock_create:
        mock_create.side_effect = ValueError("Unknown graph type: invalid-graph-type")

        with pytest.raises(ValueError) as excinfo:
            mock_create(graph_type="invalid-graph-type", config=config, device="cpu")

        assert "Unknown graph type" in str(excinfo.value)
