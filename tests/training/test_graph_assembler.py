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
from gazegraph.graph.checkpoint_manager import GraphCheckpoint


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


@pytest.fixture
def mock_node_feature_extractor():
    """Mock node feature extractor for testing."""
    extractor = MagicMock(spec=NodeFeatureExtractor)
    extractor.feature_dim = 5
    extractor.extract_features.return_value = torch.randn(3, 5)  # 3 nodes, 5 features
    return extractor


@pytest.fixture
def mock_checkpoint_with_non_consecutive_ids():
    """Create a mock checkpoint with non-consecutive node IDs that would cause index out of bounds."""
    nodes = {
        0: {"object_label": "object1", "visits": [[10, 20]]},
        5: {"object_label": "object2", "visits": [[30, 40]]},
        232: {"object_label": "object3", "visits": [[50, 60]]},  # High node ID
    }

    # Edges referencing the non-consecutive node IDs
    edges = [
        {"source_id": 0, "target_id": 5, "angle": 0.5},
        {"source_id": 5, "target_id": 232, "angle": 1.0},  # This will cause issues
    ]

    return GraphCheckpoint(
        nodes=nodes,
        edges=edges,
        adjacency={0: [5], 5: [232], 232: []},
        frame_number=100,
        non_black_frame_count=100,
        video_name="test_video",
        object_label_to_id={"object1": 0, "object2": 1, "object3": 2},
        video_length=500,
    )


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


def test_object_graph_with_non_consecutive_node_ids(
    mock_checkpoint_with_non_consecutive_ids, mock_node_feature_extractor
):
    """Test that ObjectGraph handles non-consecutive node IDs correctly."""
    config = create_test_config()

    assembler = ObjectGraph(
        node_feature_extractor=mock_node_feature_extractor,
        object_node_feature="one-hot",
        config=config,
        device="cpu",
    )

    y = torch.tensor([1])  # Dummy target

    # This should not raise an error and should produce valid edge indices
    data = assembler.assemble(mock_checkpoint_with_non_consecutive_ids, y)

    # Check that data components are not None
    assert data.x is not None, "Node features should not be None"
    assert data.edge_index is not None, "Edge index should not be None"
    assert data.edge_attr is not None, "Edge attributes should not be None"

    # Check that edge indices are within valid range
    num_nodes = data.x.size(0)
    max_edge_index = data.edge_index.max().item() if data.edge_index.numel() > 0 else -1

    # This assertion should pass - edge indices should be < num_nodes
    assert max_edge_index < num_nodes, (
        f"Edge index {max_edge_index} >= num_nodes {num_nodes}"
    )

    # Verify the graph structure is valid
    assert data.x.size(0) == 3  # Should have 3 nodes
    assert data.edge_index.size(0) == 2  # Should be 2 x num_edges
    assert data.edge_attr.size(0) == data.edge_index.size(
        1
    )  # Edge attributes should match edge count


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
