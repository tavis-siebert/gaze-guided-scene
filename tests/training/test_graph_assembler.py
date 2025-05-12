import pytest
import torch
from torch_geometric.data import Data
from types import SimpleNamespace

from gazegraph.datasets.egtea_gaze.action_record import ActionRecord
from gazegraph.training.dataset.graph_assembler import ActionGraph

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
    monkeypatch.setattr(ActionRecord, '_ensure_initialized', lambda *a, **k: None)
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
    monkeypatch.setattr(ActionRecord, 'get_records_for_video', lambda v: mock_records if v == "vid" else [])
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

def test_action_graph_empty(monkeypatch):
    # No records for video
    monkeypatch.setattr(ActionRecord, 'get_past_action_records', lambda v, f: [])
    dummy_emb = DummyNodeEmbeddings()
    ag = ActionGraph(dummy_emb, config=None, device="cpu")
    checkpoint = SimpleNamespace(video_name="none", frame_number=10)
    y = torch.tensor([1])
    data = ag.assemble(checkpoint, y)
    assert data.x.shape[0] == 0 and data.edge_index.shape[1] == 0

def test_action_graph_single_node(monkeypatch, patch_records_for_video):
    # Only one past record
    monkeypatch.setattr(ActionRecord, 'get_past_action_records', lambda v, f: [patch_records_for_video[0]])
    dummy_emb = DummyNodeEmbeddings()
    ag = ActionGraph(dummy_emb, config=None, device="cpu")
    checkpoint = SimpleNamespace(video_name="vid", frame_number=20)
    y = torch.tensor([1])
    data = ag.assemble(checkpoint, y)
    assert data.x.shape[0] == 1 and data.edge_index.shape[1] == 0
    assert torch.all(data.x[0] == dummy_emb.get_action_embedding(0))

def test_action_graph_multiple_nodes(monkeypatch, patch_records_for_video):
    # Two past records
    monkeypatch.setattr(ActionRecord, 'get_past_action_records', lambda v, f: patch_records_for_video[:2])
    dummy_emb = DummyNodeEmbeddings()
    ag = ActionGraph(dummy_emb, config=None, device="cpu")
    checkpoint = SimpleNamespace(video_name="vid", frame_number=40)
    y = torch.tensor([1])
    data = ag.assemble(checkpoint, y)
    assert data.x.shape[0] == 2 and data.edge_index.shape[1] == 1
    # Edge from node 0 to 1
    assert torch.equal(data.edge_index, torch.tensor([[0],[1]])) or torch.equal(data.edge_index, torch.tensor([[0,1],[1,2]]))
    # Embeddings correct
    assert torch.all(data.x[0] == dummy_emb.get_action_embedding(0))
    assert torch.all(data.x[1] == dummy_emb.get_action_embedding(1))