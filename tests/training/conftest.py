"""
Configuration for training-related tests.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

from gazegraph.datasets.node_embeddings import NodeEmbeddings
from gazegraph.config.config_utils import DotDict


@pytest.fixture
def mock_model():
    """Create a mock PyTorch model for testing training loops."""

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 2)

        def forward(self, x):
            return self.linear(x)

    return MockModel()


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    config = DotDict(
        {
            "dataset": DotDict(
                {
                    "embeddings": DotDict(
                        {
                            "object_label_embedding_path": "mock_object_embeddings.pth",
                            "action_label_embedding_path": "mock_action_embeddings.pth",
                            "roi_embedding_samples": 3,
                        }
                    )
                }
            )
        }
    )
    return config


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def clip_model():
    """Fixture to provide a persistent CLIP model instance for tests."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ClipModel(device=device)
    return model


@pytest.fixture
def node_embeddings(device, clip_model, mock_config):
    """Fixture for a NodeEmbeddings instance configured for testing."""
    # Disable prepopulation for most tests to avoid unnecessary computation
    return NodeEmbeddings(
        config=mock_config,
        device=device,
        prepopulate_caches=False,
        clip_model=clip_model,
    )


@pytest.fixture()
def mock_clip_model():
    return MagicMock()


@pytest.fixture()
def mock_node_embeddings(mock_config):
    # Patch the _load_caches method to prevent loading from cache files during tests
    with patch.object(NodeEmbeddings, "_load_caches"):
        node_embeddings = NodeEmbeddings(
            config=mock_config,
            device="cpu",
            prepopulate_caches=False,
            clip_model=MagicMock(),
        )
        # Clear any caches that might have been loaded
        node_embeddings._object_label_embedding_cache = {}
        node_embeddings._action_label_embedding_cache = {}
        return node_embeddings
