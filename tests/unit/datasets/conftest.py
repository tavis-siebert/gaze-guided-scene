"""
Configuration for dataset-related tests.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock

@pytest.fixture
def mock_dataset_config():
    """Mock configuration with test data paths for datasets."""
    config = MagicMock()
    config.dataset.egtea.action_annotations = "test_annotations"
    
    # Set up mock splits data
    config.dataset.ego_topo.splits = MagicMock()
    config.dataset.ego_topo.splits.train = "test_train_split.txt"
    config.dataset.ego_topo.splits.val = "test_val_split.txt"
    
    return config

@pytest.fixture
def temp_dataset_dir():
    """Create a temporary directory for test datasets."""
    with tempfile.TemporaryDirectory() as temp_dir:
        os.makedirs(os.path.join(temp_dir, "annotations"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "images"), exist_ok=True)
        yield Path(temp_dir) 