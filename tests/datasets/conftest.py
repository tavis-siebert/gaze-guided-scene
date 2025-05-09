"""
Configuration for dataset-related tests.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock

@pytest.fixture
def mock_config():
    """Mock configuration with test data paths for datasets."""
    config = MagicMock()
    
    # EGTEA config
    config.dataset.egtea.action_annotations = "test_annotations"
    
    # EgoTopo config
    config.dataset.ego_topo.splits = MagicMock()
    config.dataset.ego_topo.splits.train = "test_train_split.txt"
    config.dataset.ego_topo.splits.val = "test_val_split.txt"
    
    return config

# Keeping the mock_dataset_config as an alias for backward compatibility
@pytest.fixture
def mock_dataset_config(mock_config):
    """Alias for mock_config for backward compatibility."""
    return mock_config

@pytest.fixture
def temp_dataset_dir():
    """Create a temporary directory for test datasets."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_dir = Path(temp_dir)
        # Create common dataset directory structure
        (dataset_dir / "annotations").mkdir(exist_ok=True)
        (dataset_dir / "images").mkdir(exist_ok=True)
        yield dataset_dir 