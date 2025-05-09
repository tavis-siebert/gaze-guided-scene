"""
Configuration for training-related tests.
"""

import pytest
import torch
import os
from pathlib import Path

@pytest.fixture
def training_sample_data():
    """Return path to training test data directory."""
    return Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / "data" / "tests" / "training"

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