"""
Pytest configuration file for the project.
"""

import pytest
import os
import sys
from pathlib import Path
import torch

# Add the source directory to sys.path for importing from the package
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Register test markers
def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "unit: mark a test as a unit test")
    config.addinivalue_line("markers", "integration: mark a test as an integration test")
    config.addinivalue_line("markers", "gpu: mark a test that requires a GPU")

def pytest_addoption(parser):
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="run tests that require a GPU"
    )

def pytest_collection_modifyitems(config, items):
    has_gpu = torch.cuda.is_available()

    if has_gpu or config.getoption("--run-gpu"):
        return

    # Skip GPU tests when there is no GPU support
    skip_gpu = pytest.mark.skip(reason="need --run-gpu option to run")
    
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)

@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / "data" / "tests" 