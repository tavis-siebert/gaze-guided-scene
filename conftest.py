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