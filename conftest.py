"""
Pytest configuration file for the project.
"""

import pytest
import os
import sys
from pathlib import Path

# Add the source directory to sys.path for importing from the package
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def pytest_addoption(parser):
    parser.addoption(
        "--run-real-model",
        action="store_true",
        default=False,
        help="run tests that require a real CLIP model"
    )

def pytest_collection_modifyitems(config, items):
    # Check if we're running in a cluster environment
    is_cluster = "SLURM_JOB_ID" in os.environ
    
    # If we're on a cluster or user explicitly requested real model tests,
    # don't skip any tests
    if is_cluster or config.getoption("--run-real-model"):
        return
    
    # Skip real_model tests when running locally
    skip_real_model = pytest.mark.skip(reason="need --run-real-model option to run")
    
    for item in items:
        if "real_model" in item.keywords:
            item.add_marker(skip_real_model) 