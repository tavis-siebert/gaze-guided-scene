"""
Main pytest configuration file for tests directory.

This module sets up shared fixtures and configurations for all tests.
"""

import pytest
import os
from pathlib import Path

# Register test markers
def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "unit: mark a test as a unit test")
    config.addinivalue_line("markers", "integration: mark a test as an integration test")
    config.addinivalue_line("markers", "gpu: mark a test that requires a GPU")

@pytest.fixture
def sample_data_path():
    """Return path to test sample data directory."""
    return Path(os.path.dirname(os.path.dirname(__file__))) / "data" / "tests"

@pytest.fixture
def test_resources_dir():
    """Return the path to the test resources directory."""
    return Path(os.path.dirname(os.path.dirname(__file__))) / "data" / "tests"

@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(os.path.dirname(os.path.dirname(__file__))) / "data" / "tests"