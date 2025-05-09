"""
Main pytest configuration file for tests directory.

This module sets up shared fixtures and configurations specific to tests.
"""

import pytest

# Import shared fixtures
pytest_plugins = [
    "tests.resources.fixtures",
]

# Add test-specific fixtures below
@pytest.fixture
def sample_data_path():
    """Return path to test sample data directory."""
    import os
    from pathlib import Path
    return Path(os.path.dirname(__file__)) / "resources" / "data" 


"""
Unit test fixtures for the gazegraph package.
"""

import pytest
import os
from pathlib import Path

# Register test marker for unit tests
def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "unit: mark a test as a unit test")
    config.addinivalue_line("markers", "integration: mark a test as an integration test")
    config.addinivalue_line("markers", "gpu: mark a test that requires a GPU")