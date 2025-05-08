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