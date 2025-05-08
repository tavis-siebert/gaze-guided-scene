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
    config.addinivalue_line("markers", "real_model: mark a test that requires the actual model")
    config.addinivalue_line("markers", "mock_only: mark a test that uses mock models only") 