"""
Main pytest configuration file.

This module sets up the testing environment for all tests,
including path configuration and shared fixtures.
"""

import os
import sys
from pathlib import Path

# Add the source directory to sys.path for importing from the package
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Set up any environment variables needed for testing
os.environ["PYTHONPATH"] = f"{str(src_path)}:{os.environ.get('PYTHONPATH', '')}"

# Import shared fixtures
pytest_plugins = [
    "tests.resources.fixtures",
] 