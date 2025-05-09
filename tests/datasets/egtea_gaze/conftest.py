"""
Configuration for EGTEA Gaze dataset-related tests.
"""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

@pytest.fixture
def mock_verb_index_file():
    """Create a temporary verb index file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("take 1\n")
        f.write("put 2\n")
        f.write("open 3\n")
        f.write("close 4\n")
        f.write("wash 5\n")
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)

@pytest.fixture
def mock_noun_index_file():
    """Create a temporary noun index file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("cup 1\n")
        f.write("bowl 2\n")
        f.write("knife 3\n")
        f.write("microwave 4\n")
        f.write("fridge 5\n")
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)

@pytest.fixture
def mock_train_split_file():
    """Create a temporary train split file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("video_1\t10\t30\t1\t3\n")  # take knife
        f.write("video_1\t40\t60\t2\t1\n")  # put cup
        f.write("video_2\t5\t25\t3\t4\n")   # open microwave
        f.write("video_2\t30\t50\t4\t4\n")  # close microwave
        f.write("video_3\t15\t35\t5\t2\n")  # wash bowl
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)

@pytest.fixture
def mock_val_split_file():
    """Create a temporary val split file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("video_4\t5\t25\t1\t1\n")   # take cup
        f.write("video_4\t30\t50\t4\t5\n")  # close fridge
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)

@pytest.fixture
def setup_mock_files(mock_verb_index_file, mock_noun_index_file, mock_train_split_file, mock_val_split_file, monkeypatch):
    """Set up mock files and directories for testing."""
    # Create a mock directory structure
    os.makedirs("test_annotations", exist_ok=True)
    
    # Copy mock files to test directory
    monkeypatch.setattr("os.path.exists", lambda path: True)
    
    # Mock the open function to return our test files
    original_open = open
    
    def mock_open(file, *args, **kwargs):
        if file == "test_annotations/verb_idx.txt":
            return original_open(mock_verb_index_file, *args, **kwargs)
        elif file == "test_annotations/noun_idx.txt":
            return original_open(mock_noun_index_file, *args, **kwargs)
        elif file == "test_train_split.txt":
            return original_open(mock_train_split_file, *args, **kwargs)
        elif file == "test_val_split.txt":
            return original_open(mock_val_split_file, *args, **kwargs)
        else:
            return original_open(file, *args, **kwargs)
    
    # Apply the patch
    monkeypatch.setattr("builtins.open", mock_open)
    
    yield
    
    # Clean up
    try:
        os.rmdir("test_annotations")
    except:
        pass 