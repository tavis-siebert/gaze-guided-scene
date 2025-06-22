"""
Configuration for EGTEA Gaze dataset-related tests.
"""

import pytest
import tempfile
import os


@pytest.fixture
def mock_verb_index_file():
    """Create a temporary verb index file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
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
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("cup 1\n")
        f.write("bowl 2\n")
        f.write("knife 3\n")
        f.write("microwave 4\n")
        f.write("fridge 5\n")
        temp_path = f.name

    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def mock_split_files():
    """Create temporary train and val split files for testing."""
    # Create train split file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as train_file:
        train_file.write("video_1\t10\t30\t1\t3\n")  # take knife
        train_file.write("video_1\t40\t60\t2\t1\n")  # put cup
        train_file.write("video_2\t5\t25\t3\t4\n")  # open microwave
        train_file.write("video_2\t30\t50\t4\t4\n")  # close microwave
        train_file.write("video_3\t15\t35\t5\t2\n")  # wash bowl
        train_path = train_file.name

    # Create val split file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as val_file:
        val_file.write("video_4\t5\t25\t1\t1\n")  # take cup
        val_file.write("video_4\t30\t50\t4\t5\n")  # close fridge
        val_path = val_file.name

    yield {"train": train_path, "val": val_path}

    # Cleanup
    os.unlink(train_path)
    os.unlink(val_path)


@pytest.fixture
def setup_mock_files(
    mock_verb_index_file, mock_noun_index_file, mock_split_files, monkeypatch
):
    """Set up mock files and directories for testing."""
    # Create a mock directory structure
    os.makedirs("test_annotations", exist_ok=True)

    # Mock the open function to return our test files
    original_open = open

    def mock_open(file, *args, **kwargs):
        if file == "test_annotations/verb_idx.txt":
            return original_open(mock_verb_index_file, *args, **kwargs)
        elif file == "test_annotations/noun_idx.txt":
            return original_open(mock_noun_index_file, *args, **kwargs)
        elif file == "test_train_split.txt":
            return original_open(mock_split_files["train"], *args, **kwargs)
        elif file == "test_val_split.txt":
            return original_open(mock_split_files["val"], *args, **kwargs)
        else:
            return original_open(file, *args, **kwargs)

    # Apply patches
    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr("builtins.open", mock_open)

    yield

    # Clean up
    try:
        os.rmdir("test_annotations")
    except:
        pass
