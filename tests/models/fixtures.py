"""
Shared test fixtures and test data for various test modules.
"""

import pytest
import torch
import os
from PIL import Image
from pathlib import Path

from gazegraph.models.clip import ClipModel

# Get the test resources directory
TEST_RESOURCES_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_RESOURCES_DIR / "data"
TEST_IMAGES_DIR = TEST_DATA_DIR / "images"

@pytest.fixture
def test_resources_dir():
    """Return the path to the test resources directory."""
    return TEST_RESOURCES_DIR

@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return TEST_DATA_DIR

@pytest.fixture
def clip_model():
    """Fixture to provide a CLIP model instance for tests."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ClipModel(device=device)
    model.load()
    return model

@pytest.fixture
def test_images():
    """Fixture to load test images from the data directory."""
    # Ensure the directory exists
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
    
    images = {}
    for img_path in TEST_IMAGES_DIR.glob("*.jpg"):
        images[img_path.stem] = Image.open(img_path)
    for img_path in TEST_IMAGES_DIR.glob("*.png"):
        images[img_path.stem] = Image.open(img_path)
    return images 