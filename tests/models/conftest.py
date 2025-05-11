"""
Configuration for model-related tests.
"""

import pytest
import torch
import os
from PIL import Image
from pathlib import Path

from gazegraph.models.clip import ClipModel

# Get the test images directory path
TEST_IMAGES_DIR = Path(__file__).parent.parent.parent / "data" / "tests" / "images"

@pytest.fixture
def clip_model():
    """Fixture to provide a CLIP model instance for tests."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ClipModel(device=device)
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