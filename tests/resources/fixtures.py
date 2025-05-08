"""
Shared test fixtures and test data for various test modules.
"""

import pytest
import torch
import os
from unittest.mock import patch, MagicMock
from PIL import Image
from pathlib import Path

from gazegraph.models.clip import ClipModel

# Constants for tests
TEST_IMAGES_DIR = Path("data/tests/images")
SAMPLE_LABELS = ["apple", "microwave", "knife", "tomato", "plate"]

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
    images = {}
    for img_path in TEST_IMAGES_DIR.glob("*.jpg"):
        images[img_path.stem] = Image.open(img_path)
    for img_path in TEST_IMAGES_DIR.glob("*.png"):
        images[img_path.stem] = Image.open(img_path)
    return images

@pytest.fixture
def test_image():
    """Fixture that provides a single test image."""
    for img_path in TEST_IMAGES_DIR.glob("*.jpg"):
        return Image.open(img_path)
    for img_path in TEST_IMAGES_DIR.glob("*.png"):
        return Image.open(img_path)
    return None

@pytest.fixture
def mock_clip_model():
    """Fixture that provides a mocked CLIP model to avoid actual model loading."""
    with patch("clip.load") as mock_load:
        # Create mock model and preprocessor
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        
        # Configure mock model's encode_text method
        mock_model.encode_text.return_value = torch.ones((1, 512))
        
        # Configure mock model's encode_image method
        mock_model.encode_image.return_value = torch.ones((1, 512))
        
        # Configure mock model's __call__ method
        mock_model.return_value = (torch.tensor([[0.1, 0.8, 0.1, 0.5, 0.3]]), None)
        
        # Return the mocks from clip.load
        mock_load.return_value = (mock_model, mock_preprocess)
        
        # Configure the preprocessor to return a tensor
        mock_preprocess.return_value = torch.ones((3, 224, 224))
        
        # Create and return the model
        model = ClipModel(device="cpu")
        model.load()
        
        # Add a method to ensure tensors are moved to the correct device
        def mock_to(tensor, device):
            return tensor
            
        # Make mock_model.to() work correctly
        mock_model.to = MagicMock(return_value=mock_model)
        
        yield model 