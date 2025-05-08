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

# Get the test resources directory
TEST_RESOURCES_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_RESOURCES_DIR / "data"
TEST_IMAGES_DIR = TEST_DATA_DIR / "images"

# Constants for tests
SAMPLE_LABELS = ["apple", "microwave", "knife", "tomato", "plate"]

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

@pytest.fixture
def test_image():
    """Fixture that provides a single test image."""
    # Ensure the directory exists
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
    
    for img_path in TEST_IMAGES_DIR.glob("*.jpg"):
        return Image.open(img_path)
    for img_path in TEST_IMAGES_DIR.glob("*.png"):
        return Image.open(img_path)
    
    # Create a dummy image if no images exist
    if not list(TEST_IMAGES_DIR.glob("*.jpg")) and not list(TEST_IMAGES_DIR.glob("*.png")):
        img = Image.new('RGB', (224, 224), color=(73, 109, 137))
        os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
        img_path = TEST_IMAGES_DIR / "dummy_test_image.jpg"
        img.save(img_path)
        return img
    
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