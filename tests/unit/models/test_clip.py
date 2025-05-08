"""
Unit tests for the ClipModel class.
"""

import pytest
import torch
from pathlib import Path

from gazegraph.models.clip import ClipModel
from tests.resources.fixtures import clip_model, mock_clip_model, test_images, SAMPLE_LABELS

@pytest.mark.unit
def test_initialization():
    """Test that the CLIP model initializes properly."""
    model = ClipModel(device="cpu")
    assert model.device == "cpu"
    assert model.model is None
    assert model.preprocess is None

@pytest.mark.unit
@pytest.mark.real_model
def test_model_loading(clip_model):
    """Test that the model loads successfully."""
    assert clip_model.model is not None
    assert clip_model.preprocess is not None

@pytest.mark.unit
@pytest.mark.real_model
def test_encode_text(clip_model):
    """Test text encoding functionality."""
    texts = ["a photo of an apple", "a photo of a microwave"]
    encodings = clip_model.encode_text(texts)
    
    # Check encoding format
    assert len(encodings) == 2
    assert all(isinstance(enc, torch.Tensor) for enc in encodings)
    assert all(enc.shape[1] == 512 for enc in encodings)  # CLIP's default embedding size

@pytest.mark.unit
@pytest.mark.real_model
def test_encode_image(clip_model, test_images):
    """Test image encoding functionality."""
    for name, img in test_images.items():
        encoding = clip_model.encode_image(img)
        
        # Check encoding format
        assert isinstance(encoding, torch.Tensor)
        assert encoding.shape[1] == 512  # CLIP's default embedding size

@pytest.mark.unit
@pytest.mark.real_model
def test_tensor_image_input(clip_model, test_images):
    """Test using preprocessed tensor as input instead of PIL Image."""
    if test_images:
        img = next(iter(test_images.values()))
        # Create preprocessed tensor and move to the same device as the model
        preprocessed = clip_model.preprocess(img).unsqueeze(0).to(clip_model.device)
        
        # Test image encoding with tensor
        encoding = clip_model.encode_image(preprocessed)
        assert isinstance(encoding, torch.Tensor)
        assert encoding.shape[1] == 512
        
        # Test classification with tensor
        scores, _ = clip_model.classify(SAMPLE_LABELS, preprocessed)
        assert len(scores) == len(SAMPLE_LABELS)

@pytest.mark.unit
@pytest.mark.mock_only
def test_mock_model_loading(mock_clip_model):
    """Test that the mock model loads successfully."""
    assert mock_clip_model.model is not None
    assert mock_clip_model.preprocess is not None

@pytest.mark.unit
@pytest.mark.mock_only
def test_mock_encode_text(mock_clip_model):
    """Test text encoding functionality with mock model."""
    texts = ["a photo of an apple", "a photo of a microwave"]
    encodings = mock_clip_model.encode_text(texts)
    
    assert len(encodings) == 2
    assert all(isinstance(enc, torch.Tensor) for enc in encodings)

@pytest.mark.unit
@pytest.mark.mock_only
def test_mock_classify(mock_clip_model):
    """Test classification with mock model."""
    image = torch.ones((1, 3, 224, 224))  # Mock image tensor
    scores, best_label = mock_clip_model.classify(SAMPLE_LABELS, image)
    
    assert len(scores) == len(SAMPLE_LABELS)
    assert best_label in SAMPLE_LABELS

@pytest.mark.unit
@pytest.mark.mock_only
def test_mock_tensor_input(mock_clip_model):
    """Test using tensor input with the mock model."""
    # Create a mock tensor
    tensor = torch.ones((1, 3, 224, 224))
    
    # Move to the device (shouldn't matter for mock)
    tensor = tensor.to(mock_clip_model.device)
    
    # Test image encoding
    encoding = mock_clip_model.encode_image(tensor)
    assert isinstance(encoding, torch.Tensor)
    
    # Test classification
    scores, label = mock_clip_model.classify(SAMPLE_LABELS, tensor)
    assert len(scores) == len(SAMPLE_LABELS)
    assert label in SAMPLE_LABELS 