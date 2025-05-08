"""
Unit tests for the ClipModel class.
"""

import pytest
import torch
from pathlib import Path

from gazegraph.models.clip import ClipModel
from tests.resources.fixtures import clip_model, test_images, SAMPLE_LABELS

def test_initialization():
    """Test that the CLIP model initializes properly."""
    model = ClipModel(device="cpu")
    assert model.device == "cpu"
    assert model.model is None
    assert model.preprocess is None

def test_model_loading(clip_model):
    """Test that the model loads successfully."""
    assert clip_model.model is not None
    assert clip_model.preprocess is not None

def test_encode_text(clip_model):
    """Test text encoding functionality."""
    texts = ["a photo of an apple", "a photo of a microwave"]
    encodings = clip_model.encode_text(texts)
    
    # Check encoding format
    assert len(encodings) == 2
    assert all(isinstance(enc, torch.Tensor) for enc in encodings)
    assert all(enc.shape[1] == 512 for enc in encodings)  # CLIP's default embedding size

def test_encode_image(clip_model, test_images):
    """Test image encoding functionality."""
    for name, img in test_images.items():
        encoding = clip_model.encode_image(img)
        
        # Check encoding format
        assert isinstance(encoding, torch.Tensor)
        assert encoding.shape[1] == 512  # CLIP's default embedding size

def test_tensor_image_input(clip_model, test_images):
    """Test using preprocessed tensor as input instead of PIL Image."""
    if test_images:
        img = next(iter(test_images.values()))
        preprocessed = clip_model.preprocess(img).unsqueeze(0)
        
        # Test image encoding with tensor
        encoding = clip_model.encode_image(preprocessed)
        assert isinstance(encoding, torch.Tensor)
        assert encoding.shape[1] == 512
        
        # Test classification with tensor
        scores, _ = clip_model.classify(SAMPLE_LABELS, preprocessed)
        assert len(scores) == len(SAMPLE_LABELS) 