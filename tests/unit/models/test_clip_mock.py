"""
Mock-based unit tests for the ClipModel class.

These tests use mocked objects to avoid loading the actual CLIP model,
making them faster and less resource-intensive for continuous integration.
"""

import pytest
import torch
from unittest.mock import patch
from tests.resources.fixtures import mock_clip_model, test_image

def test_mocked_encode_text(mock_clip_model):
    """Test text encoding with mocked model."""
    texts = ["a photo of an apple", "a microwave"]
    with patch("clip.tokenize", return_value=torch.ones((2, 1))):
        encodings = mock_clip_model.encode_text(texts)
    
    assert len(encodings) == 2
    assert all(isinstance(enc, torch.Tensor) for enc in encodings)
    assert all(enc.shape == (1, 512) for enc in encodings)

def test_mocked_encode_image(mock_clip_model, test_image):
    """Test image encoding with mocked model."""
    if test_image:
        encoding = mock_clip_model.encode_image(test_image)
        assert isinstance(encoding, torch.Tensor)
        assert encoding.shape == (1, 512)

def test_mocked_classify(mock_clip_model, test_image):
    """Test classification with mocked model."""
    if test_image:
        labels = ["apple", "microwave", "knife", "tomato", "plate"]
        with patch("clip.tokenize", return_value=torch.ones((5, 1))):
            scores, best_label = mock_clip_model.classify(labels, test_image)
            
        assert len(scores) == 5
        assert best_label == "microwave"  # Based on our mock returning highest score at index 1 