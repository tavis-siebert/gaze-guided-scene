"""
Integration tests for ClipModel classification functionality.

These tests verify that the model correctly classifies real images
from the test data directory.
"""

import pytest
from tests.resources.fixtures import clip_model, test_images, SAMPLE_LABELS

def test_image_classification(clip_model, test_images):
    """Test image classification functionality with real test images."""
    # Test apple image classification
    if "apple" in test_images:
        scores, best_label = clip_model.classify(SAMPLE_LABELS, test_images["apple"])
        
        assert len(scores) == len(SAMPLE_LABELS)
        assert all(isinstance(score, float) for score in scores)
        assert best_label == "apple"
    
    # Test microwave image classification
    if "microwave" in test_images:
        scores, best_label = clip_model.classify(SAMPLE_LABELS, test_images["microwave"])
        
        assert len(scores) == len(SAMPLE_LABELS)
        assert all(isinstance(score, float) for score in scores)
        assert best_label == "microwave"

def test_custom_labels_classification(clip_model, test_images):
    """Test classification with custom labels."""
    if "ego-holding-tomatoe-and-knife" in test_images:
        custom_labels = ["cutting", "cooking", "eating", "reading"]
        scores, best_label = clip_model.classify(
            custom_labels, 
            test_images["ego-holding-tomatoe-and-knife"]
        )
        
        assert len(scores) == len(custom_labels)
        assert all(isinstance(score, float) for score in scores)
        # The model classifies the image as "cooking", which is reasonable
        assert best_label == "cooking" 