"""
Unit tests for ClipModel.
"""

import pytest
import torch
from gazegraph.models.clip import ClipModel


@pytest.mark.unit
def test_initialization():
    model = ClipModel(device="cpu")
    assert model.device == "cpu"
    assert model.name == "ViT-L/14"
    assert model.model is not None
    assert model.preprocess is not None


@pytest.mark.gpu
def test_model_loading(clip_model):
    assert clip_model.model is not None
    assert clip_model.preprocess is not None


@pytest.mark.gpu
def test_encode_texts(clip_model):
    texts = ["apple", "microwave"]
    encodings = clip_model.encode_texts(texts)
    assert len(encodings) == 2
    assert all(isinstance(e, torch.Tensor) for e in encodings)
    assert all(e.shape[1] == 768 for e in encodings)


@pytest.mark.gpu
def test_encode_image(clip_model, test_images):
    for img in test_images.values():
        encoding = clip_model.encode_image(img)
        assert isinstance(encoding, torch.Tensor)
        assert encoding.shape[1] == 768


@pytest.mark.gpu
def test_classify(clip_model, test_images):
    # Gather label candidates from test image names (split on dash)
    labels = []
    for name in test_images.keys():
        labels.extend([f"a photo of a {name.replace('_', ' ')}"])

    for name, img in test_images.items():
        name = f"a photo of a {name.replace('_', ' ')}"
        # Create preprocessed tensor and move to the same device as the model
        preprocessed = clip_model.preprocess(img).unsqueeze(0).to(clip_model.device)

        # Test image encoding with tensor
        encoding = clip_model.encode_image(preprocessed)
        assert isinstance(encoding, torch.Tensor)
        assert encoding.shape[1] == 768

        # Test classification with tensor
        scores, best_label = clip_model.classify(labels, preprocessed)
        assert len(scores) == len(labels), (
            f"Number of scores is not equal to the number of labels: {len(scores)} != {len(labels)}"
        )
        assert best_label in name, (
            f"Best label is not in the image name: {best_label} != {name}"
        )
