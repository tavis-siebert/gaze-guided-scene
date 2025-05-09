"""
Unit tests for YOLOWorldUltralyticsModel.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock

from gazegraph.models.yolo_world_ultralytics import YOLOWorldUltralyticsModel, format_class_name
from gazegraph.config.config_utils import get_config

@pytest.fixture
def model_path():
    """Fixture to provide the model path from config."""
    config = get_config()
    return Path(config.models.yolo_world.model_file_ultralytics)

@pytest.fixture
def yolo_world_ultralytics_model():
    """Fixture to provide a YOLOWorldUltralyticsModel instance."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLOWorldUltralyticsModel(conf_threshold=0.35, iou_threshold=0.7, device=device)
    return model

@pytest.mark.unit
def test_initialization():
    """Test model initialization."""
    model = YOLOWorldUltralyticsModel(conf_threshold=0.35, iou_threshold=0.7, device="cpu")
    assert model.conf_threshold == 0.35
    assert model.iou_threshold == 0.7
    assert model.device == "cpu"
    assert model.model is None
    assert model.names == []

@pytest.mark.gpu
def test_model_loading(yolo_world_ultralytics_model, model_path):
    """Test model loading."""
    # Skip if model file doesn't exist - this avoids test failures in CI
    if not model_path.exists():
        pytest.skip(f"Model file not found: {model_path}")
        
    yolo_world_ultralytics_model.load_model(model_path)
    assert yolo_world_ultralytics_model.model is not None

@pytest.mark.gpu
def test_set_classes(yolo_world_ultralytics_model, model_path):
    """Test setting object classes."""
    # Skip if model file doesn't exist
    if not model_path.exists():
        pytest.skip(f"Model file not found: {model_path}")
        
    yolo_world_ultralytics_model.load_model(model_path)
    
    class_names = ["apple", "bowl", "microwave"]
    yolo_world_ultralytics_model.set_classes(class_names)
    
    assert yolo_world_ultralytics_model.names == class_names

@pytest.mark.gpu
def test_run_inference(yolo_world_ultralytics_model, model_path, test_data_dir):
    """Test running inference on an image."""
    # Skip if model file doesn't exist
    if not model_path.exists():
        pytest.skip(f"Model file not found: {model_path}")
        
    yolo_world_ultralytics_model.load_model(model_path)
    
    # Load a test image
    img_path = test_data_dir / "clip" / "apple.jpg"
    if not img_path.exists():
        pytest.skip(f"Test image not found: {img_path}")
        
    image = Image.open(img_path)
    image_np = np.array(image)
    
    # Define text labels and object labels
    text_labels = ["a photo of a apple", "a photo of a bowl", "a photo of a microwave"]
    obj_labels = {0: "apple", 1: "bowl", 2: "microwave"}
    
    # Run inference
    detections = yolo_world_ultralytics_model.run_inference(
        image_np, text_labels, obj_labels, image_size=640
    )
    
    # Check that we get a list of detections
    assert isinstance(detections, list)
    
    # If there are detections, check their format
    if detections:
        for detection in detections:
            assert "bbox" in detection
            assert "score" in detection
            assert "class_id" in detection
            assert "class_name" in detection
            
            assert isinstance(detection["bbox"], list) and len(detection["bbox"]) == 4
            assert isinstance(detection["score"], float)
            assert isinstance(detection["class_id"], int)
            assert isinstance(detection["class_name"], str) 