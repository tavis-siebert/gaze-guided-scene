"""
Unit tests for YOLOWorld models.
"""

import pytest
import torch
import numpy as np
import os
from pathlib import Path
from PIL import Image

from gazegraph.models.yolo_world_model import YOLOWorldModel
from gazegraph.models.yolo_world_ultralytics import YOLOWorldUltralyticsModel
from gazegraph.config.config_utils import get_config

@pytest.fixture
def model_path():
    """Fixture to provide the model path from config."""
    config = get_config()
    return Path(config.models.yolo_world.model_file_ultralytics)

@pytest.fixture
def yolo_world_model():
    """Fixture to provide a YOLOWorldModel instance."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLOWorldModel.create(
        backend="ultralytics",
        device=device
    )
    return model

@pytest.mark.unit
def test_initialization():
    """Test model initialization with factory."""
    # Test ultralytics backend with explicit parameters
    model_ultralytics = YOLOWorldModel.create(
        backend="ultralytics",
        conf_threshold=0.35,
        iou_threshold=0.7,
        device="cpu"
    )
    assert isinstance(model_ultralytics, YOLOWorldUltralyticsModel)
    assert model_ultralytics.conf_threshold == 0.35
    assert model_ultralytics.iou_threshold == 0.7
    assert model_ultralytics.device == "cpu"
    assert model_ultralytics.names == []
    
    # Test direct initialization of the ultralytics model
    model_direct = YOLOWorldUltralyticsModel(
        conf_threshold=0.35,
        iou_threshold=0.7,
        device="cpu"
    )
    assert model_direct.conf_threshold == 0.35
    assert model_direct.iou_threshold == 0.7
    assert model_direct.device == "cpu"
    assert model_direct.names == []
    
    # Test using default config-based parameters
    model_from_config = YOLOWorldModel.create(
        backend="ultralytics",
        device="cpu"
    )
    # We don't assert specific values since they're now from config
    assert isinstance(model_from_config, YOLOWorldUltralyticsModel)
    assert model_from_config.device == "cpu"
    
    # Test invalid backend
    with pytest.raises(ValueError):
        YOLOWorldModel.create(backend="invalid")

@pytest.mark.gpu
def test_model_loading(yolo_world_model, model_path):
    """Test model loading."""
    # Skip if model file doesn't exist - this avoids test failures in CI
    if not model_path.exists():
        pytest.skip(f"Model file not found: {model_path}")
    
    # Model should be instantiated but not loaded yet in the fixture
    assert yolo_world_model.model is None
    
    # Load the model with explicit parameters
    model = YOLOWorldModel.create(
        backend="ultralytics",
        model_path=model_path,
        conf_threshold=0.05,
        iou_threshold=0.2
    )
    
    # Check that model was loaded
    assert model.model is not None

@pytest.mark.gpu
def test_set_classes(model_path):
    """Test setting object classes."""
    # Skip if model file doesn't exist
    if not model_path.exists():
        pytest.skip(f"Model file not found: {model_path}")
    
    # Create model with path to load immediately
    model = YOLOWorldModel.create(
        backend="ultralytics",
        model_path=model_path,
        conf_threshold=0.05,
        iou_threshold=0.2
    )
    
    class_names = ["apple", "bowl", "microwave"]
    model.set_classes(class_names)
    
    assert model.names == class_names

@pytest.mark.gpu
def test_predict(model_path, test_data_dir):
    """Test running inference on an image."""
    # Skip if model file doesn't exist
    if not model_path.exists():
        pytest.skip(f"Model file not found: {model_path}")
    
    # Create and load model
    model = YOLOWorldModel.create(
        backend="ultralytics",
        model_path=model_path,
        conf_threshold=0.05,
        iou_threshold=0.5
    )
    
    # Load a test image
    img_path = test_data_dir / "clip" / "apple.jpg"
    if not img_path.exists():
        pytest.skip(f"Test image not found: {img_path}")
    
    # Load image as PIL
    image = Image.open(img_path)
    image_np = np.array(image)
    
    # Set class names
    class_names = ["apple", "bowl", "microwave"]
    model.set_classes(class_names)
    
    # Test with numpy array
    detections = model.predict(image_np, image_size=640)
    
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
    
    # Test with PIL image
    detections_pil = model.predict(image, image_size=640)
    assert isinstance(detections_pil, list)
    
    # Test with torch tensor
    # Convert image to tensor
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    detections_tensor = model.predict(image_tensor, image_size=640)
    assert isinstance(detections_tensor, list)

@pytest.mark.gpu
def test_all_yolo_world_images(model_path):
    """Test object detection on all images in the yolo-world directory."""
    if not model_path.exists():
        pytest.skip(f"Model file not found: {model_path}")
    
    model = YOLOWorldModel.create(
        backend="ultralytics",
        model_path=model_path,
        conf_threshold=0.05,
        iou_threshold=0.2
    )
    
    test_dir = Path("data/tests/yolo-world")
    if not test_dir.exists() or not os.listdir(test_dir):
        pytest.skip(f"No test images found in {test_dir}")
    
    def get_objects_from_filename(filename):
        return [obj.lower() for obj in filename.stem.split('-')]
    
    all_images_detected = True
    
    for img_file in test_dir.glob('*.png'):
        # Load image and get expected objects
        image = Image.open(img_file).convert("RGB")
        image_np = np.array(image)
        expected_objects = get_objects_from_filename(img_file)
        
        # Set class names and run prediction
        model.set_classes(expected_objects)
        detections = model.predict(image_np, image_size=640)
        
        # Print detection details
        print(f"Image: {img_file.name}")
        print(f"Expected objects: {expected_objects}")
        
        # Check which objects were detected
        detected_objects = set()
        print("Detected objects:")
        for detection in detections:
            class_name = detection["class_name"].lower()
            score = detection["score"]
            bbox = detection["bbox"]
            print(f"  {class_name}: confidence={score:.4f}, bbox={bbox}")
            for expected in expected_objects:
                if expected in class_name:
                    detected_objects.add(expected)
        
        missing_objects = set(expected_objects) - detected_objects
        print(f"Missing objects: {list(missing_objects) if missing_objects else 'None'}")
        print()
        
        # Check if all objects were detected
        if missing_objects:
            all_images_detected = False
    
    # Final assertion
    assert all_images_detected, "Some expected objects were not detected in the test images"
    