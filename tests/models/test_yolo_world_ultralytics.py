"""
Unit tests for YOLOWorld models.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from gazegraph.models.yolo_world_model import YOLOWorldModel
from gazegraph.models.yolo_world_ultralytics import YOLOWorldUltralyticsModel
from gazegraph.config.config_utils import get_config

@pytest.fixture
def model_path():
    """Fixture to provide the model path from config."""
    config = get_config()
    return Path(config.models.yolo_world.paths.ultralytics)

@pytest.fixture
def yolo_world_model():
    """Fixture to provide a YOLOWorldModel instance."""
    model = YOLOWorldUltralyticsModel()
    return model

@pytest.mark.unit
def test_initialization():
    """Test model initialization with factory."""
    # Test with explicit parameters
    model_explicit = YOLOWorldModel.create(
        backend="ultralytics",
        conf_threshold=0.35,
        iou_threshold=0.7,
        device="cpu"
    )
    assert isinstance(model_explicit, YOLOWorldUltralyticsModel)
    assert model_explicit.conf_threshold == 0.35
    assert model_explicit.iou_threshold == 0.7
    assert model_explicit.device == "cpu"
    assert model_explicit.names == []
    
    # Test using default config values
    model_from_config = YOLOWorldModel.create(
        backend="ultralytics", 
        device="cpu"
    )
    assert isinstance(model_from_config, YOLOWorldUltralyticsModel)
    assert model_from_config.device == "cpu"
    
    # Test invalid backend
    with pytest.raises(ValueError):
        YOLOWorldModel.create(backend="invalid")

@pytest.mark.gpu
def test_set_classes(model_path):
    """Test setting object classes."""
    # Skip if model file doesn't exist
    if not model_path.exists():
        pytest.skip(f"Model file not found: {model_path}")
    
    # Create and load model
    model = YOLOWorldModel.create(
        backend="ultralytics",
        model_path=model_path
    )
    
    # Set class names
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
        conf_threshold=0.05  # Use lower threshold for tests
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
    
    # Run detection with different input types
    detections_np = model.predict(image_np)
    detections_pil = model.predict(image)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    detections_tensor = model.predict(image_tensor)
    
    # Verify detections format
    for detections in [detections_np, detections_pil, detections_tensor]:
        assert isinstance(detections, list)
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

@pytest.mark.gpu
def test_all_yolo_world_images(model_path):
    """Test object detection on all images in the yolo-world directory."""
    if not model_path.exists():
        pytest.skip(f"Model file not found: {model_path}")
    
    # Initialize model with lower threshold for tests
    model = YOLOWorldModel.create(
        backend="ultralytics",
        model_path=model_path,
        conf_threshold=0.05
    )
    
    test_dir = Path("data/tests/yolo-world")
    if not test_dir.exists() or not any(test_dir.iterdir()):
        pytest.skip(f"No test images found in {test_dir}")
    
    # Extract objects from filenames (e.g. "apple-bowl.png" -> ["apple", "bowl"])
    def get_objects_from_filename(filename):
        return [obj.lower() for obj in filename.stem.split('-')]
    
    # Track detection results
    all_detected = True
    
    # Process each test image
    for img_file in test_dir.glob('*.png'):
        # Load image and expected objects
        image = Image.open(img_file).convert("RGB")
        expected_objects = get_objects_from_filename(img_file)
        
        # Set class names and run prediction
        model.set_classes(expected_objects)
        detections = model.predict(image)
        
        # Check if all expected objects were detected
        detected_objects = {detection["class_name"].lower() for detection in detections}
        missing_objects = set(expected_objects) - detected_objects
        
        # Log results
        print(f"Image: {img_file.name}")
        print(f"Expected: {expected_objects}")
        print(f"Detected: {list(detected_objects)}")
        print(f"Missing: {list(missing_objects) if missing_objects else 'None'}")
        print()
        
        if missing_objects:
            all_detected = False
    
    assert all_detected, "Some expected objects were not detected"
    