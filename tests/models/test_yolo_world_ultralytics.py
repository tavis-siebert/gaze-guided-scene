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
        conf_threshold=0.05,
        iou_threshold=0.2,
        device=device
    )
    return model

@pytest.mark.unit
def test_initialization():
    """Test model initialization with factory."""
    # Test ultralytics backend
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
    
    # Load the model
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
        iou_threshold=0.2
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
    """Test object detection on all images in the yolo-world directory.
    """
    # Skip if model file doesn't exist
    if not model_path.exists():
        pytest.skip(f"Model file not found: {model_path}")
    
    # Create and load model
    model = YOLOWorldModel.create(
        backend="ultralytics",
        model_path=model_path,
        conf_threshold=0.05,
        iou_threshold=0.2
    )
    print(f"Using confidence threshold: {model.conf_threshold}")
    print(f"Using IoU threshold: {model.iou_threshold}")
    
    # Path to the yolo-world test images directory
    test_dir = Path("data/tests/yolo-world")
    if not test_dir.exists() or not os.listdir(test_dir):
        pytest.skip(f"No test images found in {test_dir}")
    
    def get_objects_from_filename(filename):
        # Extract the base name without extension
        base_name = filename.stem
        # Split by dash and return unique objects
        return [obj.lower() for obj in base_name.split('-')]
    
    # Track overall detection statistics
    all_results = {}
    
    # Process each image file
    for img_file in test_dir.glob('*.png'):
        print(f"\n\n{'=' * 50}")
        print(f"TESTING IMAGE: {img_file.name}")
        print(f"{'=' * 50}")
        
        # Load and convert image to RGB
        image = Image.open(img_file).convert("RGB")
        image_np = np.array(image)
        print(f"Image shape: {image_np.shape}")
        
        # Get objects from filename
        expected_objects = get_objects_from_filename(img_file)
        print(f"Objects from filename: {expected_objects}")
        
        # Set class names and run prediction
        model.set_classes(expected_objects)
        
        # Run inference
        detections = model.predict(image_np, image_size=640)
        
        # Print detailed information about all detections
        print(f"\nDetected {len(detections)} objects:")
        for i, detection in enumerate(detections):
            class_name = detection["class_name"]
            score = detection["score"]
            bbox = detection["bbox"]
            print(f"  {i+1}. {class_name}: confidence={score:.4f}, bbox={bbox}")
        
        # Check if objects from filename were detected
        missing_objects = []
        found_objects = []
        
        for expected_obj in expected_objects:
            # Check if this object or any of its synonyms were detected
            obj_found = False
            matching_synonyms = [expected_obj]
            
            matching_detections = []
            for detection in detections:
                det_name = detection["class_name"].lower()
                for obj_variant in matching_synonyms:
                    if obj_variant in det_name:
                        obj_found = True
                        matching_detections.append(
                            f"{det_name} (score: {detection['score']:.4f})"
                        )
            
            if obj_found:
                found_objects.append(f"{expected_obj} detected as: {', '.join(matching_detections)}")
            else:
                missing_objects.append(expected_obj)
        
        # Store results for this image
        image_results = {
            "filename": img_file.name,
            "expected_objects": expected_objects,
            "found_objects": found_objects,
            "missing_objects": missing_objects,
            "detection_count": len(detections),
            "detections": [
                {
                    "class_name": d["class_name"],
                    "score": d["score"],
                    "bbox": d["bbox"]
                }
                for d in detections
            ]
        }
        all_results[img_file.name] = image_results
        
        # Print summary
        print(f"\nSummary for {img_file.name}:")
        print(f"  - Found objects: {found_objects if found_objects else 'None'}")
        print(f"  - Missing objects: {missing_objects if missing_objects else 'None'}")
        
        # Print detection rate without failing
        detected_count = len(found_objects)
        total_count = len(expected_objects)
        detection_rate = detected_count / total_count if total_count > 0 else 0
        print(f"  - Detection rate: {detected_count}/{total_count} = {detection_rate:.2%}")
    
    print(f"\n\n{'=' * 50}")
    print("OVERALL DETECTION SUMMARY")
    print(f"{'=' * 50}")
    
    for filename, results in all_results.items():
        detected = len(results["found_objects"]) 
        total = len(results["expected_objects"])
        rate = detected / total if total > 0 else 0
        print(f"{filename}: {detected}/{total} objects detected ({rate:.2%})")
    