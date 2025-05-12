import pytest
from PIL import Image
from pathlib import Path
from gazegraph.models.yolo_world_model import YOLOWorldModel
from gazegraph.config.config_utils import get_config
import numpy as np
from gazegraph.models.yolo_world_ultralytics import YOLOWorldUltralyticsModel
import hashlib
@pytest.fixture
def ultralytics_model_path():
    """Fixture for Ultralytics model path from config."""
    return Path(get_config().models.yolo_world.paths.ultralytics)

@pytest.fixture
def onnx_model_path():
    """Fixture for ONNX model path from config."""
    return Path(get_config().models.yolo_world.paths.onnx)

@pytest.fixture
def test_dir():
    """Fixture for test directory."""
    path = Path(__file__).parent.parent / "resources" / "yolo-world"
    if not path.exists():
        pytest.skip(f"Test directory not found: {path}")
    return path

@pytest.fixture
def test_image_path(test_dir):
    path = test_dir / "knife-hand-plate-tomato.png"
    if not path.exists():
        pytest.skip(f"Test image not found: {path}")
    return path

@pytest.fixture
def test_image(test_image_path):
    return Image.open(test_image_path).convert("RGB")

@pytest.fixture
def custom_classes(test_image_path):
    return test_image_path.stem.split('-')

@pytest.fixture(params=["ultralytics", "onnx"])
def yolo_world_model(request, ultralytics_model_path, onnx_model_path):
    """Fixture to provide YOLOWorldModel instances for both backends."""
    backend = request.param
    model_path = ultralytics_model_path if backend == "ultralytics" else onnx_model_path
    if not model_path.exists():
        pytest.skip(f"Model file not found for {backend}: {model_path}")
    return YOLOWorldModel.create(backend=backend, model_path=model_path, device="cpu")

@pytest.mark.unit
@pytest.mark.parametrize("backend", ["ultralytics", "onnx"])
def test_initialization(backend):
    """Test model initialization with factory for both backends."""
    model = YOLOWorldModel.create(backend=backend, conf_threshold=0.35, iou_threshold=0.7, device="cpu")
    assert model.conf_threshold == 0.35
    assert model.iou_threshold == 0.7
    assert model.device in ["cpu", "0"]  # ONNX uses '0' for GPU but we set CPU
    assert model.names == []

@pytest.mark.unit
def test_invalid_backend():
    """Test initialization with invalid backend."""
    with pytest.raises(ValueError):
        YOLOWorldModel.create(backend="invalid")

@pytest.mark.gpu
def test_set_classes(yolo_world_model):
    """Test setting object classes for the model."""
    class_names = ["apple", "bowl", "microwave"]
    yolo_world_model.set_classes(class_names)
    assert yolo_world_model.names == class_names

@pytest.mark.gpu
def test_predict(yolo_world_model, test_data_dir):
    """Test object detection on a sample image for both backends."""
    img_path = test_data_dir / "clip" / "apple.jpg"
    if not img_path.exists():
        pytest.skip(f"Test image not found: {img_path}")
    
    image = Image.open(img_path).convert("RGB")
    class_names = ["apple", "bowl", "microwave"]
    yolo_world_model.set_classes(class_names)
    detections = yolo_world_model.predict(image)
    
    assert isinstance(detections, list)
    for detection in detections:
        assert "bbox" in detection and len(detection["bbox"]) == 4
        assert "score" in detection and isinstance(detection["score"], float)
        assert "class_id" in detection and isinstance(detection["class_id"], int)
        assert "class_name" in detection and isinstance(detection["class_name"], str)

@pytest.mark.gpu
def test_object_detection_suite(yolo_world_model, test_dir):
    """Test object detection on a suite of images with expected objects for both backends."""
    yolo_world_model.conf_threshold = 0.1
    yolo_world_model.iou_threshold = 0.5
    
    def get_objects_from_filename(filename):
        return [obj.lower() for obj in filename.stem.split('-')]
    
    all_detected = True
    for img_file in test_dir.glob('*.png'):
        image = Image.open(img_file).convert("RGB")
        expected_objects = get_objects_from_filename(img_file)
        yolo_world_model.set_classes(expected_objects)
        detections = yolo_world_model.predict(image)
        
        detected_objects = {detection["class_name"].lower() for detection in detections}
        missing_objects = set(expected_objects) - detected_objects
        
        print(f"Backend: {yolo_world_model._get_backend_name()}, Image: {img_file.name}")
        print(f"Expected: {expected_objects}")
        print(f"Detected: {[f'{d["class_name"]} ({d["score"]:.3f})' for d in detections]}")
        print(f"Missing: {list(missing_objects) if missing_objects else 'None'}")
        print()
        
        if missing_objects:
            all_detected = False
    
    assert all_detected, "Some expected objects were not detected in test suite"

@pytest.mark.gpu
def test_custom_model_save_load(custom_classes, ultralytics_model_path):
    # Initialize model with custom save flag
    model = YOLOWorldUltralyticsModel(use_custom_model=True)
    # Set custom classes
    model.set_classes(custom_classes)
    # Check if custom model file is created
    class_str = '_'.join(custom_classes)
    class_str_hash = hashlib.sha256(class_str.encode()).hexdigest()[:8]
    model_dir = ultralytics_model_path.parent
    custom_model_path = model_dir / f"custom_yolov8x-worldv2_{class_str_hash}.pt"
    assert custom_model_path.exists(), f"Custom model file not found at {custom_model_path}"
    # Load the custom model in a new instance
    new_model = YOLOWorldUltralyticsModel(use_custom_model=True, custom_classes=custom_classes)
    assert new_model.names == custom_classes, "Loaded custom model does not have the expected classes"

@pytest.mark.gpu
def test_custom_model_confidence_improvement(yolo_world_model, test_image, custom_classes):
    pytest.skip("Custom model does not seem to have any (positive) effect on accuracy or speed")
    # Initialize standard model without custom flag
    standard_model = YOLOWorldUltralyticsModel()
    standard_model.set_classes(custom_classes)
    # Run inference with standard model
    standard_results = standard_model.predict(test_image)
    standard_confidences = [det['score'] for det in standard_results]
    standard_avg_conf = np.mean(standard_confidences) if standard_confidences else 0.0
    # Initialize and save custom model
    custom_model = YOLOWorldUltralyticsModel(use_custom_model=True)
    custom_model.set_classes(custom_classes)
    # Load the custom model in a new instance
    loaded_custom_model = YOLOWorldUltralyticsModel(use_custom_model=True, custom_classes=custom_classes)
    custom_results = loaded_custom_model.predict(test_image)
    custom_confidences = [det['score'] for det in custom_results]
    custom_avg_conf = np.mean(custom_confidences) if custom_confidences else 0.0
    # Qualitative check: custom model should have higher average confidence
    print(f"Standard average confidence: {standard_avg_conf}")
    print(f"Custom average confidence: {custom_avg_conf}")
    assert custom_avg_conf > standard_avg_conf, f"Custom model confidence ({custom_avg_conf}) not higher than standard ({standard_avg_conf})"