import pytest
from PIL import Image
from pathlib import Path
from gazegraph.models.yolo_world_model import YOLOWorldModel
from gazegraph.config.config_utils import get_config

@pytest.fixture
def ultralytics_model_path():
    """Fixture for Ultralytics model path from config."""
    return Path(get_config().models.yolo_world.paths.ultralytics)

@pytest.fixture
def onnx_model_path():
    """Fixture for ONNX model path from config."""
    return Path(get_config().models.yolo_world.paths.onnx)

@pytest.fixture(params=["ultralytics", "onnx"])
def yolo_world_model(request, ultralytics_model_path, onnx_model_path):
    """Fixture to provide YOLOWorldModel instances for both backends."""
    backend = request.param
    model_path = ultralytics_model_path if backend == "ultralytics" else onnx_model_path
    if not model_path.exists():
        pytest.skip(f"Model file not found for {backend}: {model_path}")
    return YOLOWorldModel.create(backend=backend, model_path=model_path, device="cpu")

@pytest.mark.unit
def test_initialization():
    """Test model initialization with factory for both backends."""
    for backend in ["ultralytics", "onnx"]:
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
def test_object_detection_suite(yolo_world_model):
    """Test object detection on a suite of images with expected objects for both backends."""
    test_dir = Path("data/tests/yolo-world")
    if not test_dir.exists() or not any(test_dir.iterdir()):
        pytest.skip(f"No test images found in {test_dir}")
    
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