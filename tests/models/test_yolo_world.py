import pytest
from PIL import Image
from pathlib import Path
from gazegraph.models.yolo_world import YOLOWorldModel
from gazegraph.config.config_utils import get_config


@pytest.fixture
def model_path():
    """Fixture for YOLO World model path from config."""
    return Path(get_config().models.yolo_world.model_path)


@pytest.fixture
def test_image_path():
    """Fixture for test image path."""
    path = Path(__file__).parent.parent / "resources" / "clip" / "apple.jpg"
    if not path.exists():
        pytest.skip(f"Test image not found: {path}")
    return path


@pytest.fixture
def test_image(test_image_path):
    """Fixture for test image."""
    return Image.open(test_image_path).convert("RGB")


@pytest.mark.unit
def test_initialization():
    """Test model initialization."""
    model = YOLOWorldModel(conf_threshold=0.35, iou_threshold=0.7, device="cpu")
    assert model.conf_threshold == 0.35
    assert model.iou_threshold == 0.7
    assert model.device == "cpu"
    assert model.names == []
    assert model.session is None


@pytest.mark.gpu
def test_load_model(model_path):
    """Test model loading."""
    if not model_path.exists():
        pytest.skip(f"Model file not found: {model_path}")

    model = YOLOWorldModel(device="cpu")
    model.load_model(model_path)

    assert model.session is not None
    assert model.text_embedder is not None
    assert model.input_names is not None
    assert model.output_names is not None
    assert model.num_classes is not None


@pytest.mark.gpu
def test_set_classes(model_path):
    """Test setting object classes."""
    if not model_path.exists():
        pytest.skip(f"Model file not found: {model_path}")

    model = YOLOWorldModel(device="cpu")
    model.load_model(model_path)

    class_names = ["apple", "bowl", "microwave"]
    model.set_classes(class_names)

    assert model.classes == class_names
    assert model.class_embeddings is not None


@pytest.mark.gpu
def test_run_inference(model_path, test_image):
    """Test object detection inference."""
    if not model_path.exists():
        pytest.skip(f"Model file not found: {model_path}")

    model = YOLOWorldModel(conf_threshold=0.1, iou_threshold=0.5, device="cpu")
    model.load_model(model_path)

    class_names = ["apple", "bowl", "microwave"]
    model.set_classes(class_names)

    # Convert PIL image to tensor format expected by the model
    import torch
    import numpy as np

    # Convert PIL to numpy
    img_array = np.array(test_image)
    # Convert to torch tensor and normalize
    frame = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0

    detections = model.run_inference(frame)

    assert isinstance(detections, list)
    for detection in detections:
        assert "bbox" in detection and len(detection["bbox"]) == 4
        assert "score" in detection and isinstance(detection["score"], float)
        assert "class_id" in detection and isinstance(detection["class_id"], int)
        assert "class_name" in detection and isinstance(detection["class_name"], str)
        assert detection["class_name"] in class_names
