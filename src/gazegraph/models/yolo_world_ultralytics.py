import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Any, Optional
from ultralytics import YOLOWorld
import hashlib

from gazegraph.logger import get_logger
from gazegraph.models.yolo_world_model import YOLOWorldModel
from gazegraph.config.config_utils import get_config

logger = get_logger(__name__)

class YOLOWorldUltralyticsModel(YOLOWorldModel):
    """YOLO-World model using Ultralytics backend."""
    
    def __init__(
        self, 
        model_path: Optional[Path] = None,
        conf_threshold: Optional[float] = None, 
        iou_threshold: Optional[float] = None,
        device: Optional[str] = None,
        use_prefix: Optional[bool] = None,
        replace_underscores: Optional[bool] = None,
        use_custom_model: bool = False,
        custom_classes: Optional[List[str]] = None
    ):
        """Initialize YOLO-World Ultralytics model with optional custom model saving/loading.

        Args:
            model_path: Path to the model file.
            conf_threshold: Confidence threshold for detections.
            iou_threshold: IoU threshold for NMS.
            device: Device to run the model on.
            use_prefix: Whether to add a prefix to class names.
            replace_underscores: Whether to replace underscores with spaces in class names.
            use_custom_model: Flag to enable saving/loading a custom model with specific classes.
            custom_classes: List of custom classes to load with the custom model.
        """
        # Initialize model to None before parent constructor
        self.model = None
        self.use_custom_model = use_custom_model
        self.custom_model_path = None
        
        # Call parent constructor which handles all config
        super().__init__(model_path, conf_threshold, iou_threshold, device, use_prefix, replace_underscores)
        
        # Load custom model if specified
        if use_custom_model and custom_classes:
            self._load_custom_model(custom_classes)
        # If model is still None, load the default model
        elif self.model is None:
            config = get_config()
            default_model_path = Path(config.models.yolo_world.paths.ultralytics)
            self._load_model(default_model_path)
    
    def _get_custom_model_path(self, class_names: List[str]) -> Path:
        """Generate path for custom model based on class names."""
        config = get_config()
        model_dir = Path(config.models.yolo_world.paths.ultralytics).parent
        class_str = '_'.join(class_names)
        class_str_hash = hashlib.sha256(class_str.encode()).hexdigest()[:8]
        return model_dir / f"custom_yolov8x-worldv2_{class_str_hash}.pt"
    
    def _load_custom_model(self, class_names: List[str]) -> None:
        """Load or create a custom model for the given class names."""
        self.custom_model_path = self._get_custom_model_path(class_names)
        if self.custom_model_path.exists():
            logger.info(f"Loading custom YOLO-World model from: {self.custom_model_path}")
            self.model = YOLOWorld(str(self.custom_model_path))
            self.names = class_names
            logger.info(f"Custom YOLO-World model loaded successfully")
        else:
            logger.info(f"Custom model not found at {self.custom_model_path}, will save after setting classes")
            # Load the default model since custom one doesn't exist yet
            config = get_config()
            default_model_path = Path(config.models.yolo_world.paths.ultralytics)
            self._load_model(default_model_path)
    
    def _load_model(self, model_path: Path) -> None:
        """Load the YOLO-World model using Ultralytics."""
        try:
            logger.info(f"Loading YOLO-World Ultralytics model from: {model_path}")
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = YOLOWorld(str(model_path))
            logger.info(f"YOLO-World Ultralytics model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO-World model: {e}")
            raise
    
    def _update_model_classes(self, class_names: List[str]) -> None:
        """Update the model with the new class names and save if custom model flag is set."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Class names are already formatted by the parent class
        self.model.set_classes(class_names)
        
        # Save custom model if flag is set
        if self.use_custom_model:
            self.custom_model_path = self._get_custom_model_path(self.names)
            logger.info(f"Saving custom model to: {self.custom_model_path}")
            self.model.save(str(self.custom_model_path))
            logger.info(f"Custom model saved successfully")
    
    def _run_inference(self, image: Image.Image, image_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Run inference with the Ultralytics model."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if not isinstance(image, Image.Image):
            raise TypeError(f"YOLOWorldUltralyticsModel.predict requires PIL.Image.Image, got {type(image)}")

        # Run inference
        results = self.model.predict(
            source=image,
            imgsz=image_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device
        )
        
        # Process results
        detections = []
        if results and len(results) > 0:
            result = results[0]
            
            if len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    class_id = int(class_ids[i])
                    class_name = self.names[class_id] if class_id < len(self.names) else f"unknown_{class_id}"
                    
                    detections.append({
                        "bbox": [x1, y1, x2-x1, y2-y1],  # [x, y, width, height]
                        "score": float(scores[i]),
                        "class_id": class_id,
                        "class_name": class_name
                    })
        
        return detections 