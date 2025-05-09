import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from ultralytics import YOLOWorld
import cv2

from gazegraph.logger import get_logger
from gazegraph.models.yolo_world_model import YOLOWorldModel

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
        replace_underscores: Optional[bool] = None
    ):
        """Initialize YOLO-World Ultralytics model."""
        # Initialize model to None before parent constructor
        self.model = None
        
        # Call parent constructor which handles all config
        super().__init__(model_path, conf_threshold, iou_threshold, device, use_prefix, replace_underscores)
    
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
        """Update the model with the new class names."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Class names are already formatted by the parent class
        self.model.set_classes(class_names)
    
    def _run_inference(self, image: np.ndarray, image_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Run inference with the Ultralytics model."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Ensure correct format for Ultralytics: RGB in np.ndarray (HWC format)
        if isinstance(image, torch.Tensor):
            # Convert from BCHW (0.0-1.0) to HWC (0-255)
            image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        # For OpenCV format (BGR), convert to RGB
        if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
            # Check if image is likely in BGR format (from OpenCV)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_path = Path("data/tests/out/input_image.jpg")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

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
            result_path = Path("data/tests/out/result.jpg")
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(str(result_path))
            
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