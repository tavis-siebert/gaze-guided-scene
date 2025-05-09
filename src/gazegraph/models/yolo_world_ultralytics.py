import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from ultralytics import YOLOWorld

from gazegraph.logger import get_logger

logger = get_logger(__name__)

class YOLOWorldUltralyticsModel:
    """Handles YOLO-World model loading and inference using Ultralytics."""
    
    def __init__(
        self, 
        conf_threshold: float = 0.35, 
        iou_threshold: float = 0.7,
        device: Optional[str] = None
    ):
        """Initialize YOLO-World Ultralytics model."""
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = "cuda" if (device is None and torch.cuda.is_available()) else (device or "cpu")
        self.model = None
        self.names = []
    
    def load_model(self, model_path: Path) -> None:
        """Load the YOLO-World model using Ultralytics."""
        try:
            logger.info(f"Loading YOLO-World model from: {model_path}")
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = YOLOWorld(str(model_path))
            logger.info(f"YOLO-World model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO-World model: {e}")
            raise
    
    def set_classes(self, class_names: List[str]) -> None:
        """Set object classes for the model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        self.names = class_names
        
        # Format class names for prompts
        formatted_names = [f"a photo of a {name.replace('_', ' ')}" for name in class_names]
        self.model.set_classes(formatted_names)
        
        logger.info(f"Set {len(class_names)} class names")
    
    def run_inference(
        self, 
        frame: Any, 
        text_labels: List[str] = None,
        obj_labels: Dict[int, str] = None,
        image_size: int = 640
    ) -> List[Dict[str, Any]]:
        """Run YOLO-World inference on an image frame."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Set classes if provided and not already set
        if text_labels and not self.names:
            class_names = []
            for label in text_labels:
                for prefix in ["a photo of a ", "a picture of a "]:
                    if label.startswith(prefix):
                        label = label[len(prefix):]
                        break
                class_names.append(label)
            self.set_classes(class_names)
        
        # Convert tensor to numpy array if needed
        if isinstance(frame, torch.Tensor):
            image = frame.permute(1, 2, 0).cpu().numpy() if frame.shape[0] == 3 else frame.cpu().numpy()
        else:
            image = frame
        
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
                        "bbox": [x1, y1, x2-x1, y2-y1],
                        "score": float(scores[i]),
                        "class_id": class_id,
                        "class_name": class_name
                    })
        
        return detections 