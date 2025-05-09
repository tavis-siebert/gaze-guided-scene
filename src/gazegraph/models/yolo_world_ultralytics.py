import torch
import numpy as np
import cv2
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
    
    def load_model(self, model_path: Path, num_workers: Optional[int] = None) -> None:
        """Load the YOLO-World model using Ultralytics."""
        try:
            logger.info(f"Loading YOLO-World model from: {model_path}")
            
            if not model_path.exists():
                logger.error(f"Model file does not exist at {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = YOLOWorld(str(model_path))
            
            logger.info(f"YOLO-World model loaded successfully on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO-World model: {e}")
            raise
    
    def set_classes(self, class_names: List[str]) -> None:
        """Set object classes for the model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        self.names = class_names

        def format_class_name(name: str) -> str:
            """Format a class name by replacing underscores with spaces."""
            return name.replace('_', ' ')
        
        formatted_class_names = [f"a photo of a {format_class_name(name)}" for name in class_names]
        self.model.set_classes(formatted_class_names)
        
        logger.info(f"Set {len(class_names)} class names for YOLO-World Ultralytics")
    
    def run_inference(
        self, 
        frame: Any, 
        text_labels: List[str],
        obj_labels: Dict[int, str],
        image_size: int = 640
    ) -> List[Dict[str, Any]]:
        """Run YOLO-World inference on an image frame."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Set classes if not already set
        if not self.names:
            # Extract class names from text labels
            class_names = []
            for label in text_labels:
                # Remove prefix like "a photo of a " or "a picture of a "
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
        
        # Run inference with Ultralytics
        results = self.model.predict(source=image,
            imgsz=image_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device
        )
        
        # Process results to match expected format
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            
            # Extract boxes, confidence scores, and class IDs
            if len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                # Convert to expected format
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    width = x2 - x1
                    height = y2 - y1
                    
                    class_id = int(class_ids[i])
                    class_name = self.names[class_id] if class_id < len(self.names) else f"unknown_{class_id}"
                    
                    detections.append({
                        "bbox": [x1, y1, width, height],
                        "score": float(scores[i]),
                        "class_id": class_id,
                        "class_name": class_name
                    })
        
        return detections 