import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from ultralytics import YOLOWorld

from gazegraph.logger import get_logger
from gazegraph.models.yolo_world_model import YOLOWorldModel

logger = get_logger(__name__)

class YOLOWorldUltralyticsModel(YOLOWorldModel):
    """YOLO-World model using Ultralytics backend."""
    
    def __init__(
        self, 
        model_path: Path = None,
        conf_threshold: float = 0.35, 
        iou_threshold: float = 0.7,
        device: Optional[str] = None
    ):
        """Initialize YOLO-World Ultralytics model.
        
        Args:
            model_path: Path to the model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on
        """
        self.model = None
        super().__init__(model_path, conf_threshold, iou_threshold, device)
    
    def _load_model(self, model_path: Path) -> None:
        """Load the YOLO-World model using Ultralytics.
        
        Args:
            model_path: Path to the model file
        """
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
        """Update the model with the new class names.
        
        Args:
            class_names: List of class names
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Format class names for prompts
        formatted_names = [f"a photo of a {name.replace('_', ' ')}" for name in class_names]
        self.model.set_classes(formatted_names)
    
    def _run_inference(self, image: np.ndarray, image_size: int = 640) -> List[Dict[str, Any]]:
        """Run inference with the Ultralytics model.
        
        Args:
            image: Input image as RGB numpy array
            image_size: Size to resize the image to for inference
            
        Returns:
            List of detections with bbox, score, class_id, and class_name
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

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