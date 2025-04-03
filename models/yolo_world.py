import torch
import numpy as np
import cv2
from yolo_world_onnx import YOLOWORLD
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

class YOLOWorldModel:
    """
    Handles YOLO-World model loading and inference for object detection.
    """
    
    def __init__(
        self, 
        conf_threshold: float = 0.35, 
        iou_threshold: float = 0.7,
        device: Optional[str] = None
    ):
        """
        Initialize the YOLO-World model.
        
        Args:
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for non-maximum suppression
            device: Device to run inference on ('cpu', '0' for first GPU, etc.)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        
        # Determine device
        if device is None:
            device = "0" if torch.cuda.is_available() else "cpu"
        self.device = device
    
    def load_model(self, model_path: Path) -> None:
        """
        Load the YOLO-World ONNX model.
        
        Args:
            model_path: Path to the ONNX model file
        """
        try:
            logger.info(f"Loading YOLO-World model from: {model_path}")
            
            if not model_path.exists():
                logger.error(f"Model file does not exist at {model_path}")
                available_models = list(model_path.parent.glob("*.onnx"))
                if available_models:
                    logger.error(f"Available ONNX models in directory: {[m.name for m in available_models]}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Log model file size
            model_size_mb = model_path.stat().st_size / (1024 * 1024)
            logger.info(f"Model file size: {model_size_mb:.2f} MB")
            
            self.model = YOLOWORLD(str(model_path), device=self.device)
            logger.info(f"YOLO-World model '{model_path.name}' loaded successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO-World model: {e}")
            raise
    
    def set_classes(self, class_names: List[str]) -> None:
        """
        Set the object classes for the model.
        
        Args:
            class_names: List of class names
        """
        if self.model is None:
            raise RuntimeError("YOLO-World model not loaded. Call load_model() first.")
        
        self.model.set_classes(class_names)
        logger.info(f"Set {len(class_names)} class names for YOLO-World")
    
    def run_inference(
        self, 
        frame: torch.Tensor, 
        text_labels: List[str],
        obj_labels: Dict[int, str],
        image_size: int = 640
    ) -> List[Dict[str, Any]]:
        """
        Run YOLO-World inference on an image frame.
        
        Args:
            frame: The image frame tensor (C, H, W)
            text_labels: List of text prompts for YOLO-World
            obj_labels: Dictionary mapping class indices to object labels
            image_size: Input image size for the model
            
        Returns:
            List of dictionaries containing detection results
        """
        if self.model is None:
            raise RuntimeError("YOLO-World model not loaded. Call load_model() first.")
        
        # Extract class names from text labels if we haven't set them yet
        if not hasattr(self.model, 'names') or not self.model.names:
            class_names = [label.replace("a picture of a ", "") for label in text_labels]
            self.set_classes(class_names)
        
        # Convert torch tensor to numpy array (H, W, C)
        if isinstance(frame, torch.Tensor):
            if frame.dim() == 3 and frame.shape[0] == 3:  # If in format (C, H, W)
                image = frame.permute(1, 2, 0).cpu().numpy()
            else:
                image = frame.cpu().numpy()
        else:
            image = frame
        
        # Convert to BGR for OpenCV if it's RGB
        if image.shape[2] == 3:  # RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Run inference
        boxes, scores, class_ids = self.model(
            image, 
            conf=self.conf_threshold,
            imgsz=image_size,
            iou=self.iou_threshold
        )
        
        # Create list of detection results
        detections = []
        
        for box, score, class_id in zip(boxes, scores, class_ids):
            # Get box coordinates (x_center, y_center, width, height)
            x, y, w, h = box
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            
            # Get class name
            class_name = self.model.names[class_id]
            
            # Add detection to results
            detections.append({
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # [x, y, width, height]
                "score": float(score),
                "class_id": int(class_id),
                "class_name": class_name
            })
        
        return detections 