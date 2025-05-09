import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from ultralytics import YOLOWorld
from PIL import Image
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
    
    def _ensure_numpy_hwc_format(self, frame: Union[np.ndarray, torch.Tensor, Image.Image]) -> np.ndarray:
        """
        Ensures the image is in numpy HWC format with proper channel ordering for YOLOWorld.
        YOLOWorld expects:
        - numpy arrays in HWC format with RGB channels (uint8, 0-255)
        - OR PIL images
        - OR torch tensors in BCHW format with RGB channels (float32, 0.0-1.0)
        
        Parameters:
            frame: Input image as PIL.Image, np.ndarray, or torch.Tensor
            
        Returns:
            np.ndarray: Image in RGB HWC format (uint8, 0-255)
        """
        # Handle PIL Images
        if isinstance(frame, Image.Image):
            # PIL is already in RGB format, just convert to numpy
            return np.array(frame)
            
        # Handle torch tensors
        elif isinstance(frame, torch.Tensor):
            # Check for batch dimension (BCHW format)
            if frame.dim() == 4:
                # If it's a batch, take the first image
                logger.warning("Received batch of images, using only the first one")
                frame = frame[0]
            
            # If tensor is in channel-first format (CHW), convert to (HWC)
            if frame.dim() == 3 and frame.shape[0] in [1, 3, 4]:
                # Convert CHW -> HWC
                image = frame.permute(1, 2, 0).cpu().numpy()
            else:
                image = frame.cpu().numpy()
                
            # Handle value range: if float and max value <= 1.0, convert to uint8
            if image.dtype in [np.float32, np.float64] and image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            
            # Keep as RGB (YOLOWorld uses RGB)
            return image
            
        # Handle numpy arrays (typically from OpenCV - BGR format)
        elif isinstance(frame, np.ndarray):
            # Ensure HWC format (if not already)
            if frame.ndim == 3 and frame.shape[2] == 3:
                # OpenCV images are BGR, convert to RGB
                import cv2
                # Check if this might be a BGR image from OpenCV
                if frame.dtype == np.uint8:
                    # Convert BGR to RGB (cv2 images are typically BGR)
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame
            else:
                logger.warning(f"Unexpected numpy array shape: {frame.shape}")
                return frame
        else:
            logger.warning(f"Unsupported image type: {type(frame)}")
            return frame
    
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
        
        # Ensure proper image format
        image = self._ensure_numpy_hwc_format(frame)

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