import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from PIL import Image

from gazegraph.logger import get_logger

logger = get_logger(__name__)

class YOLOWorldModel(ABC):
    """Base class for YOLO-World models with different backends."""
    
    @staticmethod
    def create(
        backend: str = "ultralytics",
        model_path: Path = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.7,
        device: Optional[str] = None
    ) -> 'YOLOWorldModel':
        """Factory method to create the appropriate YOLO-World model.
        
        Args:
            backend: Backend to use ("ultralytics" or "onnx")
            model_path: Path to the model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on
        """
        if backend.lower() == "ultralytics":
            from gazegraph.models.yolo_world_ultralytics import YOLOWorldUltralyticsModel
            return YOLOWorldUltralyticsModel(model_path, conf_threshold, iou_threshold, device)
        elif backend.lower() == "onnx":
            from gazegraph.models.yolo_world_onnx import YOLOWorldOnnxModel
            return YOLOWorldOnnxModel(model_path, conf_threshold, iou_threshold, device)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def __init__(
        self, 
        model_path: Path,
        conf_threshold: float = 0.35, 
        iou_threshold: float = 0.7,
        device: Optional[str] = None
    ):
        """Initialize the YOLO-World model.
        
        Args:
            model_path: Path to the model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._setup_device(device)
        self.names = []
        
        if model_path:
            self._load_model(model_path)
    
    def _setup_device(self, device: Optional[str]) -> None:
        """Set up device for inference."""
        self.device = "cuda" if (device is None and torch.cuda.is_available()) else (device or "cpu")
    
    @abstractmethod
    def _load_model(self, model_path: Path) -> None:
        """Load the model from the given path."""
        pass
    
    def set_classes(self, class_names: List[str]) -> None:
        """Set object classes for the model."""
        self.names = class_names
        self._update_model_classes(class_names)
        logger.info(f"Set {len(class_names)} class names")
    
    @abstractmethod
    def _update_model_classes(self, class_names: List[str]) -> None:
        """Update the model with the new class names."""
        pass
    
    def _preprocess_image(self, image: Union[np.ndarray, torch.Tensor, Image.Image]) -> np.ndarray:
        """Convert image to RGB numpy array in HWC format."""
        # Handle PIL Images
        if isinstance(image, Image.Image):
            return np.array(image)
            
        # Handle torch tensors
        elif isinstance(image, torch.Tensor):
            # Handle batch dimension
            if image.dim() == 4:
                image = image[0]
            
            # Convert CHW -> HWC if needed
            if image.dim() == 3 and image.shape[0] in [1, 3, 4]:
                image = image.permute(1, 2, 0).cpu().numpy()
            else:
                image = image.cpu().numpy()
                
            # Scale to 0-255 if needed
            if image.dtype in [np.float32, np.float64] and image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            
            return image
            
        # Handle numpy arrays
        elif isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed (assuming BGR for OpenCV inputs)
            if image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8:
                import cv2
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        else:
            logger.warning(f"Unsupported image type: {type(image)}")
            return image
    
    @abstractmethod
    def _run_inference(self, image: np.ndarray, image_size: int = 640) -> List[Dict[str, Any]]:
        """Run inference with the model-specific implementation."""
        pass
    
    def predict(
        self, 
        image: Union[np.ndarray, torch.Tensor, Image.Image], 
        class_names: Optional[List[str]] = None,
        image_size: int = 640
    ) -> List[Dict[str, Any]]:
        """Detect objects in the input image.
        
        Args:
            image: Input image (numpy array, torch tensor, or PIL image)
            class_names: Optional list of class names to detect
            image_size: Size to resize the image to for inference
            
        Returns:
            List of detections with bbox, score, class_id, and class_name
        """
        # Set classes if provided
        if class_names and (not self.names or set(class_names) != set(self.names)):
            self.set_classes(class_names)
            
        # Preprocess the image to numpy RGB format
        preprocessed_image = self._preprocess_image(image)
        
        # Run model-specific inference
        return self._run_inference(preprocessed_image, image_size) 