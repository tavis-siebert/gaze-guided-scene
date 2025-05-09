import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from PIL import Image

from gazegraph.logger import get_logger
from gazegraph.config.config_utils import get_config

logger = get_logger(__name__)

class YOLOWorldModel(ABC):
    """Base class for YOLO-World models with different backends."""
    
    @staticmethod
    def create(
        backend: Optional[str] = None,
        model_path: Optional[Path] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        device: Optional[str] = None,
        use_prefix: Optional[bool] = None,
        replace_underscores: Optional[bool] = None
    ) -> 'YOLOWorldModel':
        """Factory method to create the appropriate YOLO-World model."""
        config = get_config().models.yolo_world
        
        # Use provided backend or default from config
        actual_backend = backend or config.backend
        
        # Validate backend before accessing paths
        if actual_backend.lower() not in ["ultralytics", "onnx"]:
            raise ValueError(f"Unsupported backend: {actual_backend}")
        
        # Get model path
        if model_path is None:
            model_path = Path(config.paths[actual_backend])
        
        # Create appropriate model instance
        if actual_backend.lower() == "ultralytics":
            from gazegraph.models.yolo_world_ultralytics import YOLOWorldUltralyticsModel
            return YOLOWorldUltralyticsModel(model_path, conf_threshold, iou_threshold, device, use_prefix, replace_underscores)
        else:  # must be "onnx" based on validation above
            from gazegraph.models.yolo_world_onnx import YOLOWorldOnnxModel
            return YOLOWorldOnnxModel(model_path, conf_threshold, iou_threshold, device, use_prefix, replace_underscores)
    
    def __init__(
        self, 
        model_path: Optional[Path] = None,
        conf_threshold: Optional[float] = None, 
        iou_threshold: Optional[float] = None,
        device: Optional[str] = None,
        use_prefix: Optional[bool] = None,
        replace_underscores: Optional[bool] = None
    ):
        """Initialize the YOLO-World model."""
        config = get_config().models.yolo_world
        
        # Get backend-specific config
        backend_name = self._get_backend_name()
        backend_config = getattr(config, backend_name)
        
        # Set thresholds (use provided values or defaults from config)
        self.conf_threshold = conf_threshold if conf_threshold is not None else backend_config.conf_threshold
        self.iou_threshold = iou_threshold if iou_threshold is not None else backend_config.iou_threshold
        
        # Set text prompt formatting options
        self.use_prefix = False if use_prefix is None else use_prefix  # Default: no prefix
        self.replace_underscores = True if replace_underscores is None else replace_underscores  # Default: replace underscores
        
        # Set up device
        self.device = "cuda" if (device is None and torch.cuda.is_available()) else (device or "cpu")
        
        # Initialize state
        self.names = []
        
        # Load model if path provided
        if model_path:
            self._load_model(model_path)
    
    def _get_backend_name(self) -> str:
        """Get the backend name for this model implementation."""
        if self.__class__.__name__ == 'YOLOWorldUltralyticsModel':
            return 'ultralytics'
        elif self.__class__.__name__ == 'YOLOWorldOnnxModel':
            return 'onnx'
        else:
            raise ValueError(f"Unknown model class: {self.__class__.__name__}")
    
    @abstractmethod
    def _load_model(self, model_path: Path) -> None:
        """Load the model from the given path."""
        pass
    
    def format_class_name(self, class_name: str) -> str:
        """Format a class name according to the current settings."""
        formatted_name = class_name
        
        # Replace underscores with spaces if enabled
        if self.replace_underscores:
            formatted_name = formatted_name.replace('_', ' ')
            
        # Add prefix if enabled
        if self.use_prefix:
            formatted_name = f"a photo of a {formatted_name}"
            
        return formatted_name
    
    def format_class_names(self, class_names: List[str]) -> List[str]:
        """Format a list of class names according to current settings."""
        return [self.format_class_name(name) for name in class_names]
    
    def set_classes(self, class_names: List[str]) -> None:
        """Set object classes for the model."""
        self.names = class_names
        formatted_class_names = self.format_class_names(class_names)
        self._update_model_classes(formatted_class_names)
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
    def _run_inference(self, image: np.ndarray, image_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Run inference with the model-specific implementation."""
        pass
    
    def predict(
        self, 
        image: Union[np.ndarray, torch.Tensor, Image.Image], 
        class_names: Optional[List[str]] = None,
        image_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Detect objects in the input image."""
        # Set classes if provided
        if class_names and (not self.names or set(class_names) != set(self.names)):
            self.set_classes(class_names)
            
        # Get default image size from config if not provided
        if image_size is None:
            image_size = get_config().models.yolo_world.image_size
            
        # Preprocess the image to numpy RGB format
        preprocessed_image = self._preprocess_image(image)
        
        # Run model-specific inference
        return self._run_inference(preprocessed_image, image_size) 