import torch
import numpy as np
from PIL import Image
import cv2
import onnxruntime as ort
from pathlib import Path
from typing import Dict, List, Any, Optional

from gazegraph.logger import get_logger
from gazegraph.models.onnx_utils import make_session_options
from gazegraph.models.clip import ClipModel
from gazegraph.models.yolo_world_model import YOLOWorldModel

logger = get_logger(__name__)

class YOLOWorldOnnxModel(YOLOWorldModel):
    """YOLO-World model using ONNX Runtime backend."""
    
    def __init__(
        self, 
        model_path: Optional[Path] = None,
        conf_threshold: Optional[float] = None, 
        iou_threshold: Optional[float] = None,
        device: Optional[str] = None,
        use_prefix: Optional[bool] = None,
        replace_underscores: Optional[bool] = None
    ):
        """Initialize YOLO-World ONNX model."""
        # Initialize attributes that will be set in _load_model
        self.session = None
        self.clip_model = None
        self.class_embeddings = None
        self.input_names = None
        self.output_names = None
        self.num_classes = None
        
        # Configure ONNX-specific device string - ONNX uses "0" for first GPU
        self.device = "0" if (device is None and torch.cuda.is_available()) else (device or "cpu")
        self.text_embedding_device = "cuda" if self.device == "0" else "cpu"
        
        # Call parent constructor
        super().__init__(model_path, conf_threshold, iou_threshold, self.device, use_prefix, replace_underscores)
    
    def _load_model(self, model_path: Path, num_workers: Optional[int] = None) -> None:
        """Load the YOLO-World ONNX model."""
        try:
            logger.info(f"Loading YOLO-World ONNX model from: {model_path}")
            
            if not model_path.exists():
                logger.error(f"Model file does not exist at {model_path}")
                available_models = list(model_path.parent.glob("*.onnx"))
                if available_models:
                    logger.error(f"Available ONNX models: {[m.name for m in available_models]}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Configure session options
            sess_options = make_session_options(num_workers)
            
            # Select appropriate providers
            providers = ["CUDAExecutionProvider"] if self.device == "0" else ["CPUExecutionProvider"]
            
            # Create inference session
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=providers
            )
            
            # Initialize CLIP model for text embeddings
            self.clip_model = ClipModel(name="ViT-B/32", device=self.text_embedding_device)
            
            # Get model details
            model_inputs = self.session.get_inputs()
            self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
            self.num_classes = model_inputs[1].shape[1]
            
            model_outputs = self.session.get_outputs()
            self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
            
            logger.info(f"YOLO-World ONNX model loaded successfully on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO-World ONNX model: {e}")
            raise
    
    def _update_model_classes(self, class_names: List[str]) -> None:
        """Update the model with the new class names."""
        if self.session is None:
            raise RuntimeError("Model not loaded")
        
        # Convert class names to embeddings with CLIP
        txt_feats = self.clip_model.encode_texts(class_names)
        
        # Process the embeddings
        txt_feats = torch.cat(txt_feats, dim=0)
        txt_feats /= txt_feats.norm(dim=1, keepdim=True)
        self.class_embeddings = txt_feats.unsqueeze(0)
    
    def _run_inference(self, image: Image.Image, image_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Run inference with the ONNX model."""
        if self.session is None or self.class_embeddings is None:
            raise RuntimeError("Model not loaded or classes not set")
        
        if not isinstance(image, Image.Image):
            raise TypeError(f"YOLOWorldOnnxModel.predict requires PIL.Image.Image, got {type(image)}")
        image_np = np.array(image)
        # Get original image dimensions
        h, w = image_np.shape[:2]
        # Convert to BGR for OpenCV processing
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Prepare embeddings (pad if needed)
        embeddings = self.class_embeddings
        if embeddings.shape[1] != self.num_classes:
            embeddings = torch.nn.functional.pad(
                embeddings, 
                (0, 0, 0, self.num_classes - embeddings.shape[1]), 
                mode='constant'
            )
        
        # Preprocess image
        input_img = cv2.resize(image_bgr, (image_size, image_size))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB) / 255.0  # To RGB and normalize
        input_img = input_img.transpose(2, 0, 1)  # HWC to CHW
        input_img = np.expand_dims(input_img, 0).astype(np.float32)  # Add batch dimension
        
        # Run inference
        outputs = self.session.run(
            self.output_names,
            {
                self.input_names[0]: input_img,
                self.input_names[1]: embeddings.cpu().numpy().astype(np.float32)
            }
        )
        
        # Process outputs
        # Normalize output shape to (num_preds, dims)
        dims = 4 + self.num_classes
        pred_arr = np.squeeze(outputs[0], axis=0)
        
        # Flatten extra dimensions if necessary
        if pred_arr.ndim > 2:
            pred_arr = pred_arr.reshape(-1, pred_arr.shape[-1])
            
        # Handle single detection as 1-row array
        if pred_arr.ndim == 1:
            pred_arr = pred_arr[np.newaxis, :]
            
        # Abort if not 2D
        if pred_arr.ndim != 2:
            return []
            
        # Transpose if dims axis is first
        if pred_arr.shape[1] != dims and pred_arr.shape[0] == dims:
            pred_arr = pred_arr.T
            
        # Validate expected dims
        if pred_arr.shape[1] != dims:
            raise ValueError(f"Unexpected output shape: {pred_arr.shape}")
            
        predictions = pred_arr
        
        # Filter by confidence
        scores = np.max(predictions[:, 4:], axis=1)
        mask = scores >= self.conf_threshold
        if not np.any(mask):
            return []
            
        predictions = predictions[mask]
        scores = scores[mask]
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        
        # Get boxes (convert from normalized to pixel coordinates)
        boxes = predictions[:, :4]
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / image_size * w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / image_size * h
        
        # Apply NMS
        raw_indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.conf_threshold,
            self.iou_threshold
        )
        
        # Handle empty results
        if raw_indices is None or (hasattr(raw_indices, '__len__') and len(raw_indices) == 0):
            return []
            
        # Flatten indices to list of ints
        if isinstance(raw_indices, (int, np.integer)):
            flat_indices = [int(raw_indices)]
        else:
            flat_indices = np.array(raw_indices).reshape(-1).tolist()
        
        # Build detection results
        detections = []
        for idx in flat_indices:
            idx = int(idx)
            x, y, width, height = boxes[idx]
            detections.append({
                "bbox": [x - width/2, y - height/2, width, height],  # [x, y, width, height]
                "score": float(scores[idx]),
                "class_id": int(class_ids[idx]),
                "class_name": self.names[class_ids[idx]]
            })
            
        return detections 