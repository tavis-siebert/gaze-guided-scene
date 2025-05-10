import torch
import numpy as np
import cv2
import json
import clip
import onnxruntime as ort
from pathlib import Path
from typing import Dict, List, Any, Optional

from gazegraph.logger import get_logger
from gazegraph.models.onnx_utils import make_session_options

logger = get_logger(__name__)

class TextEmbedder:
    def __init__(self, device: str = "cuda"):
        """Initialize CLIP text embedder."""
        self.device = "cuda" if device == "0" else device
        self.model, _ = clip.load("ViT-B/32", device=self.device)

    def __call__(self, text: List[str]) -> torch.Tensor:
        """Embed text using CLIP model."""
        text_token = clip.tokenize(text).to(self.device)
        txt_feats = [self.model.encode_text(token).detach() for token in text_token.split(1)]
        txt_feats = torch.cat(txt_feats, dim=0)
        txt_feats /= txt_feats.norm(dim=1, keepdim=True)
        return txt_feats.unsqueeze(0)

class YOLOWorldModel:
    """Handles YOLO-World model loading and inference for object detection."""
    
    def __init__(
        self, 
        conf_threshold: float = 0.35, 
        iou_threshold: float = 0.7,
        device: Optional[str] = None
    ):
        """Initialize YOLO-World model."""
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = "0" if (device is None and torch.cuda.is_available()) else (device or "cpu")
        self.text_embedding_device = self.device
        
        # Will be initialized when loading model
        self.session = None
        self.text_embedder = None
        self.class_embeddings = None
        self.names = []
        self.input_names = None
        self.output_names = None
        self.num_classes = None
    
    def load_model(self, model_path: Path, num_workers: Optional[int] = None) -> None:
        """Load the YOLO-World ONNX model.
        
        Args:
            model_path: Path to the ONNX model file
            num_workers: Number of parallel workers that will use ONNX Runtime.
                         Used to properly allocate CPU threads.
        """
        try:
            logger.info(f"Loading YOLO-World model from: {model_path}")
            
            if not model_path.exists():
                logger.error(f"Model file does not exist at {model_path}")
                available_models = list(model_path.parent.glob("*.onnx"))
                if available_models:
                    logger.error(f"Available ONNX models: {[m.name for m in available_models]}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Configure session options to avoid thread affinity issues
            sess_options = make_session_options(num_workers)
            
            providers = ["CUDAExecutionProvider"] if self.device == "0" else ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=providers
            )
            self.text_embedder = TextEmbedder(device=self.text_embedding_device)
            
            # Get model details
            model_inputs = self.session.get_inputs()
            self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
            self.num_classes = model_inputs[1].shape[1]
            
            model_outputs = self.session.get_outputs()
            self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
            
            logger.info(f"YOLO-World model loaded successfully on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO-World model: {e}")
            raise
    
    def set_classes(self, classes: List[str]) -> None:
        """Set object classes for the model."""
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
                
        def clean_label(class_name: str) -> str:
          no_prefix = class_name.replace("a picture of a ", "").replace("a photo of a ", "") # Yolo World seems to perform better without prefix
          no_underscores = no_prefix.replace("_", " ")
          return no_underscores.strip()

        self.classes = classes
        processed_classes = [clean_label(class_name) for class_name in classes]

        self.class_embeddings = self.text_embedder(processed_classes)
        logger.info(f"Set {len(self.classes)} class names for YOLO-World")
    
    def run_inference(
        self, 
        frame: torch.Tensor,
        image_size: int = 640
    ) -> List[Dict[str, Any]]:
        """Run YOLO-World inference on an image frame."""
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self.classes is None:
            raise RuntimeError("Classes not set. Call set_classes() first.")

        # Prepare input image
        if isinstance(frame, torch.Tensor):
            image = frame.permute(1, 2, 0).cpu().numpy() if frame.shape[0] == 3 else frame.cpu().numpy()
        else:
            image = frame
            
        if image.shape[2] == 3:  # RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        # Prepare embeddings
        embeddings = self.class_embeddings
        if embeddings.shape[1] != self.num_classes:
            embeddings = torch.nn.functional.pad(
                embeddings, 
                (0, 0, 0, self.num_classes - embeddings.shape[1]), 
                mode='constant'
            )
        
        # Preprocess image
        h, w = image.shape[:2]
        input_img = cv2.resize(image, (image_size, image_size))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB) / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_img = np.expand_dims(input_img, 0).astype(np.float32)

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
        
        # Get boxes
        boxes = predictions[:, :4]
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / image_size * w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / image_size * h
        
        # Apply NMS and handle various index return types
        raw_indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.conf_threshold,
            self.iou_threshold
        )
        # Return empty list if no detections
        if raw_indices is None or (hasattr(raw_indices, '__len__') and len(raw_indices) == 0):
            return []
        # Flatten to list of ints
        if isinstance(raw_indices, (int, np.integer)):
            flat_indices = [int(raw_indices)]
        else:
            flat_indices = np.array(raw_indices).reshape(-1).tolist()
        # Build detections
        detections = []
        for idx in flat_indices:
            idx = int(idx)
            x, y, width, height = boxes[idx]
            detections.append({
                "bbox": [x - width/2, y - height/2, width, height],
                "score": float(scores[idx]),
                "class_id": int(class_ids[idx]),
                "class_name": self.classes[int(class_ids[idx])]
            })
        return detections 