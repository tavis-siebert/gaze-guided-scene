import torch
import clip
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable
from transformers import CLIPProcessor, CLIPModel as HFCLIPModel
from PIL import Image

from gazegraph.logger import get_logger

logger = get_logger(__name__)

class ClipModel:
    """Wrapper for CLIP model providing text and image encoding capabilities."""
    
    def __init__(self, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize CLIP model.
        
        Args:
            device: Device to run model on
        """
        self.device = device
        self.model = None
        self.preprocess = None
        
    def load(self, name: str = "ViT-B/32", jit: bool = False, download_root: str = None) -> None:
        """Load CLIP model and preprocessor.
        
        Args:
            name: Model name or checkpoint path
            jit: Whether to load JIT optimized model
            download_root: Path to download model files
        """
        self.model, self.preprocess = clip.load(
            name=name,
            device=self.device,
            jit=jit,
            download_root=download_root
        )
        
    def encode_text(self, texts: List[str]) -> List[torch.Tensor]:
        """Encode text inputs using CLIP.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            List of text embeddings as tensors
        """
        if self.model is None:
            self.load()
            
        text_tokens = clip.tokenize(texts).to(self.device)

        with torch.no_grad():
            text_features = [self.model.encode_text(token).detach() for token in text_tokens.split(1)]
            
        return text_features
        
    def encode_image(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """Encode image input using CLIP.
        
        Args:
            image: PIL Image or preprocessed tensor
            
        Returns:
            Image embedding tensor
        """
        if self.model is None:
            self.load()
            
        if isinstance(image, Image.Image):
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            
        return image_features

class ClipImageClassificationModel:
    """
    Handles CLIP model loading, processing, and inference for object detection.
    """
    
    def __init__(self, model_id: str = "openai/clip-vit-base-patch16"):
        """
        Initialize the CLIP model.
        
        Args:
            model_id: The HuggingFace model ID for CLIP
        """
        self.model_id = model_id
        self.processor = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self, local_model_dir: Optional[Path] = None) -> None:
        """
        Load the CLIP model, trying local directory first then falling back to online.
        
        Args:
            local_model_dir: Path to local model directory, if available
        """
        try:
            if local_model_dir:
                logger.info(f"Loading CLIP model from local directory: {local_model_dir}")
                self.processor = CLIPProcessor.from_pretrained(str(local_model_dir))
                self.model = HFCLIPModel.from_pretrained(str(local_model_dir))
            else:
                raise FileNotFoundError("No local model directory provided")
        except Exception as e:
            logger.warning(f"Failed to load local model, downloading from {self.model_id}: {e}")
            self.processor = CLIPProcessor.from_pretrained(self.model_id)
            self.model = HFCLIPModel.from_pretrained(self.model_id)
            
        self.model = self.model.to(self.device)
        logger.info(f"Using device: {self.device} for CLIP model")
    
    def run_inference(self, frame: torch.Tensor, text_labels: List[str], obj_labels: Dict[int, str]) -> str:
        """
        Run CLIP inference on an image frame.
        
        Args:
            frame: The image frame tensor
            text_labels: List of text prompts for CLIP
            obj_labels: Dictionary mapping class indices to object labels
            
        Returns:
            The predicted object label
        """
        if self.processor is None or self.model is None:
            raise RuntimeError("CLIP model not loaded. Call load_model() first.")
            
        inputs = self.processor(
            text=text_labels,
            images=frame,
            return_tensors='pt',
            padding=True
        )
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        probs = outputs.logits_per_image.softmax(dim=1)
        predicted_class_idx = probs.argmax(dim=1).item()
        
        return obj_labels[predicted_class_idx]