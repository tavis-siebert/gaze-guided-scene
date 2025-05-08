import torch
import clip
from pathlib import Path
from typing import Dict, List, Optional
from transformers import CLIPProcessor, CLIPModel as HFCLIPModel

from gazegraph.logger import get_logger

logger = get_logger(__name__)

class ClipTextEmbeddingModel:
    def __init__(self, device: str = "cuda"):
        """Initialize CLIP text embedder."""
        self.device = "cuda" if device == "0" else device
        self.model, _ = clip.load("ViT-B/32", device=self.device)

    def __call__(self, text: List[str]) -> List[torch.Tensor]:
        """Embed text using CLIP model."""
        text_token = clip.tokenize(text).to(self.device)
        txt_feats = [self.model.encode_text(token).detach() for token in text_token.split(1)]
        return txt_feats

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