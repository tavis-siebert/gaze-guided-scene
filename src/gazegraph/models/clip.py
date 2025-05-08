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