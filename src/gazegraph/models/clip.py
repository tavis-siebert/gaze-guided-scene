import torch
import clip
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Tuple
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

    def load(self, name: str = "ViT-L/14", jit: bool = False, download_root: str = None) -> None:
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

    def classify(self, labels: List[str], image: Union[Image.Image, torch.Tensor]) -> Tuple[List[float], str]:
        """Classify an image by computing similarity scores against provided labels.

        Args:
            labels: Candidate class labels
            image: PIL Image or preprocessed tensor

        Returns:
            A tuple of (scores, best_label) where scores is a list of similarity scores 
            corresponding to labels and best_label is the label with the highest score.
        """
        if self.model is None:
            self.load()

        if isinstance(image, Image.Image):
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        else:
            image_tensor = image.to(self.device)

        text_tokens = clip.tokenize(labels).to(self.device)

        with torch.no_grad():
            logits_per_image, _ = self.model(image_tensor, text_tokens)

        scores = logits_per_image[0].tolist()
        best_index = int(logits_per_image.argmax(dim=1)[0])
        best_label = labels[best_index]
        return scores, best_label