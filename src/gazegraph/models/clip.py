import torch
import clip
from typing import List, Tuple
from PIL import Image

from gazegraph.logger import get_logger

logger = get_logger(__name__)


class ClipModel:
    """Wrapper for CLIP model providing text and image encoding capabilities."""

    def __init__(
        self,
        name: str = "ViT-L/14",
        jit: bool = False,
        download_root: str | None = None,
        device: str | None = None,
    ):
        """Initialize CLIP model.

        Args:
            name: Model name or checkpoint path
            jit: Whether to load JIT optimized model
            download_root: Path to download model files
            device: Device to run model on
        """
        self.name = name
        self.device = (
            device if device else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model, self.preprocess = clip.load(
            name=name,
            device=self.device,
            jit=jit,
            download_root=download_root,  # type: ignore
        )

    def encode_texts(self, texts: List[str]) -> List[torch.Tensor]:
        """Encode text inputs using CLIP.

        Args:
            texts: List of text strings to encode

        Returns:
            List of text embeddings as tensors
        """
        text_tokens = clip.tokenize(texts).to(self.device)

        with torch.no_grad():
            text_features = [
                self.model.encode_text(token).detach() for token in text_tokens.split(1)
            ]

        return text_features

    def encode_image(self, image: Image.Image | torch.Tensor) -> torch.Tensor:
        """Encode image input using CLIP.

        Args:
            image: PIL Image or preprocessed tensor

        Returns:
            Image embedding tensor
        """
        if isinstance(image, Image.Image):
            image = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)

        return image_features

    def classify(
        self, labels: List[str], image: Image.Image | torch.Tensor
    ) -> Tuple[List[float], str]:
        """Classify an image by computing similarity scores against provided labels.

        Args:
            labels: Candidate class labels
            image: PIL Image or preprocessed tensor

        Returns:
            A tuple of (scores, best_label) where scores is a list of similarity scores
            corresponding to labels and best_label is the label with the highest score.
        """
        image_features = self.encode_image(image)
        text_features_list = self.encode_texts(labels)
        text_features = torch.cat(text_features_list, dim=0)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        scores = logits_per_image[0].tolist()
        best_index = int(logits_per_image.argmax(dim=1)[0])
        best_label = labels[best_index]
        return scores, best_label
