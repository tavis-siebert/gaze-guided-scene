import cv2
import torch
import numpy as np
from typing import Tuple, List


class SIFT:
    """
    Handles feature extraction using SIFT (Scale-Invariant Feature Transform).
    """

    def __init__(self):
        """Initialize the SIFT feature detector."""
        self.detector = cv2.SIFT_create()

    def extract_features(
        self, frame: torch.Tensor
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Extract SIFT features from a frame.

        Args:
            frame: The input frame tensor (C, H, W)

        Returns:
            Tuple of (keypoints, descriptors)
        """
        # Convert to grayscale for SIFT
        gray_frame = cv2.cvtColor(frame.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY)

        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.detector.detectAndCompute(gray_frame, None)

        return keypoints, descriptors
