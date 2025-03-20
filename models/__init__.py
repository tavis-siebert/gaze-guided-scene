"""
Models module for feature extraction and inference.

This module provides classes for feature extraction (SIFT) and 
object detection (CLIP) used in the scene graph construction.
"""

from models.sift import SIFT
from models.clip import ClipModel

__all__ = [
    'SIFT',
    'ClipModel'
] 