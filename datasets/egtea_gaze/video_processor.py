"""
Video processing module for EGTEA Gaze+ videos.

This module provides classes for loading and processing video data.
"""

import torch
import torchvision as tv
from typing import Dict, List, Tuple, Optional, Iterator, Any, Set
from pathlib import Path

from datasets.egtea_gaze.gaze_data.gaze_io_sample import parse_gtea_gaze
from datasets.egtea_gaze.video_metadata import VideoMetadata
from config.config_utils import get_config
from logger import get_logger

logger = get_logger(__name__)

class VideoProcessor:
    """
    Handles video loading and frame extraction for EGTEA Gaze+ videos.
    """
    
    def __init__(self, video_path: str, config=None):
        """
        Initialize the video processor.
        
        Args:
            video_path: Path to the video file
            config: Configuration object (optional)
        """
        self.video_path = video_path
        self.config = config or get_config()
        self.stream = tv.io.VideoReader(str(video_path), 'video')
    
    def __iter__(self) -> Iterator:
        """Make the processor iterable to get frames."""
        return self
    
    def __next__(self) -> Tuple[torch.Tensor, int, bool]:
        """
        Get the next frame from the video.
        
        Returns:
            Tuple of (frame, frame_number, is_black_frame)
        """
        try:
            frame = next(self.stream)
            is_black = frame['data'].count_nonzero().item() == 0
            return frame['data'], frame['pts'], is_black
        except StopIteration:
            raise StopIteration
    
    @classmethod
    def from_video_name(cls, video_name: str, metadata: Optional[VideoMetadata] = None) -> 'VideoProcessor':
        """
        Create a VideoProcessor from a video name.
        
        Args:
            video_name: Name of the video
            metadata: VideoMetadata object for accessing paths
            
        Returns:
            VideoProcessor instance
        """
        if metadata is None:
            metadata = VideoMetadata()
            
        video_path = metadata.get_video_path(video_name)
        return cls(video_path)


class VideoDataManager:
    """
    Manages video data access including gaze data and metadata.
    """
    
    def __init__(self, video_name: str, config=None):
        """
        Initialize the video data manager.
        
        Args:
            video_name: Name of the video
            config: Configuration object (optional)
        """
        self.video_name = video_name
        self.config = config or get_config()
        self.metadata = VideoMetadata(self.config)
        
        # Load gaze data
        gaze_path = self.metadata.get_gaze_data_path(video_name)
        self.raw_gaze_data = parse_gtea_gaze(gaze_path)
        
        # Get video length and action records
        self.video_length = self.metadata.get_video_length(video_name)
        self.action_records = self.metadata.get_records_for_video(video_name)
        
        # Get frame range from action annotations
        self.first_frame, self.last_frame = self.metadata.get_action_frame_range(video_name)
        
        logger.info(f"Initialized VideoDataManager for {video_name}")
        logger.info(f"Video length: {self.video_length} frames")
        logger.info(f"Action frame range: {self.first_frame} - {self.last_frame}")
        logger.info(f"Loaded {len(self.action_records)} action records")
        logger.info(f"Loaded {len(self.raw_gaze_data)} gaze points")
    
    def create_video_processor(self) -> VideoProcessor:
        """
        Create a VideoProcessor for this video.
        
        Returns:
            VideoProcessor instance
        """
        return VideoProcessor.from_video_name(self.video_name, self.metadata)
    
    def get_future_actions(self, frame_number: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get future action labels for a specific frame.
        
        Args:
            frame_number: Frame number to get future actions for
            
        Returns:
            Dictionary of action labels or None if insufficient data
        """
        return VideoMetadata.get_future_action_labels(self.video_name, frame_number)
    
    @property
    def labels_to_int(self) -> Dict[str, int]:
        """Get mapping from object labels to class indices."""
        return self.metadata.labels_to_int
    
    @property
    def num_object_classes(self) -> int:
        """Get number of object classes."""
        return self.metadata.num_object_classes 