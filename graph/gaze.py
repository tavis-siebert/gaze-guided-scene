"""
Module for processing gaze data in real-time, filtering noise and classifying fixations.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Any, Iterator
from enum import IntEnum
import numpy as np
from egtea_gaze.constants import (
    GAZE_TYPE_FIXATION, 
    GAZE_TYPE_SACCADE, 
    GAZE_TYPE_UNTRACKED, 
    GAZE_TYPE_UNKNOWN,
    GAZE_TYPE_TRUNCATED
)
from config.config_utils import DotDict


class GazeType(IntEnum):
    """Enum for gaze types, extended from constants."""
    UNTRACKED = GAZE_TYPE_UNTRACKED
    FIXATION = GAZE_TYPE_FIXATION
    SACCADE = GAZE_TYPE_SACCADE
    UNKNOWN = GAZE_TYPE_UNKNOWN
    TRUNCATED = GAZE_TYPE_TRUNCATED
    
    # Extended type for processed gaze data
    NOISY_FIXATION = 10  # Choosing a value that doesn't conflict with existing constants


@dataclass
class GazePoint:
    """Represents a single gaze data point with position and classification."""
    x: float  # Normalized x coordinate (0-1)
    y: float  # Normalized y coordinate (0-1)
    raw_type: GazeType  # Original gaze type from dataset
    processed_type: GazeType  # Classified gaze type after processing
    frame_idx: int  # Frame index in the video


class GazeProcessor:
    """
    Processes raw gaze data to identify true fixations vs. noisy fixations.
    Acts as an iterator to provide frame-by-frame access to processed gaze data.
    """
    
    def __init__(self, config: DotDict, gaze_data: np.ndarray):
        """
        Initialize the gaze processor.
        
        Args:
            config: Configuration dictionary
            gaze_data: Array of gaze data points, each with x, y, type
        """
        self.fixation_threshold = config.graph.fixation_window_threshold
        self.gaze_data = gaze_data
        self.current_idx = 0
        self.consecutive_fixations = 0  # Counter for consecutive fixations
    
    def __iter__(self) -> 'GazeProcessor':
        """Make the processor iterable to process gaze points."""
        self.current_idx = 0
        self.consecutive_fixations = 0
        return self
    
    def __next__(self) -> GazePoint:
        """
        Get the next processed gaze point.
        
        Returns:
            GazePoint: The processed gaze point for the current frame
            
        Raises:
            StopIteration: When there are no more frames to process
        """
        if self.current_idx >= len(self.gaze_data):
            raise StopIteration
        
        gaze_point = self._process_current_frame()
        self.current_idx += 1
        return gaze_point
    
    def _process_current_frame(self) -> GazePoint:
        """
        Process the current frame's gaze data.
        
        Returns:
            GazePoint: The processed gaze point
        """
        frame_idx = self.current_idx
        x, y = self.gaze_data[frame_idx, 0], self.gaze_data[frame_idx, 1]
        raw_type = GazeType(int(self.gaze_data[frame_idx, 2]))
        
        processed_type = self._classify_gaze_type(frame_idx)
        
        return GazePoint(
            x=x,
            y=y,
            raw_type=raw_type,
            processed_type=processed_type,
            frame_idx=frame_idx
        )
    
    def _classify_gaze_type(self, frame_idx: int) -> GazeType:
        """
        Classify the gaze type for the current frame.
        
        Args:
            frame_idx: Index of the current frame
            
        Returns:
            GazeType: The classified gaze type
        """
        current = int(self.gaze_data[frame_idx, 2])
        
        # If current frame is not a fixation, reset counter and return original type
        if current != GazeType.FIXATION:
            self.consecutive_fixations = 0
            return GazeType(current)
        
        # Current frame is a fixation; increment counter
        self.consecutive_fixations += 1
        
        # Once we've reached threshold fixations, all subsequent fixations are true
        if self.consecutive_fixations >= self.fixation_threshold:
            return GazeType.FIXATION
        
        # Look ahead to see if there are enough consecutive fixations
        frames_needed = self.fixation_threshold - self.consecutive_fixations
        lookahead_end = min(frame_idx + 1 + frames_needed, len(self.gaze_data))
        lookahead = self.gaze_data[frame_idx + 1:lookahead_end, 2]
        
        # If there aren't enough future frames, we cannot complete a fixation block
        if len(lookahead) < frames_needed:
            return GazeType.NOISY_FIXATION
        
        # If the needed future frames are all fixations, then the current frame
        # is part of a valid fixation block
        if all(int(val) == GazeType.FIXATION for val in lookahead):
            return GazeType.FIXATION
        else:
            return GazeType.NOISY_FIXATION
            
    def reset(self) -> None:
        """Reset the processor to start from the beginning."""
        self.current_idx = 0
        self.consecutive_fixations = 0
    
    def process_all(self) -> List[GazePoint]:
        """
        Process the entire gaze data sequence at once.
        
        Returns:
            List[GazePoint]: List of processed gaze points
        """
        self.reset()
        return list(self) 