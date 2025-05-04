"""
Video metadata module for EGTEA Gaze+ dataset.

This module provides classes for handling video metadata including length,
loading and accessing action records.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
import glob
import torch
from tqdm import tqdm

from datasets.egtea_gaze.action_record import ActionRecord
from datasets.egtea_gaze.constants import NUM_ACTION_CLASSES
from config.config_utils import get_config
from logger import get_logger

logger = get_logger(__name__)

class VideoMetadata:
    """
    Encapsulates metadata for the EGTEA Gaze+ video dataset.
    
    Features:
    - Loads video lengths
    - Manages action records grouped by video name
    - Provides access to annotated actions and their mappings
    """
    # Class-level variables for sharing metadata across instances
    _initialized = False
    _video_lengths = {}
    _records_by_video = defaultdict(list)
    _all_records = []
    _obj_labels = {}
    _labels_to_int = {}
    
    def __init__(self, config=None):
        """
        Initialize video metadata.
        
        Args:
            config: Configuration object (optional)
        """
        self.config = config or get_config()
        
        # Load data if not already loaded at class level
        if not VideoMetadata._initialized:
            self._initialize_class_data()
    
    def _initialize_class_data(self):
        """Initialize all class-level data including action records and name mappings."""
        # Load mappings and object labels
        self._load_object_labels()
        
        # Load action name mappings
        self._load_action_name_mappings()
        
        # Load video metadata and action records from train split
        train_file = self.config.dataset.ego_topo.splits.train
        self._load_video_data(train_file, is_training=True)
        
        # Load validation split if needed
        val_file = self.config.dataset.ego_topo.splits.val
        if val_file != train_file:  # Only load val if different from train
            self._load_video_data(val_file, is_training=False)
        
        # Initialize action id mapping using training records
        if not ActionRecord.get_action_mapping():
            train_records = [r for r in VideoMetadata._all_records 
                            if self._is_training_video(r.path)]
            ActionRecord.set_action_mapping(train_records)
        
        VideoMetadata._initialized = True
        logger.info(f"Initialized VideoMetadata with {len(VideoMetadata._video_lengths)} videos "
                   f"and {len(VideoMetadata._all_records)} action records")
    
    def _is_training_video(self, video_path: str) -> bool:
        """Check if a video is in the training set."""
        return any(video_path in r for r in 
                  self._get_records_from_file(self.config.dataset.ego_topo.splits.train))
    
    def _load_object_labels(self):
        """Load object labels from noun index file."""
        noun_idx_path = Path(self.config.dataset.egtea.noun_idx_file)
        
        with open(noun_idx_path) as f:
            for line in f:
                if not line.strip():
                    continue
                    
                line = line.strip().split(' ')
                if len(line) < 2:
                    continue
                    
                class_idx = int(line[-1]) - 1  # Convert to 0-indexed
                label = ' '.join(line[:-1])
                
                VideoMetadata._obj_labels[class_idx] = label
                VideoMetadata._labels_to_int[label] = class_idx
                
        logger.info(f"Loaded {len(VideoMetadata._obj_labels)} object labels")
    
    def _load_action_name_mappings(self):
        """Load action name mappings."""
        action_annotations_dir = self.config.dataset.egtea.action_annotations
        ActionRecord.load_name_mappings(action_annotations_dir)
    
    def _load_video_data(self, ann_file: str, is_training: bool = False):
        """
        Load video lengths and action records from annotation file.
        
        Args:
            ann_file: Path to annotation file
            is_training: Whether this is the training split
        """
        # Load video lengths
        self._load_video_lengths(ann_file)
        
        # Load action records
        records = self._get_records_from_file(ann_file)
        
        # Add these records to our collections
        for record in records:
            video_name = self._extract_video_name(record.path)
            if video_name not in VideoMetadata._records_by_video:
                VideoMetadata._records_by_video[video_name] = []
            VideoMetadata._records_by_video[video_name].append(record)
            if record not in VideoMetadata._all_records:
                VideoMetadata._all_records.append(record)
        
        # Sort records by start frame for each video
        for video in VideoMetadata._records_by_video:
            VideoMetadata._records_by_video[video].sort(key=lambda r: r.start_frame)
    
    def _load_video_lengths(self, ann_file: str):
        """
        Load video lengths from the annotation file.
        
        Args:
            ann_file: Path to the annotation file
        """
        nframes_file = ann_file.replace('.csv', '_nframes.csv')
        
        try:
            with open(nframes_file) as f:
                for line in f.read().strip().split('\n'):
                    if not line.strip():
                        continue
                    k, v = line.strip().split('\t')
                    VideoMetadata._video_lengths[k] = int(v)
        except Exception as e:
            logger.error(f"Error loading video lengths from {nframes_file}: {e}")
    
    def _get_records_from_file(self, ann_file: str) -> List[ActionRecord]:
        """
        Load action records from an annotation file.
        
        Args:
            ann_file: Path to the annotation file
            
        Returns:
            List of ActionRecord objects
        """
        records = []
        try:
            with open(ann_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = ActionRecord(line.strip().split('\t'))
                    records.append(record)
            logger.info(f"Loaded {len(records)} records from {ann_file}")
        except Exception as e:
            logger.error(f"Error loading records from {ann_file}: {e}")
        
        return records
    
    @staticmethod
    def _extract_video_name(path: str) -> str:
        """Extract video name from path."""
        return os.path.basename(path).split('-')[0]
    
    def get_action_frame_range(self, video_name: str) -> Tuple[int, int]:
        """
        Get the first and last frame from action annotations for a video.
        
        Args:
            video_name: Name of the video
            
        Returns:
            Tuple of (first_frame, last_frame)
        """
        records = self.get_records_for_video(video_name)
        
        if not records:
            return 0, self.get_video_length(video_name)
            
        first_frame = min(r.start_frame for r in records)
        last_frame = max(r.end_frame for r in records)
        
        return first_frame, last_frame
    
    def get_records_for_video(self, video_name: str) -> List[ActionRecord]:
        """
        Get action records for a specific video.
        
        Args:
            video_name: Name of the video
            
        Returns:
            List of ActionRecord objects for the video
        """
        return VideoMetadata._records_by_video.get(video_name, [])
    
    def get_video_length(self, video_name: str) -> int:
        """
        Get the length (in frames) of a video.
        
        Args:
            video_name: Name of the video
            
        Returns:
            Number of frames in the video
        """
        return VideoMetadata._video_lengths.get(video_name, 0)
    
    @property
    def obj_labels(self) -> Dict[int, str]:
        """Get object labels dictionary (class_idx -> label)."""
        return VideoMetadata._obj_labels.copy()
    
    @property
    def labels_to_int(self) -> Dict[str, int]:
        """Get mapping from object labels to class indices."""
        return VideoMetadata._labels_to_int.copy()
    
    @property
    def num_object_classes(self) -> int:
        """Get number of object classes."""
        return len(VideoMetadata._obj_labels)
    
    @staticmethod
    def get_future_action_labels(video_name: str, current_frame: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Create action label tensors for future actions.
        
        Args:
            video_name: Name of the video
            current_frame: Current frame number
            
        Returns:
            Dictionary containing action labels or None if insufficient data
        """
        records = VideoMetadata._records_by_video.get(video_name, [])
        return ActionRecord.create_future_action_labels(records, current_frame)
    
    @staticmethod
    def get_all_records() -> List[ActionRecord]:
        """Get all action records."""
        return VideoMetadata._all_records.copy()
    
    @staticmethod
    def get_video_names() -> List[str]:
        """Get list of all video names."""
        return list(VideoMetadata._records_by_video.keys())
    
    def get_video_path(self, video_name: str) -> str:
        """
        Get full path to a video file.
        
        Args:
            video_name: Name of the video
            
        Returns:
            Full path to the video file
        """
        return str(Path(self.config.dataset.egtea.raw_videos) / f"{video_name}.mp4")
    
    def get_gaze_data_path(self, video_name: str) -> str:
        """
        Get full path to a gaze data file.
        
        Args:
            video_name: Name of the video
            
        Returns:
            Full path to the gaze data file
        """
        return str(Path(self.config.dataset.egtea.gaze_data) / f"{video_name}.txt") 