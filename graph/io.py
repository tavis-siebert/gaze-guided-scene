import torch
import torchvision as tv
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set, Iterator
from collections import defaultdict

from graph.record import Record
from egtea_gaze.constants import NUM_ACTION_CLASSES

class VideoProcessor:
    """
    Handles video loading and frame extraction.
    """
    
    def __init__(self, video_path):
        """
        Initialize the video processor.
        
        Args:
            video_path: Path to the video file
        """
        self.video_path = video_path
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


class DataLoader:
    """
    Handles loading and processing of dataset files.
    """
    
    @staticmethod
    def load_object_labels(noun_idx_path: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
        """
        Load object labels from the noun index file.
        
        Args:
            noun_idx_path: Path to the noun index file
            
        Returns:
            Tuple of (class_idx -> label, label -> class_idx) dictionaries
        """
        obj_labels, labels_to_int = {}, {}
        
        with open(noun_idx_path) as f:
            for line in f:
                line = line.split(' ')
                class_idx, label = int(line[1]) - 1, line[0]
                obj_labels[class_idx] = label
                labels_to_int[label] = class_idx
                
        return obj_labels, labels_to_int
    
    @staticmethod
    def load_video_lengths(ann_file: str) -> Dict[str, int]:
        """
        Load video lengths from the annotation file.
        
        Args:
            ann_file: Path to the annotation file
            
        Returns:
            Dictionary mapping video names to their lengths
        """
        nframes_file = ann_file.replace('.csv', '_nframes.csv')
        vid_lengths = {}
        
        with open(nframes_file) as f:
            for line in f.read().strip().split('\n'):
                k, v = line.split('\t')
                vid_lengths[k] = int(v)
                
        return vid_lengths
    
    @staticmethod
    def load_records(ann_file: str) -> Tuple[List[Record], Dict[str, List[Record]]]:
        """
        Load action records from the annotation file.
        
        Args:
            ann_file: Path to the annotation file
            
        Returns:
            Tuple of (all records, records grouped by video)
        """
        records = []
        records_by_vid = defaultdict(list)
        
        with open(ann_file) as f:
            for line in f:
                record = Record(line.strip().split('\t'))
                records.append(record)
                records_by_vid[record.path].append(record)
                
        # Sort records by end frame for each video
        for video in records_by_vid:
            records_by_vid[video].sort(key=lambda r: r.end_frame)
            
        return records, records_by_vid