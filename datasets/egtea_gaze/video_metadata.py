"""
Video metadata module for EGTEA Gaze+ dataset.

Simplified abstraction for video metadata, loading action records,
and using noun labels from ActionRecord.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import torch

from datasets.egtea_gaze.action_record import ActionRecord
from config.config_utils import get_config
from logger import get_logger

logger = get_logger(__name__)

class VideoMetadata:
    """
    Simplified metadata management for EGTEA Gaze+ videos.
    """

    def __init__(self, config=None):
        """
        Args:
            config: Configuration object. If not provided, will use the default config.
        """
        self.config = config or get_config()
        # Load action records first, which will also load name mappings
        ActionRecord.load_name_mappings(self.config.dataset.egtea.action_annotations)
        self.records_by_video, train_records = self._load_records()
        ActionRecord.initialize_action_mapping(train_records)
        
        self.obj_labels, self.labels_to_int = ActionRecord.get_noun_label_mappings()
        self.num_object_classes = len(self.obj_labels)
        
        # Load video lengths for both train and val splits
        self.video_lengths = {}
        train_lengths = self.load_video_lengths(self.config.dataset.ego_topo.splits.train_video_lengths)
        val_lengths = self.load_video_lengths(self.config.dataset.ego_topo.splits.val_video_lengths)
        
        # Merge both dictionaries
        self.video_lengths.update(train_lengths)
        self.video_lengths.update(val_lengths)
        
        logger.info(f"Initialized metadata for {len(self.records_by_video)} videos with {self.num_object_classes} object classes")

    def _load_records(self) -> Tuple[Dict[str, List[ActionRecord]], List[ActionRecord]]:
        """
        Load action records for train and val splits.

        Returns:
            A mapping from video names to records and a list of training records.
        """
        records_by_video: Dict[str, List[ActionRecord]] = defaultdict(list)
        train_records: List[ActionRecord] = []
        splits = self.config.dataset.ego_topo.splits
        for split_name in ("train", "val"):
            ann_file = getattr(splits, split_name)
            try:
                with open(ann_file) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        record = ActionRecord(line.strip().split("\t"))
                        records_by_video[record.video_name].append(record)
                        if split_name == "train":
                            train_records.append(record)
            except FileNotFoundError:
                logger.error(f"Action records file not found at {ann_file}")
                raise
            except Exception as e:
                logger.error(f"Error loading action records for {split_name} split: {e}")
                raise
        
        for recs in records_by_video.values():
            recs.sort(key=lambda r: r.end_frame)
        
        return records_by_video, train_records

    def get_records_for_video(self, video_name: str) -> List[ActionRecord]:
        """Get action records for a video."""
        return self.records_by_video.get(video_name, [])

    def get_action_frame_range(self, video_name: str) -> Tuple[int, int]:
        """Get the start and end frame of actions for a video."""
        recs = self.get_records_for_video(video_name)
        if not recs:
            raise ValueError(f"No action records found for video {video_name}")
        start = min(r.start_frame for r in recs)
        end = max(r.end_frame for r in recs)
        return start, end

    def get_future_action_labels(self, video_name: str, current_frame: int) -> Optional[Dict[str, torch.Tensor]]:
        """Create labels for future action prediction at a frame."""
        records = self.get_records_for_video(video_name)
        return ActionRecord.create_future_action_labels(records, current_frame)

    def get_all_records(self) -> List[ActionRecord]:
        """Get all action records."""
        return [r for recs in self.records_by_video.values() for r in recs]

    def get_video_names(self) -> List[str]:
        """List all video names."""
        return list(self.records_by_video.keys())

    def get_video_path(self, video_name: str) -> str:
        """Get full path to a video file."""
        return str(Path(self.config.dataset.egtea.raw_videos) / f"{video_name}.mp4")

    def get_gaze_data_path(self, video_name: str) -> str:
        """Get full path to a gaze data file."""
        return str(Path(self.config.dataset.egtea.gaze_data) / f"{video_name}.txt")

    @staticmethod
    def load_video_lengths(video_lengths_file: str) -> Dict[str, int]:
        """
        Load video lengths from the annotation file.
        
        Args:
            video_lengths_file: Path to the video lengths file
            
        Returns:
            Dictionary mapping video names to their lengths
        """
        vid_lengths = {}
        
        try:
            with open(video_lengths_file) as f:
                for line in f.read().strip().split('\n'):
                    k, v = line.split('\t')
                    vid_lengths[k] = int(v)
            logger.info(f"Loaded {len(vid_lengths)} video lengths from {video_lengths_file}")
        except FileNotFoundError:
            logger.warning(f"Video lengths file not found at {video_lengths_file}")
        except Exception as e:
            logger.error(f"Error loading video lengths: {e}")
                
        return vid_lengths 

    def get_video_length(self, video_name: str) -> int:
        """Get the length (number of frames) for a video.
        
        Args:
            video_name: Name of the video
            
        Returns:
            Number of frames in the video, or 0 if not found
        """
        return self.video_lengths.get(video_name, 0) 