"""
Video metadata module for EGTEA Gaze+ dataset.

Simplified abstraction for video metadata, loading object labels,
action records, and video lengths.
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
            config: Optional configuration object.
        """
        self.config = config or get_config()
        self.obj_labels, self.labels_to_int = self._load_object_labels()
        ActionRecord.load_name_mappings(self.config.dataset.egtea.action_annotations)
        self.records_by_video, train_records = self._load_records()
        ActionRecord.set_action_mapping(train_records)
        self.video_lengths = self._load_video_lengths()
        self.num_object_classes = len(self.obj_labels)
        logger.info(f"Initialized metadata for {len(self.records_by_video)} videos")

    def _load_object_labels(self) -> Tuple[Dict[int, str], Dict[str, int]]:
        """
        Load object labels from noun index file.

        Returns:
            A tuple of (class_idx_to_label, label_to_class_idx).
        """
        path = Path(self.config.dataset.egtea.noun_idx_file)
        obj_labels: Dict[int, str] = {}
        labels_to_int: Dict[str, int] = {}
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                label = " ".join(parts[:-1])
                idx = int(parts[-1]) - 1
                obj_labels[idx] = label
                labels_to_int[label] = idx
        logger.info(f"Loaded {len(obj_labels)} object labels")
        return obj_labels, labels_to_int

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
            with open(ann_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = ActionRecord(line.strip().split("\t"))
                    records_by_video[record.video_name].append(record)
                    if split_name == "train":
                        train_records.append(record)
        for recs in records_by_video.values():
            recs.sort(key=lambda r: r.end_frame)
        return records_by_video, train_records

    def _load_video_lengths(self) -> Dict[str, int]:
        """
        Load video lengths from corresponding nframes files.

        Returns:
            Mapping from video name to frame count.
        """
        lengths: Dict[str, int] = {}
        splits = self.config.dataset.ego_topo.splits
        for split_name in ("train", "val"):
            ann_file = getattr(splits, split_name)
            nframes_file = ann_file.replace(".csv", "_nframes.csv")
            try:
                with open(nframes_file) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        vid, val = line.strip().split("\t")
                        lengths[vid] = int(val)
            except Exception as e:
                logger.error(f"Failed loading lengths from {nframes_file}: {e}")
        return lengths

    def get_records_for_video(self, video_name: str) -> List[ActionRecord]:
        """
        Get action records for a specific video.

        Args:
            video_name: Name of the video.

        Returns:
            List of ActionRecord objects.
        """
        return self.records_by_video.get(video_name, [])

    def get_video_length(self, video_name: str) -> int:
        """
        Get the total number of frames in a video.

        Args:
            video_name: Name of the video.

        Returns:
            Frame count.
        """
        return self.video_lengths.get(video_name, 0)

    def get_action_frame_range(self, video_name: str) -> Tuple[int, int]:
        """
        Get the start and end frame of actions for a video.

        Args:
            video_name: Name of the video.

        Returns:
            (first_frame, last_frame).
        """
        recs = self.get_records_for_video(video_name)
        if not recs:
            return 0, self.get_video_length(video_name)
        start = min(r.start_frame for r in recs)
        end = max(r.end_frame for r in recs)
        return start, end

    def get_future_action_labels(self, video_name: str, current_frame: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Create labels for future action prediction at a frame.

        Args:
            video_name: Name of the video.
            current_frame: Frame index.

        Returns:
            Dict with next_action, future_actions, future_actions_ordered or None.
        """
        records = self.get_records_for_video(video_name)
        return ActionRecord.create_future_action_labels(records, current_frame)

    def get_all_records(self) -> List[ActionRecord]:
        """Get all action records."""
        return [r for recs in self.records_by_video.values() for r in recs]

    def get_video_names(self) -> List[str]:
        """List all video names."""
        return list(self.records_by_video.keys())

    def get_video_path(self, video_name: str) -> str:
        """
        Get full path to a video file.

        Args:
            video_name: Name of the video.

        Returns:
            Path as string.
        """
        return str(Path(self.config.dataset.egtea.raw_videos) / f"{video_name}.mp4")

    def get_gaze_data_path(self, video_name: str) -> str:
        """
        Get full path to a gaze data file.

        Args:
            video_name: Name of the video.

        Returns:
            Path as string.
        """
        return str(Path(self.config.dataset.egtea.gaze_data) / f"{video_name}.txt") 