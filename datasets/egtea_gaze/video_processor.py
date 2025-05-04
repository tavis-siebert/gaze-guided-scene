"""
Unified video handling for EGTEA Gaze+.

Provides Video class for accessing frames, gaze data, metadata, and future action labels.
"""

import torch
import torchvision as tv
from pathlib import Path
from typing import Iterator, List, Optional, Dict, Tuple

from datasets.egtea_gaze.video_metadata import VideoMetadata
from datasets.egtea_gaze.gaze_data.gaze_io_sample import parse_gtea_gaze
from config.config_utils import get_config
from logger import get_logger

logger = get_logger(__name__)

class Video:
    """
    Unified interface for an EGTEA Gaze+ video, including frame streaming,
    gaze points, metadata, and action labels.
    """

    def __init__(self, video_name: str, config=None):
        """
        Args:
            video_name: Name of the video (without extension).
            config: Optional configuration object.
        """
        self.video_name = video_name
        self.config = config or get_config()
        self.metadata = VideoMetadata(self.config)

        # Load paths
        self.video_path = Path(self.config.dataset.egtea.raw_videos) / f"{video_name}.mp4"
        self.gaze_path = Path(self.config.dataset.egtea.gaze_data) / f"{video_name}.txt"

        # Load data
        self.stream = tv.io.VideoReader(str(self.video_path), 'video')
        self.gaze_data = parse_gtea_gaze(str(self.gaze_path))

        # Metadata
        self.length = self.metadata.get_video_length(video_name)
        self.records = self.metadata.get_records_for_video(video_name)
        self.first_frame, self.last_frame = self.metadata.get_action_frame_range(video_name)

        logger.info(f"Loaded video '{video_name}' ({self.length} frames), records: {len(self.records)}")

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int, bool, Optional[any]]]:
        """Iterate over frames with gaze points.

        Yields:
            Tuple of (frame_tensor, frame_number, is_black_frame, gaze_point).
        """
        for frame, pts in ((f['data'], f['pts']) for f in self.stream):
            is_black = frame.count_nonzero().item() == 0
            gp = self._next_gaze_point(pts)
            yield frame, pts, is_black, gp

    def _next_gaze_point(self, pts: int):
        """Retrieve gaze point matching a frame timestamp."""
        # Assuming gaze_data entries include frame or timestamp
        for entry in self.gaze_data:
            if entry.frame == pts:
                return entry
        return None

    def get_future_actions(self, frame_number: int) -> Optional[Dict[str, torch.Tensor]]:
        """Get future action labels at a given frame."""
        return self.metadata.get_future_action_labels(self.video_name, frame_number)

    def get_object_labels(self) -> Dict[int, str]:
        """Get mapping from class index to object label."""
        return self.metadata.obj_labels  # renamed internally, adjust accordingly

    def get_labels_to_int(self) -> Dict[str, int]:
        """Get mapping from object label to class index."""
        return self.metadata.labels_to_int 