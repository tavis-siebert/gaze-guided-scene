"""
Unified video handling for EGTEA Gaze+.

Provides Video class for accessing frames, gaze data, metadata, and future action labels.
"""

import torch
import torchvision as tv
from pathlib import Path
from typing import Iterator, List, Optional, Dict, Tuple, Generator

from gazegraph.datasets.egtea_gaze.video_metadata import VideoMetadata
from gazegraph.datasets.egtea_gaze.action_record import ActionRecord
from gazegraph.datasets.egtea_gaze.gaze_io_sample import parse_gtea_gaze
from gazegraph.config.config_utils import get_config
from gazegraph.logger import get_logger
from gazegraph.graph.gaze import GazeProcessor, GazePoint

# Import parse_gtea_gaze from the correct location
import sys
import os
from pathlib import Path

# Add data directory to path if needed
data_dir = Path(os.environ.get('DATA_DIR', 'data'))
gaze_data_path = data_dir / 'egtea_gaze' / 'gaze_data'
sys.path.append(str(gaze_data_path))

logger = get_logger(__name__)

class Video:
    """
    Unified interface for an EGTEA Gaze+ video, including frame streaming,
    processed gaze points, and metadata.
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
        self.video_metadata = self.stream.get_metadata()
        self.fps = self.video_metadata["video"]["fps"][0]
        
        raw_gaze = parse_gtea_gaze(str(self.gaze_path))
        self.gaze_processor = GazeProcessor(self.config, raw_gaze)
        self.records = self.metadata.get_records_for_video(video_name)
        self.first_frame, self.last_frame = self.metadata.get_action_frame_range(video_name)
        self.length = self.metadata.get_video_length(video_name)

        logger.info(f"Loaded video '{video_name}' with {len(self.records)} records, "
                   f"action frames: {self.first_frame} to {self.last_frame}, "
                   f"length: {self.length}, fps: {self.fps}")

    def __iter__(self):
        """Prepare iterators for frames and gaze."""
        self._frame_iter = iter(self.stream)
        self._gaze_iter = iter(self.gaze_processor)
        return self

    def __next__(self) -> Tuple[torch.Tensor, int, bool, Optional[GazePoint]]:
        """Get next frame and corresponding processed gaze."""
        frame_dict = next(self._frame_iter)
        frame = frame_dict['data']
        pts = frame_dict['pts']
        is_black = frame.count_nonzero().item() == 0
        gaze_point = next(self._gaze_iter)
        return frame, pts, is_black, gaze_point
        
    def seek_to_frame(self, frame_num: int):
        """
        Seek to a specific frame in the video stream and gaze data.

        Args:
            frame_num: The frame index to seek to (0-indexed).
        """
        try:
            # Convert frame index to seconds
            time_sec = frame_num / self.fps

            # Seek to the start frame in the video stream
            self.stream.seek(time_sec)

            # Create a new iterator after seeking (necessary for some backends)
            self._frame_iter = iter(self.stream)

            # Also seek in gaze data to maintain synchronization
            # Re-initializing the gaze processor is a simple way to reset the iterator
            self.gaze_processor = GazeProcessor(self.config, parse_gtea_gaze(str(self.gaze_path)))
            self._gaze_iter = iter(self.gaze_processor)

            # Skip gaze points until we reach frame_num to synchronize with the video stream
            current_gaze_frame = 0
            while current_gaze_frame < frame_num:
                next(self._gaze_iter)
                current_gaze_frame += 1

            logger.debug(f"Seeked video and gaze data for {self.video_name} to frame {frame_num}")

        except Exception as e:
            logger.warning(f"Error during seek to frame {frame_num}: {e}")


    def get_future_actions(self, frame_number: int) -> Optional[Dict[str, torch.Tensor]]:
        """Get future action labels at a given frame."""
        return self.metadata.get_future_action_labels(self.video_name, frame_number)

    def get_labels_to_int(self) -> Dict[str, int]:
        """Get mapping from object label to class index."""
        return self.metadata.object_label_to_id
    
    def get_action_names(self) -> Dict[int, str]:
        """Get mapping from action indices to human-readable names."""
        return ActionRecord.get_action_names() 