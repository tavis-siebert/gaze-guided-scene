"""
Module for processing gaze data in real-time, filtering noise and classifying fixations.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import IntEnum
import numpy as np
from collections import deque

from gazegraph.datasets.egtea_gaze.constants import (
    GAZE_TYPE_FIXATION,
    GAZE_TYPE_SACCADE,
    GAZE_TYPE_UNTRACKED,
    GAZE_TYPE_UNKNOWN,
    GAZE_TYPE_TRUNCATED,
)
from gazegraph.config.config_utils import DotDict
from gazegraph.logger import get_logger

logger = get_logger(__name__)


class GazeType(IntEnum):
    """Enum for gaze types, extended from constants."""

    UNTRACKED = GAZE_TYPE_UNTRACKED
    FIXATION = GAZE_TYPE_FIXATION
    SACCADE = GAZE_TYPE_SACCADE
    UNKNOWN = GAZE_TYPE_UNKNOWN
    TRUNCATED = GAZE_TYPE_TRUNCATED

    # Extended types for processed gaze data
    NOISY_FIXATION = (
        10  # Choosing a value that doesn't conflict with existing constants
    )
    SMOOTHED_FIXATION = (
        11  # For saccades reclassified as fixations based on spatial stability
    )


@dataclass
class GazePoint:
    """Represents a single gaze data point with position and classification."""

    x: float  # Normalized x coordinate (0-1)
    y: float  # Normalized y coordinate (0-1)
    raw_type: GazeType  # Original gaze type from dataset
    type: GazeType  # Classified gaze type after processing
    frame_idx: int  # Frame index in the video

    @property
    def position(self) -> Tuple[float, float]:
        """Return the position as a tuple (x, y)."""
        return (self.x, self.y)


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
        self.fixation_threshold = config.gaze.fixation_window_threshold
        self.gaze_data = gaze_data
        self.current_idx = 0
        self.consecutive_fixations = 0  # Counter for consecutive fixations

        # Parameters for spatial stability detection - with fallback values
        self.spatial_window_size = config.gaze.spatial_window_size
        self.spatial_distance_threshold = config.gaze.spatial_distance_threshold
        self.reclassify_saccades = config.gaze.reclassify_saccades

        # Buffer for recent gaze points (for spatial stability analysis)
        self.gaze_buffer = deque(maxlen=self.spatial_window_size)
        self.unknown_neighbor_distance = config.gaze.unknown_neighbor_distance

        if config.gaze.preprocess_gaze:
            self._preprocess_gaze_data()

    def _preprocess_gaze_data(self) -> None:
        """
        Preprocess the entire gaze data sequence to smooth out incorrect classifications
        and handle spatial stability.
        """
        # Create a copy of the original data to modify
        processed_data = self.gaze_data.copy()

        # First pass: identify spatially stable sequences that might be mistakenly classified as saccades
        for i in range(len(processed_data)):
            # Skip untracked or truncated data
            if int(processed_data[i, 2]) in (GazeType.UNTRACKED, GazeType.TRUNCATED):
                continue

            # Look at a window of points
            window_start = max(0, i - self.spatial_window_size // 2)
            window_end = min(len(processed_data), i + self.spatial_window_size // 2 + 1)
            window = processed_data[window_start:window_end]

            # Skip if too few valid points in window
            valid_points = window[
                ~np.isin(window[:, 2], [GazeType.UNTRACKED, GazeType.TRUNCATED])
            ]
            if (
                len(valid_points) < 3
            ):  # Need at least 3 points for meaningful stability analysis
                continue

            # Check spatial stability
            current_point = processed_data[i, :2]
            distances = np.linalg.norm(valid_points[:, :2] - current_point, axis=1)
            median_distance = np.median(distances)

            # If spatially stable but classified as saccade, reclassify
            if (
                median_distance < self.spatial_distance_threshold
                and int(processed_data[i, 2]) == GazeType.SACCADE
                and self.reclassify_saccades
            ):
                processed_data[i, 2] = GazeType.SMOOTHED_FIXATION

        # Second pass: fill short gaps between fixations
        for i in range(1, len(processed_data) - 1):
            if (
                int(processed_data[i, 2]) == GazeType.SACCADE
                and (
                    int(processed_data[i - 1, 2])
                    in (GazeType.FIXATION, GazeType.SMOOTHED_FIXATION)
                )
                and (
                    int(processed_data[i + 1, 2])
                    in (GazeType.FIXATION, GazeType.SMOOTHED_FIXATION)
                )
            ):
                # Check if the positions are close
                prev_pos = processed_data[i - 1, :2]
                next_pos = processed_data[i + 1, :2]
                current_pos = processed_data[i, :2]

                # Calculate distances
                dist_to_prev = np.linalg.norm(current_pos - prev_pos)
                dist_to_next = np.linalg.norm(current_pos - next_pos)

                # If current point is spatially close to surrounding fixations, reclassify
                if (
                    dist_to_prev < self.spatial_distance_threshold
                    and dist_to_next < self.spatial_distance_threshold
                ):
                    processed_data[i, 2] = GazeType.SMOOTHED_FIXATION

        # Handle UNKNOWN gaze points by interpolation
        for i in range(len(processed_data)):
            if int(processed_data[i, 2]) in (
                GazeType.UNKNOWN,
                GazeType.UNTRACKED,
                GazeType.TRUNCATED,
            ):
                interp = self._interpolate_gaze(i)
                if interp is not None:
                    processed_data[i, 0] = interp[0]
                    processed_data[i, 1] = interp[1]
                    processed_data[i, 2] = interp[2]

        # Update the internal gaze data with the processed version
        self.gaze_data = processed_data

    def _interpolate_gaze(self, idx: int) -> Optional[Tuple[float, float, int]]:
        """Interpolate gaze position and type for UNKNOWN at idx."""
        n = len(self.gaze_data)
        prev_idx, next_idx = idx - 1, idx + 1
        # Find previous valid
        while prev_idx >= 0 and int(self.gaze_data[prev_idx, 2]) in (
            GazeType.UNTRACKED,
            GazeType.TRUNCATED,
            GazeType.UNKNOWN,
        ):
            prev_idx -= 1
        # Find next valid
        while next_idx < n and int(self.gaze_data[next_idx, 2]) in (
            GazeType.UNTRACKED,
            GazeType.TRUNCATED,
            GazeType.UNKNOWN,
        ):
            next_idx += 1
        if prev_idx < 0 or next_idx >= n:
            return None
        prev = self.gaze_data[prev_idx]
        next = self.gaze_data[next_idx]
        # Interpolate position
        alpha = (idx - prev_idx) / (next_idx - prev_idx)
        x = prev[0] + alpha * (next[0] - prev[0])
        y = prev[1] + alpha * (next[1] - prev[1])
        # Decide type
        prev_type, next_type = int(prev[2]), int(next[2])
        dist = np.linalg.norm(prev[:2] - next[:2])
        if (
            prev_type == GazeType.SACCADE and next_type == GazeType.SACCADE
        ) or dist > self.unknown_neighbor_distance:
            interp_type = GazeType.SACCADE
        else:
            interp_type = GazeType.FIXATION
        return (x, y, interp_type)

    def __iter__(self) -> "GazeProcessor":
        """Make the processor iterable to process gaze points."""
        self.current_idx = 0
        self.consecutive_fixations = 0
        self.gaze_buffer.clear()
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

        # Update the gaze buffer for spatial stability analysis
        self.gaze_buffer.append((x, y, raw_type))

        type = self._classify_gaze_type(frame_idx)

        return GazePoint(x=x, y=y, raw_type=raw_type, type=type, frame_idx=frame_idx)

    def _is_spatially_stable(self, current_pos: Tuple[float, float]) -> bool:
        """
        Determine if the current gaze position is spatially stable based on recent history.

        Args:
            current_pos: Current position (x, y)

        Returns:
            bool: True if the position is spatially stable
        """
        if len(self.gaze_buffer) < 3:  # Need at least a few points for stability check
            return False

        # Extract valid positions from buffer
        valid_positions = [
            (x, y)
            for x, y, t in self.gaze_buffer
            if t not in (GazeType.UNTRACKED, GazeType.TRUNCATED)
        ]

        if len(valid_positions) < 3:
            return False

        # Calculate distances from current position to all valid positions
        distances = [
            np.sqrt((x - current_pos[0]) ** 2 + (y - current_pos[1]) ** 2)
            for x, y in valid_positions
        ]

        # If median distance is below threshold, consider it stable
        return np.median(distances) < self.spatial_distance_threshold

    def _classify_gaze_type(self, frame_idx: int) -> GazeType:
        """
        Classify the gaze type for the current frame.

        Args:
            frame_idx: Index of the current frame

        Returns:
            GazeType: The classified gaze type
        """
        current_type = int(self.gaze_data[frame_idx, 2])
        current_pos = (self.gaze_data[frame_idx, 0], self.gaze_data[frame_idx, 1])

        # If current frame is a smoothed fixation (from preprocessing)
        if current_type == GazeType.SMOOTHED_FIXATION:
            self.consecutive_fixations += 1
            return GazeType.FIXATION

        # If untracked or truncated, reset counter and return original type
        if current_type in (GazeType.UNTRACKED, GazeType.TRUNCATED):
            self.consecutive_fixations = 0
            return GazeType(current_type)

        # If current frame is a saccade but spatially stable, consider reclassifying
        if current_type == GazeType.SACCADE:
            if self._is_spatially_stable(current_pos) and self.reclassify_saccades:
                self.consecutive_fixations += 1
                return GazeType.FIXATION
            else:
                self.consecutive_fixations = 0
                return GazeType.SACCADE

        # Handle original fixation classification
        if current_type == GazeType.FIXATION:
            # Current frame is a fixation; increment counter
            self.consecutive_fixations += 1

            # Once we've reached threshold fixations, all subsequent fixations are true
            if self.consecutive_fixations >= self.fixation_threshold:
                return GazeType.FIXATION

            # Look ahead to see if there are enough consecutive fixations
            frames_needed = self.fixation_threshold - self.consecutive_fixations
            lookahead_end = min(frame_idx + 1 + frames_needed, len(self.gaze_data))
            lookahead = self.gaze_data[frame_idx + 1 : lookahead_end, 2]

            # If there aren't enough future frames, we cannot complete a fixation block
            if len(lookahead) < frames_needed:
                return GazeType.NOISY_FIXATION

            # Check if future frames are fixations or reclassified saccades (spatially stable)
            valid_fixation_types = (GazeType.FIXATION, GazeType.SMOOTHED_FIXATION)
            if all(int(val) in valid_fixation_types for val in lookahead):
                return GazeType.FIXATION
            else:
                return GazeType.NOISY_FIXATION

        # Default case: return original type
        return GazeType(current_type)

    def reset(self) -> None:
        """Reset the processor to start from the beginning."""
        self.current_idx = 0
        self.consecutive_fixations = 0
        self.gaze_buffer.clear()

    def process_all(self) -> List[GazePoint]:
        """
        Process the entire gaze data sequence at once.

        Returns:
            List[GazePoint]: List of processed gaze points
        """
        self.reset()
        return list(self)
