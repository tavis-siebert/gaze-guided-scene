"""Constants used in graph visualization components."""
from typing import Dict

from egtea_gaze.constants import (
    GAZE_TYPE_UNTRACKED,
    GAZE_TYPE_FIXATION,
    GAZE_TYPE_SACCADE,
    GAZE_TYPE_UNKNOWN,
    GAZE_TYPE_TRUNCATED,
    FPS
)

# Gaze type styling information
GAZE_TYPE_INFO: Dict[int, Dict[str, str]] = {
    GAZE_TYPE_UNTRACKED: {"color": "gray", "label": "Untracked"},
    GAZE_TYPE_FIXATION: {"color": "blue", "label": "Fixation"},
    GAZE_TYPE_SACCADE: {"color": "red", "label": "Saccade"},
    GAZE_TYPE_UNKNOWN: {"color": "purple", "label": "Unknown"},
    GAZE_TYPE_TRUNCATED: {"color": "orange", "label": "Truncated"}
}

# Node styling
NODE_BACKGROUND = {
    "default": "gray",
    "last_added": "blue"
}

NODE_BORDER = {
    "default": "black", 
    "current": GAZE_TYPE_INFO[GAZE_TYPE_FIXATION]["color"]
}

# Dashboard settings
TARGET_FPS = 12
DEFAULT_PLAY_INTERVAL_MS = 1000 // TARGET_FPS
FRAME_INTERVALS = [1, 2, 5, 10]
PLAYBACK_SPEED_DEFAULT = 2

def compute_playback_speed(frame_interval: int, original_fps: int = FPS) -> float:
    """Compute playback speed multiplier based on frame interval and original FPS."""
    return TARGET_FPS / original_fps * frame_interval


PLAYBACK_SPEEDS = {interval: f"{compute_playback_speed(interval):.1f}x" for interval in FRAME_INTERVALS}
PLAYBACK_SPEED_MIN = min(FRAME_INTERVALS)
PLAYBACK_SPEED_MAX = max(FRAME_INTERVALS)
PLAYBACK_SPEED_MARKS = {str(speed): label for speed, label in PLAYBACK_SPEEDS.items()} 