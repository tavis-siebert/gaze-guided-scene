"""Constants used in graph visualization components."""
from typing import Dict

from egtea_gaze.constants import (
    GAZE_TYPE_UNTRACKED,
    GAZE_TYPE_FIXATION,
    GAZE_TYPE_SACCADE,
    GAZE_TYPE_UNKNOWN,
    GAZE_TYPE_TRUNCATED
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
DEFAULT_PLAY_INTERVAL_MS = 1000 // 24  # ~24 fps
DEFAULT_FRAME_CACHE_SIZE = 100
DEFAULT_EDGE_HOVER_POINTS = 20 