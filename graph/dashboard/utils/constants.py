"""Constants used in graph visualization."""
from typing import Dict

from datasets.egtea_gaze.constants import (
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
NODE_BACKGROUND = 'gray'
NODE_BORDER = 'black'
NODE_BASE_SIZE = 60
NODE_FONT_SIZE = 11
NODE_FONT_COLOR = 'white'

# Edge styling
EDGE_WIDTH = 2.5
EDGE_COLOR = '#888'
EDGE_HOVER_OPACITY = 0.1
EDGE_HOVER_SIZE = 6
EDGE_LABEL_FONT_SIZE = 18
EDGE_LABEL_COLOR = '#000'

# Layout parameters
MAX_ANGLE_NODES = 25
MAX_EDGE_HOVER_POINTS = 50
EDGE_HOVER_POINTS = 20
LAYOUT_RADIUS_STEP = 0.5
LAYOUT_JITTER_SCALE = 0.05
LAYOUT_START_JITTER_SCALE = 0.02

# Figure styling
FIGURE_HEIGHT = 450
FIGURE_MARGIN = dict(l=20, r=20, t=20, b=20)
FIGURE_BG_COLOR = 'white'
FIGURE_PAPER_BG_COLOR = 'white'
FIGURE_HOVER_LABEL = dict(
    bgcolor="white",
    font_size=12,
    font_family="Arial"
)

# SVG icon path data
DIAGRAM_PROJECT_ICON_PATH = "M0 80C0 53.5 21.5 32 48 32l96 0c26.5 0 48 21.5 48 48l0 16 192 0 0-16c0-26.5 21.5-48 48-48l96 0c26.5 0 48 21.5 48 48l0 96c0 26.5-21.5 48-48 48l-96 0c-26.5 0-48-21.5-48-48l0-16-192 0 0 16c0 1.7-.1 3.4-.3 5L272 288l96 0c26.5 0 48 21.5 48 48l0 96c0 26.5-21.5 48-48 48l-96 0c-26.5 0-48-21.5-48-48l0-96c0-1.7 .1-3.4 .3-5L144 224l-96 0c-26.5 0-48-21.5-48-48L0 80z"

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