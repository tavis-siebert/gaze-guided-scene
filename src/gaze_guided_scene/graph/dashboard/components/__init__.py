"""Dashboard UI components."""

from gaze_guided_scene.graph.dashboard.components.base import BaseComponent
from gaze_guided_scene.graph.dashboard.components.video_display import VideoDisplay
from gaze_guided_scene.graph.dashboard.components.graph_display import GraphDisplay
from gaze_guided_scene.graph.dashboard.components.playback_controls import PlaybackControls
from gaze_guided_scene.graph.dashboard.components.meta_info import MetaInfo

__all__ = [
    'BaseComponent',
    'VideoDisplay',
    'GraphDisplay',
    'PlaybackControls',
    'MetaInfo',
] 