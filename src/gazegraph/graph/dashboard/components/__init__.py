"""Dashboard UI components."""

from gazegraph.graph.dashboard.components.base import BaseComponent
from gazegraph.graph.dashboard.components.video_display import VideoDisplay
from gazegraph.graph.dashboard.components.graph_display import GraphDisplay
from gazegraph.graph.dashboard.components.playback_controls import PlaybackControls
from gazegraph.graph.dashboard.components.meta_info import MetaInfo

__all__ = [
    'BaseComponent',
    'VideoDisplay',
    'GraphDisplay',
    'PlaybackControls',
    'MetaInfo',
] 