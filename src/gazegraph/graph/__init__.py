"""
Graph module for scene graph construction and manipulation.

This module provides classes and utilities for building, manipulating,
and visualizing scene graphs based on gaze data and object detection.
"""

# Core graph components
from gaze_guided_scene.graph.node import Node, VisitRecord
from gaze_guided_scene.graph.edge import Edge
from gaze_guided_scene.graph.utils import AngleUtils, GraphTraversal

# Import Graph separately to avoid circular dependencies
from gaze_guided_scene.graph.graph import Graph

# Checkpoint handling
from gaze_guided_scene.graph.checkpoint_manager import CheckpointManager, GraphCheckpoint

# Dashboard imports
from gaze_guided_scene.graph.dashboard.playback.event import GraphEvent
from gaze_guided_scene.graph.dashboard.playback import Playback

# Import visualizer after graph to avoid circular imports
from gaze_guided_scene.graph.visualizer import GraphVisualizer, visualize_graph_construction

# Utility functions
from gaze_guided_scene.graph.utils import get_roi, FeatureMatcher

# Graph building
from gaze_guided_scene.graph.graph_builder import GraphBuilder
from gaze_guided_scene.graph.graph_processor import build_graph, build_graphs

# Tracing
from gaze_guided_scene.graph.graph_tracer import GraphTracer

# Gaze processing
from gaze_guided_scene.graph.gaze import GazeProcessor, GazePoint, GazeType

# Object detection
from gaze_guided_scene.graph.object_detection import ObjectDetector

__all__ = [
    'Graph',
    'Node',
    'VisitRecord',
    'Edge',
    'AngleUtils',
    'GraphTraversal',
    'Record',
    'DataLoader',
    'VideoProcessor',
    'ObjectDetector',
    'GraphTracer',
    'CheckpointManager',
    'GraphCheckpoint',
    'GraphBuilder',
    'build_graph',
    'build_graphs',
    'GazeProcessor',
    'GazePoint',
    'GazeType',
]
