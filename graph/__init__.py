"""
Graph module for scene graph construction and manipulation.

This module provides classes and utilities for building, manipulating,
and visualizing scene graphs based on gaze data and object detection.
"""

# Core graph components
from graph.node import Node, VisitRecord
from graph.edge import Edge
from graph.graph import Graph
from graph.utils import AngleUtils, GraphTraversal
from graph.visualizer import GraphVisualizer

# Checkpoint handling
from graph.checkpoint_manager import CheckpointManager, GraphCheckpoint

# Dashboard imports
from graph.dashboard.playback.event import GraphEvent
from graph.dashboard.playback import Playback

# Utility functions
from graph.utils import get_roi, FeatureMatcher
from graph.visualizer import visualize_graph_construction

# Data handling
from graph.io import DataLoader, VideoProcessor
from graph.record import Record

# Graph building
from graph.build_graph import GraphBuilder, build_graph

# Tracing
from graph.graph_tracer import GraphTracer

# Gaze processing
from graph.gaze import GazeProcessor, GazePoint, GazeType

# Object detection
from graph.object_detection import ObjectDetector

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
    'GazeProcessor',
    'GazePoint',
    'GazeType',
]
