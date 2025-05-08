"""
Graph module for scene graph construction and manipulation.

This module provides classes and utilities for building, manipulating,
and visualizing scene graphs based on gaze data and object detection.
"""

# Core graph components
from gazegraph.graph.node import Node, VisitRecord
from gazegraph.graph.edge import Edge
from gazegraph.graph.utils import AngleUtils, GraphTraversal

# Import Graph separately to avoid circular dependencies
from gazegraph.graph.graph import Graph

# Checkpoint handling
from gazegraph.graph.checkpoint_manager import CheckpointManager, GraphCheckpoint

# Dashboard imports
from gazegraph.graph.dashboard.playback.event import GraphEvent
from gazegraph.graph.dashboard.playback import Playback

# Import visualizer after graph to avoid circular imports
from gazegraph.graph.visualizer import GraphVisualizer, visualize_graph_construction

# Utility functions
from gazegraph.graph.utils import get_roi, FeatureMatcher

# Graph building
from gazegraph.graph.graph_builder import GraphBuilder
from gazegraph.graph.graph_processor import build_graph, build_graphs

# Tracing
from gazegraph.graph.graph_tracer import GraphTracer

# Gaze processing
from gazegraph.graph.gaze import GazeProcessor, GazePoint, GazeType

# Object detection
from gazegraph.graph.object_detection import ObjectDetector

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
