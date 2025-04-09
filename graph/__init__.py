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
from graph.io import Record, DataLoader, VideoProcessor

# Graph building
from graph.build_graph import GraphBuilder, build_graph

# Tracing
from graph.graph_tracer import GraphTracer

# Action utils
from graph.action_utils import ActionUtils

# Gaze processing
from graph.gaze import GazeProcessor, GazePoint

# Object detection
from graph.object_detection import ObjectDetector, Detection

__all__ = [
    'Node',
    'VisitRecord',
    'Edge',
    'Graph',
    'GraphCheckpoint',
    'AngleUtils',
    'GraphTraversal',
    'GraphVisualizer',
    
    # Utility classes
    'FeatureMatcher',
    
    # Data handling
    'Record',
    'DataLoader',
    'VideoProcessor',
    
    # Graph building
    'GraphBuilder',
    
    # Tracing and visualization
    'GraphTracer',
    'GraphEvent',
    'Playback',
    'visualize_graph_construction',
    
    # Functions
    'get_roi',
    'build_graph',
    
    # New modules
    'CheckpointManager',
    'ActionUtils',
    
    # Gaze processing
    'GazeProcessor',
    'GazePoint',
    
    # Object detection
    'ObjectDetector',
    'Detection'
]
