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

# Dashboard imports
from graph.dashboard.playback.event import GraphEvent
from graph.dashboard.playback import GraphPlayback

# Utility functions
from graph.utils import get_roi, FeatureMatcher
from graph.visualizer import visualize_graph_construction

# Data handling
from graph.io import Record, DataLoader, get_future_action_labels, VideoProcessor

# Graph building
from graph.build_graph import GraphBuilder, build_graph

# Tracing
from graph.graph_tracer import GraphTracer

__all__ = [
    'Node',
    'VisitRecord',
    'Edge',
    'Graph',
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
    'GraphPlayback',
    'visualize_graph_construction',
    
    # Functions
    'get_future_action_labels',
    'get_roi',
    'build_graph'
]
