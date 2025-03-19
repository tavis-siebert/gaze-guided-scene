"""
Graph module for scene graph construction and manipulation.

This module provides classes and utilities for building, manipulating,
and visualizing scene graphs based on gaze data and object detection.
"""

# Core graph components
from graph.node import Node, VisitRecord, NeighborInfo, NodeManager
from graph.edge import Edge
from graph.graph import Graph, Position, EdgeFeature, EdgeIndex

# Utility functions
from graph.utils import get_roi, AngleUtils, GraphTraversal, FeatureMatcher
from graph.visualizer import (
    GraphVisualizer, 
    GraphEvent, 
    GraphPlayback, 
    InteractiveGraphVisualizer,
    visualize_graph_construction
)

# Data handling
from graph.io import Record, DataLoader, get_future_action_labels, VideoProcessor

# Graph building
from graph.build_graph import GraphBuilder, build_graph

# Tracing
from graph.graph_tracer import GraphTracer

__all__ = [
    # Core classes
    'Node',
    'Edge',
    'Graph',
    'GraphBuilder',
    
    # Utility classes
    'GraphTraversal',
    'AngleUtils',
    'FeatureMatcher',
    'NodeManager',
    'GraphVisualizer',
    
    # Data handling
    'Record',
    'DataLoader',
    'VideoProcessor',
    
    # Tracing and visualization
    'GraphTracer',
    'GraphEvent',
    'GraphPlayback',
    'InteractiveGraphVisualizer',
    'visualize_graph_construction',
    
    # Type aliases
    'VisitRecord',
    'NeighborInfo',
    'Position',
    'EdgeFeature',
    'EdgeIndex',
    
    # Functions
    'get_future_action_labels',
    'get_roi',
    'build_graph'
]
