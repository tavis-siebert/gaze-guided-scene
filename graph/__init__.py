"""
Graph module for scene graph construction and manipulation.

This module provides classes and utilities for building, manipulating,
and visualizing scene graphs based on gaze data and object detection.
"""

from graph.node import Node, VisitRecord, NeighborInfo
from graph.graph import Graph
from graph.utils import (
    GraphTraversal,
    AngleUtils,
    FeatureMatcher,
    NodeManager,
    EdgeManager,
    GraphVisualizer,
    get_all_nodes,
    get_angle_bin,
    update_graph,
    print_levels
)

__all__ = [
    # Classes
    'Node',
    'Graph',
    'GraphTraversal',
    'AngleUtils',
    'FeatureMatcher',
    'NodeManager',
    'EdgeManager',
    'GraphVisualizer',
    
    # Type aliases
    'VisitRecord',
    'NeighborInfo',
    
    # Functions
    'get_all_nodes',
    'get_angle_bin',
    'update_graph',
    'print_levels'
]
