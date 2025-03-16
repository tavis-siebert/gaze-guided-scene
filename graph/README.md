# Graph Module

This module provides functionality for building, manipulating, and visualizing scene graphs based on gaze data and object detection.

## Overview

The graph module constructs scene graphs from video data and gaze information. It uses models from the `models/` directory for object detection (CLIP) and feature extraction (SIFT). The resulting graphs can be used for various downstream tasks, such as action anticipation.

## Key Components

- **Graph**: Represents a scene graph with nodes and edges
- **Node**: Represents an object in the scene with feature management
- **GraphBuilder**: Builds scene graphs from video data and processes gaze fixations
- **NodeManager**: Utilities for managing nodes in the scene graph
- **AngleUtils**: Utilities for angle calculations and conversions
- **FeatureMatcher**: Utilities for matching features between images
- **GraphTraversal**: Utilities for traversing and exploring graph structures
- **GraphVisualizer**: Utilities for visualizing graph structures
- **VideoProcessor**: Handles video loading and frame extraction
- **DataLoader**: Handles loading and processing of dataset files

## Code Organization

The module is organized into several files:

- `build_graph.py`: Contains the `GraphBuilder` class for constructing scene graphs
- `graph.py`: Contains the `Graph` class for representing scene graphs
- `node.py`: Contains the `Node` and `NodeManager` classes for managing graph nodes
- `utils.py`: Contains utility functions and classes for graph operations
- `visualizer.py`: Contains the `GraphVisualizer` class for graph visualization
- `io.py`: Contains data loading and processing utilities

## Usage

See the project root README.md for usage examples. 