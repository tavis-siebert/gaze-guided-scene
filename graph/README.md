# Graph Module

This module provides functionality for building, manipulating, and visualizing scene graphs based on gaze data and object detection.

## Overview

The graph module constructs scene graphs from video data and gaze information. It uses models from the `models/` directory for object detection (CLIP) and feature extraction (SIFT). The resulting graphs can be used for various downstream tasks, such as action anticipation.

## Key Components

- **Graph**: Represents a scene graph with nodes and edges
- **Node**: Represents an object in the scene with feature management
- **GraphBuilder**: Builds scene graphs from video data and processes gaze fixations

## Usage

See the project root README.md for usage examples. 