# Graph Module

A module for building and managing scene graphs from egocentric video and gaze data.

## Core Components

- **Graph** (`graph.py`): Scene graph representation with nodes and edges
- **Node** (`node.py`): Object representation with feature management
- **GraphBuilder** (`build_graph.py`): Constructs graphs from video and gaze data

## Key Features

- Processes gaze fixations and saccades to identify objects
- Builds scene graphs with objects as nodes and transitions as edges
- Extracts and normalizes features at specified timestamps
- Supports visualization and traversal of graph structures

## Implementation

The module is organized into:

```
build_graph.py  # Graph construction from video data
graph.py        # Graph data structure and operations
node.py         # Node representation and management
utils.py        # Utility functions for graph operations
visualizer.py   # Graph visualization tools
io.py           # Data loading and processing utilities
```

## Usage

Import the module components to build and manipulate scene graphs:

```python
from graph.build_graph import build_graph
from graph.graph import Graph
from graph.node import Node

# See main.py and build_dataset.py for usage examples 