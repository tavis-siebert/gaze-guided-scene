# Gaze-Guided Scene Graphs for Egocentric Action Prediction

A framework for building scene graphs from egocentric video using gaze data to predict future actions.

## Overview

This project builds scene graphs from egocentric video and gaze data to capture spatial and temporal relationships between objects, which can be used for action anticipation. The system:

1. Processes fixations and saccades to identify objects using CLIP
2. Constructs a scene graph with nodes (objects) and edges (transitions)
3. Extracts features at specified timestamps for downstream tasks
4. Provides interactive visualization of the graph construction process

## Setup

### Prerequisites

- Python 3.8+
- Access to a Dropbox token for downloading the EGTEA Gaze+ dataset

### Quick Start

1. **Setup environment**:
   Choose the appropriate setup script based on your environment:
   
   For Student Cluster:
   ```bash
   source scripts/setup_student_cluster_env.sh
   ```
   
   For Euler Cluster:
   ```bash
   source scripts/setup_euler_cluster_env.sh
   ```

2. **Create a Dropbox token**:
   - Create an app at [Dropbox App Console](https://www.dropbox.com/developers/apps/)
   - Enable `sharing.read` permission
   - Generate an OAuth 2.0 token
   - Add to `.env` file: `DROPBOX_TOKEN=your_token_here`
   
   > **Note:** Dropbox tokens expire after a period of time. If you encounter authentication errors, you'll need to generate a new token following the steps above.

3. **Download dataset**:
   ```bash
   python main.py setup-scratch
   ```

4. **Build the dataset**:
   ```bash
   python main.py build
   ```
   
   To enable tracing for visualization:
   ```bash
   python main.py build --videos VIDEO_NAME --enable-tracing
   ```

5. **Visualize graph construction** (requires prior trace generation):
   ```bash
   python main.py visualize --video-name VIDEO_NAME
   ```

## Project Structure

- **datasets/**: Dataset processing scripts and files
- **egtea_gaze/**: Action and gaze annotations/processing
- **graph/**: Scene graph construction and visualization
  - **Core Components**: Graph, Node, GraphBuilder, GraphTracer, GraphVisualizer
  - **Key Features**: Processes gaze data, builds scene graphs, extracts features, visualizes construction
- **models/**: Feature extraction (SIFT) and object detection (CLIP)
- **config/**: Configuration files and utilities
- **logger.py**: Centralized logging

## Configuration System

The project uses YAML configuration files with a hierarchical structure:

```
base:                # Root directories
directories:         # Directory structure
dataset:             # Dataset settings and paths
  timestamps:        # Feature extraction timestamps
  egtea:             # EGTEA dataset paths
  ego_topo:          # Ego-topo data paths
  output:            # Output paths
models:              # Model settings
external:            # External resources
processing:          # Processing settings
trace_dir:           # Directory for graph construction trace files
```

**Key features**:
- Path references: `${path.to.reference}`
- Environment variables: `${USER}`, `${REPO_ROOT}`
- Configuration files: `student_cluster_config.yaml`, `euler_cluster_config.yaml`

## Command-Line Interface

```bash
python main.py [options] <command>
```

**Commands**:
- `setup-scratch`: Download and setup dataset
- `build`: Build the scene graph dataset
  - Options:
    - `--device {gpu|cpu}`: Device to use (default: gpu)
    - `--videos VIDEO_NAME [VIDEO_NAME ...]`: Specific videos to process
    - `--enable-tracing`: Enable graph construction tracing for visualization
- `visualize`: Visualize the graph construction process
  - Options:
    - `--video-name VIDEO_NAME`: Name of the video to visualize (required)
    - `--video-path PATH`: Path to the video file (optional)
    - `--port PORT`: Server port (default: 8050)
    - `--debug`: Run in debug mode

**Global options**:
- `--config`: Custom config file path (default: config/student_cluster_config.yaml)
- `--log-level`: Set logging level
- `--log-file`: Specify log file

For help:
```bash
python main.py --help
python main.py <command> --help
```

## Graph Tracing and Visualization

The project includes a tracing and visualization system for recording and analyzing the graph construction process.

### Tracing System

Trace files are stored in the `traces` directory (configurable via `trace_dir` in config):
- Each video gets its own trace file named `{video_name}_trace.jsonl`
- Each file contains events like fixations, saccades, node/edge creation
- Rerunning a trace for the same video overwrites its previous trace file

To generate traces for visualization:
```bash
# For a single video
python main.py build --videos VIDEO_NAME --enable-tracing

# For multiple videos (each gets its own trace file)
python main.py build --videos VIDEO1 VIDEO2 VIDEO3 --enable-tracing
```

### Visualization Dashboard

The interactive dashboard displays the graph construction process:

```bash
python main.py visualize --video-name VIDEO_NAME [--video-path PATH] [--port PORT]
```

**Dashboard Components**:
- **Dashboard** - Main component that integrates all others
- **Playback** - Handles trace file loading and graph state management
- **VideoDisplay** - Manages video frames and overlay visualization
- **GraphDisplay** - Handles graph visualization and interaction
- **PlaybackControls** - Playback navigation controls

**Key Features**:
- Video player with gaze position and region-of-interest overlays
- Graph visualization showing nodes (objects) and their relationships
- Interactive playback controls for navigating through frames

**Direct API Usage**:
```python
from graph.visualizer import visualize_graph_construction

visualize_graph_construction("path/to/trace_file.jsonl", "path/to/video.mp4")
```
