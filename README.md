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

4. **Build scene graphs**:
   ```bash
   python main.py build-graph
   ```
   
   To enable tracing for visualization:
   ```bash
   python main.py build-graph --videos VIDEO_NAME --enable-tracing
   ```

5. **Train a model**:
   ```bash
   python main.py train --task future_actions
   ```
   
   Or for next action prediction:
   ```bash
   python main.py train --task next_action
   ```

6. **Visualize graph construction** (requires prior trace generation):
   ```bash
   python main.py visualize --video-name VIDEO_NAME
   ```

## Project Structure

- **datasets/**: Dataset processing scripts and files
- **egtea_gaze/**: Action and gaze annotations/processing
- **graph/**: Scene graph construction and visualization
  - **Core Components**: Graph, Node, GraphBuilder, GraphTracer, GraphVisualizer
  - **dashboard/**: Interactive visualization dashboard
    - **components/**: UI components (VideoDisplay, GraphDisplay, PlaybackControls, MetaInfo)
    - **playback/**: Event handling and graph state management
    - **utils/**: Utility functions and constants
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

## TensorBoard Integration

The project includes TensorBoard integration for visualizing training metrics and model performance. The logs are stored under `logs/{task}/` by default.

```bash
# Run training with TensorBoard logging
python main.py train --task next_action --device gpu

# Run training with TensorBoard logging
python main.py train --task future_actions --device gpu

# Launch TensorBoard to view metrics
tensorboard --logdir logs
```


## Command-Line Interface

```bash
python main.py [options] <command>
```

**Commands**:
- `setup-scratch`: Download and setup dataset
- `build-graph`: Build scene graphs from videos
  - Options:
    - `--device {gpu|cpu}`: Device to use (default: gpu)
    - `--videos VIDEO_NAME [VIDEO_NAME ...]`: Specific videos to process
    - `--enable-tracing`: Enable graph construction tracing for visualization
- `train`: Train a GNN on a specified task
  - Options:
    - `--device {gpu|cpu}`: Device to use (default: gpu)
    - `--task {future_actions|next_action}`: Task to train on
- `visualize`: Visualize the graph construction process
  - Options:
    - `--video-name VIDEO_NAME`: Name of the video to visualize (used to locate trace file if trace-path not provided)
    - `--video-path PATH`: Path to the video file (optional when using --video-name, required with --trace-path)
    - `--trace-path PATH`: Path to the trace file (alternative to --video-name)
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

## Dataset Creation Workflow

The project follows a two-stage approach for dataset creation and model training:

1. **Graph Building Stage** (`build-graph` command):
   - Processes videos to extract objects from gaze fixations
   - Constructs scene graphs based on object transitions
   - Saves raw graph checkpoints for each video frame to `datasets/graphs/{split}/{video_name}_graph.pth`
   - Each checkpoint contains raw graph data (nodes, edges, adjacency information)

2. **Model Training Stage** (`train` command):  
   - Loads raw graph checkpoints from disk
   - Performs feature engineering during data loading
   - Creates batched data for training and validation
   - Applies optional augmentations (e.g., node dropping)

This separation provides several benefits:
- Feature engineering experiments without rebuilding scene graphs
- Different sampling strategies for different tasks
- Data augmentation at training time
- More efficient training iterations

Example dataset loading (see `datasets/example_dataloader.py`):
```python
# Create a data loader
dataloader = create_dataloader(
    root_dir="datasets/graphs",
    split="train",
    val_timestamps=config.training.val_timestamps,
    task_mode="future_actions",
    batch_size=64,
    node_drop_p=0.1,  # Optional augmentation
    max_droppable=2,
    shuffle=True
)

# Use in training
for batch in dataloader:
    # batch.x: Node features
    # batch.edge_index: Graph connectivity
    # batch.edge_attr: Edge features
    # batch.y: Labels
    ...
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
python main.py build-graph --videos VIDEO_NAME --enable-tracing

# For multiple videos (each gets its own trace file)
python main.py build-graph --videos VIDEO1 VIDEO2 VIDEO3 --enable-tracing
```

### Visualization Dashboard

The interactive dashboard displays the graph construction process:

```bash
# Using video name (video and trace files are located using config paths)
python main.py visualize --video-name VIDEO_NAME [--video-path PATH] [--port PORT]

# Using full paths
python main.py visualize --trace-path /path/to/trace_file.jsonl --video-path /path/to/video.mp4 [--port PORT]
```

**Dashboard Components**:
- **Dashboard** - Main component that integrates all visualization components
- **Playback** - Handles trace file loading and graph state management
- **VideoDisplay** - Manages video frames and overlay visualization
- **GraphDisplay** - Handles graph visualization and interaction
- **PlaybackControls** - Playback navigation controls
- **MetaInfo** - Displays information about the video and trace files

**Key Features**:
- **Interactive Graph Visualization**: Angle-based node positioning with stable layout, directional edges with symbolic notation, node highlight animations, gaze transition arrows, interactive hover details, timeline markers, and improved empty state visualization.

- **Video Playback**: Frame-by-frame playback with gaze overlays, fixation/saccade visualization, object detection highlighting, and synchronized graph display.

- **Intuitive Controls**: Play/pause with emoji buttons, real-time speed control, timeline slider with event markers, navigation buttons, and MM:SS time display.

- **Performance Optimizations**: Figure caching, optimized layout initialization, FIFO frame caching, precomputed SVG paths, edge hover thresholds, reduced framerate, static background rendering, and batch frame processing.