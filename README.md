# Gaze-Guided Scene Graphs for Egocentric Action Prediction

A framework for building scene graphs from egocentric video using gaze data to predict future actions.

## Overview

This project builds scene graphs from egocentric video and gaze data to capture spatial and temporal relationships between objects, which can be used for action anticipation. The system:

1. Processes fixations and saccades to identify objects using CLIP
2. Constructs a scene graph with nodes (objects) and edges (transitions)
3. Extracts features at specified timestamps for downstream tasks
4. Provides interactive visualization of the graph construction process

## Testing

### Running Tests Locally

To run tests locally (using mock models):

```bash
# Run all tests, automatically skipping those requiring real models
pytest tests/

# Run only unit tests
pytest tests/unit/

# Force run tests requiring real models
pytest --run-real-model tests/
```

### Running Tests on Cluster

For more resource-intensive tests, you can submit a job to the cluster:

```bash
# Submit test job to cluster
sbatch src/gazegraph/scripts/run_tests.sh
```

Test logs will be written to `logs/tests.out`.

## Project Structure

```
src/gazegraph/           # Main package code
├── config/              # Configuration files and utilities
├── datasets/            # Dataset loaders and processors
│   └── egtea_gaze/      # EGTEA Gaze+ dataset specific code
├── graph/               # Scene graph construction and processing
├── models/              # Neural network models
├── scripts/             # Utility scripts
├── training/            # Training infrastructure
├── logger.py            # Logging utilities
└── main.py              # Main entry point

data/                    # Data storage
├── egtea_gaze/          # EGTEA Gaze+ dataset
│   ├── action_annotation/
│   └── gaze_data/
├── graphs/              # Generated scene graphs
└── traces/              # Execution traces for visualization

logs/                    # Training and execution logs
```

## Setup

### Prerequisites

- Python 3.12+
- Access to a Dropbox token for downloading the EGTEA Gaze+ dataset
- [uv package manager](https://astral.sh/uv) (setup scripts will install if not present)

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

   These scripts will:
   - Check if uv is installed, and install it if necessary
   - Ensure the proper Python version is available
   - Synchronize project dependencies using the lockfile (`uv sync`)

2. **Create a Dropbox token**:
   - Create an app at [Dropbox App Console](https://www.dropbox.com/developers/apps/)
   - Enable `sharing.read` permission
   - Generate an OAuth 2.0 token
   - Create a `.env` file in the project root and add the following: 
     ```
     # Dropbox token for dataset downloads
     DROPBOX_TOKEN=your_token_here
     
     # Optional: Custom config path
     CONFIG_PATH=path/to/your/config.yaml
     ```
   
   > **Note:** Dropbox tokens expire after a period of time. If you encounter authentication errors, you'll need to generate a new token following the steps above.

3. **Download dataset**:
   ```bash
   ./run.sh setup-scratch
   ```

4. **Build scene graphs**:
   ```bash
   ./run.sh build-graphs
   ```
   
   To enable tracing for visualization:
   ```bash
   ./run.sh build-graphs --videos VIDEO_NAME --enable-tracing
   ```

5. **Train a model**:
   ```bash
   ./run.sh train --task future_actions
   ```
   
   Or for next action prediction:
   ```bash
   ./run.sh train --task next_action
   ```

6. **Visualize graph construction** (requires prior trace generation):
   ```bash
   ./run.sh visualize --video-name VIDEO_NAME
   ```

## Component Descriptions

- **graph/**: Scene graph construction and visualization
  - **Core Components**: 
    - **Graph, Node**: Core data structures
    - **GraphBuilder**: Processes a single video to build a scene graph
    - **GraphProcessor**: Handles multi-video processing with parallel execution
    - **GraphTracer**: Records trace data during graph construction
    - **GraphVisualizer**: Visualizes the graph construction process
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
- Environment variable `CONFIG_PATH` to specify configuration file path

## Environment Variables

The project supports configuration via environment variables in a `.env` file in the project root:

1. **Create a `.env` file**:
   ```bash
   cp .env.example .env
   ```

2. **Available environment variables**:
   ```
   # Configuration file path
   CONFIG_PATH=path/to/your/config.yaml
   
   # Dropbox token for dataset downloads
   DROPBOX_TOKEN=your_token_here
   ```

Environment variables take precedence over defaults in the code. When specified, `CONFIG_PATH` 
determines which configuration file is loaded by default, but can still be overridden with the
`--config` command-line argument.

## TensorBoard Integration

The project includes TensorBoard integration for visualizing training metrics and model performance. The logs are stored under `logs/{task}/` by default.

```bash
# Run training with TensorBoard logging
./run.sh train --task next_action --device gpu

# Run training with TensorBoard logging
./run.sh train --task future_actions --device gpu

# Launch TensorBoard to view metrics
tensorboard --logdir logs
```

## Command-Line Interface

```bash
./run.sh [options] <command>
```

**Commands**:
- `setup-scratch`: Download and setup dataset
- `build-graphs`: Build scene graphs from videos
- `train`: Train a GNN on a specified task
- `visualize`: Visualize graph construction process

**Global Options**:
- `--config`: Path to custom config file
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `--log-file`: Path to log file
