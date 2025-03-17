# Gaze-Guided Scene Graphs for Egocentric Action Prediction

A framework for building scene graphs from egocentric video using gaze data to predict future actions.

## Overview

This project builds scene graphs from egocentric video and gaze data to capture spatial and temporal relationships between objects, which can be used for action anticipation. The system:

1. Processes fixations and saccades to identify objects using CLIP
2. Constructs a scene graph with nodes (objects) and edges (transitions)
3. Extracts features at specified timestamps for downstream tasks

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

## Project Structure

- **datasets/**: Dataset processing scripts and files
- **egtea_gaze/**: Action and gaze annotations/processing
- **graph/**: Scene graph construction
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
```

**Key features**:
- Path references: `${path.to.reference}`
- Environment variables: `${USER}`, `${REPO_ROOT}`
- Configuration files: `student_cluster_config.yaml`, `euler_cluster_config.yaml`

To use a custom config:
```bash
python main.py --config path/to/config.yaml <command>
```

## Command-Line Interface

```bash
python main.py [options] <command>
```

**Commands**:
- `setup-scratch`: Download and setup dataset
- `build`: Build the scene graph dataset

**Global options**:
- `--config`: Custom config file path
- `--log-level`: Set logging level
- `--log-file`: Specify log file

For help:
```bash
python main.py --help
python main.py <command> --help
```
