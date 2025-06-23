# Gaze-Guided Scene Graphs for Egocentric Action Prediction

A framework for building scene graphs from egocentric video using gaze data to predict future actions.

## Overview

This project builds scene graphs from egocentric video and gaze data to capture spatial and temporal relationships between objects, which can be used for action anticipation. The system:

1. Processes fixations and saccades to identify objects using CLIP
2. Constructs a scene graph with nodes (objects) and edges (transitions)
3. Extracts features at specified timestamps for downstream tasks
4. Provides interactive visualization of the graph construction process

Note: Refer to "Quick Start" section for instructions on how to visualize the graph construction process using the provided sample data.

## Project Structure

```
src/gazegraph/               # Main package code
├── config/                  # YAML configuration files and utilities
├── datasets/                # Dataset loaders and processors
│   └── egtea_gaze/          # EGTEA Gaze+ dataset specific code
├── graph/                   # Scene graph construction and processing
│   ├── dashboard/           # Interactive visualization dashboard
│   │   ├── components/      # UI components (video, graph display, controls)
│   │   ├── playback/        # Event handling and state management
│   │   └── utils/           # Dashboard utilities and SVG generation
│   └── *.py                 # Core graph processing (builder, tracer, visualizer)
├── models/                  # Feature extraction and object detection (CLIP, SIFT, YOLO-World)
├── training/                # Training infrastructure
│   ├── dataset/             # Graph datasets, dataloaders, and augmentations
│   ├── evaluation/          # Metrics and evaluation utilities
│   └── tasks/               # Task definitions (next_action, future_actions)
├── logger.py, main.py       # Logging and main entry point
└── setup_scratch.py         # Dataset setup utilities

datasets/sample_data/         # Sample data for graph visualization
figures/                     # Documentation and visualization outputs
scripts/                     # Utility and build scripts  
tests/                       # Test suite with component-specific tests and fixtures
```

## Setup

### Prerequisites

- Python 3.12+
- Access to a Dropbox token for downloading the EGTEA Gaze+ dataset
- [uv package manager](https://astral.sh/uv) (setup scripts will install if not present)

### Quick Start

> **Note:** The graph visualization only requires the raw video file and corresponding trace file to be present, which we provide in `datasets/sample_data/`. All other steps, except for the environment setup, are optional.


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
   scripts/run.sh setup-scratch
   ```

4. **Build scene graphs**:
   ```bash
   scripts/run.sh build-graphs
   ```
   
   To enable tracing for visualization:
   ```bash
   scripts/run.sh build-graphs --videos VIDEO_NAME --enable-tracing
   ```

5. **Train a model**:
   ```bash
   scripts/run.sh train --task future_actions
   ```
   
   Or for next action prediction:
   ```bash
   scripts/run.sh train --task next_action
   ```

6. **Visualize graph construction**

   To visualize the graph construction process e.g. using the sample data:
   ```bash
   scripts/run.sh visualize --video-path datasets/sample_data/OP03-R02-TurkeySandwich.mp4 --trace-path datasets/sample_data/OP03-R02-TurkeySandwich_trace.jsonl
   ```

   The interactive dashboard should now be available at http://127.0.0.1:8050/

### Running Gaze-Augmented Ego-Topo Experiments

This project can generate gaze-augmented features that can be used to enhance the performance of the [Ego-Topo](https://github.com/facebookresearch/ego-topo) model. The workflow involves generating fixation labels and ROI features using this repository, composing them with the base features provided by Ego-Topo, and then using the resulting feature file within a cloned Ego-Topo repository.

#### Prerequisites

- You must have our fork of the Ego-Topo repository cloned and set up according to its official instructions. You can clone it via:
  ```bash
  git clone https://github.com/jankulik/ego-topo-gaze.git
  ```
- Ensure you have downloaded the base features from the Ego-Topo project (e.g., `train_lfb_s30_verb.pkl` and `val_lfb_s30_verb.pkl`).

#### Step-by-Step Workflow

**Step 1: Setup This Repository**

If you haven't already, set up the `gaze-guided-scene-graph` environment and download the necessary datasets by following the "Quick Start" instructions.

**Step 2: Generate Fixation Labels**

The first step is to process the EGTEA Gaze+ videos to determine the most likely fixated object for each frame. This is done using the `label-fixations` command.

This command will produce two files:
- `*_label_map.pkl`: A map from each video frame to a detected object label.
- `*_roi_map.pkl`: (Optional) A map from each video frame to a CLIP embedding of the object's Region of Interest (ROI).

Run the command for both the training and validation splits. The `--skip-roi` flag can be used to speed up the process if you only need object labels initially.

```bash
# For the training split
scripts/run.sh label-fixations --in-pkl /path/to/ego-topo/features/train_lfb_s30_verb.pkl

# For the validation split
scripts/run.sh label-fixations --in-pkl /path/to/ego-topo/features/val_lfb_s30_verb.pkl
```

> **Note:** The output files will be saved in the directory specified by `directories.features` in your `config.yaml`, which defaults to `data/features/`.

**Step 3: Compose Gaze-Augmented Features**

Next, combine the base Ego-Topo features with the fixation labels (and optionally ROI embeddings) you just generated. Use the `compose-features` command for this.

There are several composition `modes` available. For gaze-augmented Ego-Topo, you might use:
- `onehot`: Concatenates the base feature with a one-hot vector of the fixated object label.
- `clip`: Concatenates the base feature with a CLIP text embedding of the fixated object label.
- `roi`: Concatenates the base feature with a CLIP image embedding of the fixated object's ROI.

```bash
# Example: Compose features for the training split using one-hot encoding
scripts/run.sh compose-features \
  --mode onehot \
  --base-pkl /path/to/ego-topo/features/train_lfb_s30_verb.pkl \
  --fix-pkl data/features/train_lfb_s30_verb_label_map.pkl \
  --out-pkl data/features/composed_train_onehot.pkl

# Example: Compose features for the validation split using one-hot encoding
scripts/run.sh compose-features \
  --mode onehot \
  --base-pkl /path/to/ego-topo/features/val_lfb_s30_verb.pkl \
  --fix-pkl data/features/val_lfb_s30_verb_label_map.pkl \
  --out-pkl data/features/composed_val_onehot.pkl
```

Repeat this step for each composition mode you wish to evaluate.

**Step 4: Train with Augmented Features in the Ego-Topo Repository**

Finally, use the newly created composed feature files (`composed_*.pkl`) to train the model within your cloned Ego-Topo repository.

You will need to modify the training script in the Ego-Topo repository to point to these new feature files. For example, you might change its `main.py` or configuration to load `composed_train_onehot.pkl` instead of the original `train_lfb_s30_verb.pkl`.

Please refer to the [Ego-Topo repository's documentation](https://github.com/facebookresearch/ego-topo) for specific instructions on how to run training with custom feature files.

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
sbatch scripts/run_tests.sh
```

Test logs will be written to `logs/tests.out`.

## TensorBoard Integration

The project includes TensorBoard integration for visualizing training metrics and model performance. The logs are stored under `logs/{task}/` by default.

```bash
# Run training with TensorBoard logging
scripts/run.sh train --task next_action --device gpu

# Run training with TensorBoard logging
scripts/run.sh train --task future_actions --device gpu

# Launch TensorBoard to view metrics
tensorboard --logdir logs
```

## Command-Line Interface

```bash
scripts/run.sh [options] <command>
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
