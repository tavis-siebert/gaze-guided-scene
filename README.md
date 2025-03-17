## Gaze-Guided Scene Graphs for Egocentric Action Prediction

### Directory Structure

- **datasets/**: Contains scripts for processing datasets as well as dataset files.
- **egtea_gaze/**: Annotations and processing for actions and gaze
- **graph/**: Handles scene graph construction and related operations.
- **models/**: Includes model definitions for feature extraction (SIFT) and object detection (CLIP).
- **logger.py**: Centralized logging configuration for the entire project.
- **config/**: Configuration files and utilities for path management.

### Scene Graph Construction

The project builds scene graphs from video data and gaze information. The graph construction process follows these steps:

1. Load object labels and dataset information
2. Initialize CLIP model for object detection
3. For each video:
   - Load gaze data and video frames
   - Process fixations and saccades to identify objects
   - Build a scene graph by creating nodes for objects and edges for transitions
   - Extract and normalize features at specified timestamps
   - Save graph state for downstream tasks

The resulting graph structure captures the spatial and temporal relationships between objects in the scene, which can be used for tasks like action anticipation.

### Path Configuration System

The project uses a centralized path configuration system to manage all file and directory paths:

- **Configuration Files**: All paths are defined in YAML configuration files in the `config/` directory
- **Path Resolution**: Paths can reference other paths using `${path.to.reference}` syntax
- **Environment Variables**: Paths can include environment variables like `${USER}` or `${REPO_ROOT}`
- **Hierarchical Structure**: Configuration is organized in a logical hierarchy for better readability

The configuration is organized into the following main sections:

- **base**: Root directories for the project
  - `scratch_dir`: Base scratch directory
  - `repo_root`: Repository root directory

- **directories**: Logical grouping of directory structures
  - `repo`: Repository directories
  - `scratch`: Scratch directories

- **dataset**: Dataset-related paths and settings
  - `timestamps`: Timestamp ratios for feature extraction
  - `egtea`: EGTEA Gaze+ dataset paths
  - `ego_topo`: Ego-topo data paths
  - `output`: Output paths for generated datasets

- **models**: Model settings and paths
  - `clip`: CLIP model paths and settings

- **external**: External resources
  - `urls`: URLs for downloading datasets and models

- **processing**: Processing settings
  - `n_cores`: Number of CPU cores to use for processing

This hierarchical structure makes the configuration more readable and easier to maintain. The main configuration files are:

- `student_cluster_config.yaml`: Configuration for the student cluster environment
- `euler_cluster_config.yaml`: Configuration for the Euler cluster environment

To use a custom configuration file, specify it with the `--config` argument:
```bash
python main.py --config path/to/custom_config.yaml [command]
```

**Important**: All path configurations and settings should be done in the configuration files rather than through command-line arguments. This ensures that all dependent paths are properly updated and maintains consistency throughout the codebase.

### Logging System

The project uses a centralized logging system that provides consistent logging across all modules:

- **Configuration**: Logging is configured in `logger.py` and initialized in `main.py`
- **Log Levels**: Supports standard Python logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **Command Line Control**: Log level and output file can be specified via command line arguments:
  ```bash
  python main.py --log-level DEBUG --log-file logs/debug.log [other arguments]
  ```
- **Environment Variables**: Log level can also be set using the `LOG_LEVEL` environment variable

Each module gets its own logger instance with the module name, allowing for fine-grained log filtering and organization.

### Setup

Clone this repository and navigate to its directory

#### Setup the Python Environment

1. Run the setup script:
   ```bash
   source scripts/setup_env.sh
   ```
   This script will:
   - Create a virtual environment called `venv`
   - Install all dependencies from `requirements.txt`
   - Activate the virtual environment

   You may rerun the script to update dependencies.

Note: Always use `source scripts/setup_env.sh` rather than `sh scripts/setup_env.sh` or `./scripts/setup_env.sh` to ensure the virtual environment is activated in your current shell.

#### Create a Dropbox Access Token

To download the Egtea Gaze dataset and allow automatic dataset download to the scratch directory, you need a Dropbox access token.

1. Make sure you are logged in to Dropbox in your browser.

2. Go to [Dropbox App Console](https://www.dropbox.com/developers/apps/)

3. Create a new app (any app type will work)

4. In the app's permissions page, enable the `sharing.read` permission

5. Generate an OAuth 2.0 access token

6. Copy the `.env.example` file to `.env` and add your access token:
    ```bash
    cp .env.example .env
    ```

7. Update the `DROPBOX_TOKEN` variable in the `.env` file with your access token

Note: You may need to regenerate the access token (steps 5 and 7) if it expires.

#### Setup the Scratch Directory and Download the Egtea Gaze Dataset

The script will download both the raw videos and the cropped video clips and place them in the scratch directory.

1. Run the setup script:
    ```bash
    python main.py setup-scratch
    ```

### Usage

The project provides a command-line interface through `main.py`. The interface is designed to be simple and focused on the main tasks:

```bash
python main.py --config path/to/config.yaml <command>
```

Available commands:
- `setup-scratch`: Setup scratch directories and download required files
  - Options: `--dropbox-token` to provide a Dropbox access token
- `build`: Build the dataset
  - Options: `--debug` to process only one video per split

Global options:
- `--config`: Path to a custom configuration file (default: `config/student_cluster_config.yaml`)
- `--log-level`: Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `--log-file`: Path to log file (if not specified, logs to console only)

For detailed usage of each command, run:
```bash
python main.py --help
python main.py <command> --help
```

**Configuration-based approach**: All path configurations and settings should be modified in the YAML configuration files rather than through command-line arguments. This ensures that all dependent paths are properly updated and maintains consistency throughout the codebase.

Note: Make sure you have activated the virtual environment before running the commands (see [Setup the Python Environment](#setup-the-python-environment)).
