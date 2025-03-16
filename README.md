## Gaze-Guided Scene Graphs for Egocentric Action Prediction

### Directory Structure

- **datasets/**: Contains scripts for processing datasets as well as dataset files.
- **egtea_gaze/**: Annotations and processing for actions and gaze
- **graph/**: Handles scene graph construction and related operations.
- **models/**: Includes model definitions for feature extraction (SIFT) and object detection (CLIP).

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

The project provides a command-line interface through `main.py`. All commands support using a custom configuration file (default: `config/student_cluster_config.yaml`):
```bash
python main.py --config path/to/config.yaml <command>
```

Available commands:
- `setup-scratch`: Setup scratch directories and download required files
- `build`: Build the dataset

For detailed usage of each command, run:
```bash
python main.py --help
python main.py <command> --help
```

Note: Make sure you have activated the virtual environment before running the commands (see [Setup the Python Environment](#setup-the-python-environment)).
