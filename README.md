## Gaze-Guided Scene Graphs for Egocentric Action Prediction
### Directory Structure

- **datasets/**: Contains scripts for processing datasets as well as dataset files.
- **egtea_gaze/**: Annotations and processing for actions and gaze
- **graph/**: Handles scene graph construction and related operations.
- **models/**: Includes model definitions (TODO).

### Setup
1. Clone this repository and navigate to its directory

#### Python Environment
2. Run the setup script:
   ```bash
   source setup_env.sh
   ```
   This script will:
   - Create a virtual environment called `venv`
   - Install all dependencies from `requirements.txt`
   - Activate the virtual environment

Note: Always use `source setup_env.sh` rather than `sh setup_env.sh` or `./setup_env.sh` to ensure the virtual environment is activated in your current shell.