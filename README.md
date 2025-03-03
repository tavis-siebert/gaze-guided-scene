## Gaze-Guided Scene Graphs for Egocentric Action Prediction
### Directory Structure

- **datasets/**: Contains scripts for processing datasets as well as dataset files.
- **egtea_gaze/**: Annotations and processing for actions and gaze
- **graph/**: Handles scene graph construction and related operations.
- **models/**: Includes model definitions (TODO).

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

To download the Egtea Gaze dataset and allow the `setup_scratch.py` script to automatically download the dataset directly to the scratch directory, you need a Dropbox access token.

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
    python main.py setup-egtea-scratch
    ```