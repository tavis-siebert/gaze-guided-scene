"""Constants specific to the EGTEA Gaze+ dataset."""

# Video properties
FPS = 24  # Fixed frame rate of all videos
RESOLUTION = (640, 480)  # Fixed resolution of all videos
NUM_ACTION_CLASSES = 106  # Number of action classes from action_idx.txt

# Gaze data properties
GAZE_RESOLUTION = (1280, 960)  # Original resolution of gaze data before normalization

# Gaze types
GAZE_TYPE_UNTRACKED = 0  # No gaze point available
GAZE_TYPE_FIXATION = 1  # Pause of gaze
GAZE_TYPE_SACCADE = 2  # Jump of gaze
GAZE_TYPE_UNKNOWN = 3  # Unknown gaze type returned by BeGaze
GAZE_TYPE_TRUNCATED = 4  # Gaze out of range of the video 