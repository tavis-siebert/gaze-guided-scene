import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from config.config_utils import DotDict

# Clip-level utils
def get_action_clip_maps(config: DotDict) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Returns two maps:
        action_to_clips: maps action label to all clip names with that action
        clips_to_action: maps all clip names to their corresponding action
    """
    action_labels_path = Path(config.paths.egtea_dir) / "action_annotation/raw_annotations/action_labels.csv"
    
    with open(action_labels_path) as f:
        csv_reader = csv.reader(f, delimiter=';')
        next(csv_reader, None)  # skip header

        action_to_clips = defaultdict(list)  # action_label -> clip_name map
        clips_to_action = {}
        for row in csv_reader:
            clip_name, action_label = row[1].strip(), row[5].strip()
            action_to_clips[action_label].append(clip_name)
            clips_to_action[clip_name] = action_label
        
        return action_to_clips, clips_to_action

def get_all_clips_from_video(video_name: str, config: DotDict, sort_temporally: bool = False) -> List[Path]:
    """
    Given a video name (e.g. OP01-R01-PastaSalad) returns all clips cropped from the video.
    Raw videos can be found at https://www.dropbox.com/scl/fi/o7mrc7okncgoz14a49e5q/video_links.txt?rlkey=rcz1ffw4eoibod8mmyj1nmyot&e=1&dl=0 
    Cropped clips can be found at https://www.dropbox.com/scl/fi/97r0kjz65wb6xf0mjpcd0/video_clips.tar?rlkey=flcqqd91lyxtm6nlsh4vjzvkq&e=1&dl=0 

    Args:
        video_name: Name of the full video (e.g. OP01-R01-PastaSalad)
        config: Configuration dictionary
        sort_temporally: If True, sorts all clips temporally based on frame number in original video
    """
    clips_path = Path(config.paths.scratch_dir) / 'egtea_gaze/cropped_clips' / video_name
    all_clips = list(clips_path.glob('*.mp4'))
    
    if sort_temporally:
        all_clips.sort()
    return all_clips