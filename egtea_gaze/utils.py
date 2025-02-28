import csv
from pathlib import Path
from collections import defaultdict

#TODO move from hardcoded paths to pathlib or something. 
# each person will have to edit this according to their directories though, so maybe not worth it
SCRATCH = '/cluster/scratch/tsiebert'
EGTEA_DIR = '/cluster/home/tsiebert/egocentric_vision/egtea_gaze'
resolution = (640,480)
fps = 24

# Clip-level utils
def get_action_clip_maps():
    """
    Returns two maps
        action_to_clips: maps action label to all clip names with that action
        clips_to_action: maps all clip names to their corresponding action
    """
    with open(EGTEA_DIR + "/action_annotation/raw_annotations/action_labels.csv") as f:
        csv_reader = csv.reader(f, delimiter=';')
        next(csv_reader, None)  # skip header

        action_to_clips = defaultdict(list)  # action_label -> clip_name map
        clips_to_action = {}
        for row in csv_reader:
            clip_name, action_label = row[1].strip(), row[5].strip()
            action_to_clips[action_label].append(clip_name)
            clips_to_action[clip_name] = action_label
        
        return action_to_clips, clips_to_action

def get_all_clips_from_video(video_name, sort_temporally=False):
    """
    Given a video name (e.g. OP01-R01-PastaSalad)) returns all clips cropped from the video
    Raw videos can be found at https://www.dropbox.com/scl/fi/o7mrc7okncgoz14a49e5q/video_links.txt?rlkey=rcz1ffw4eoibod8mmyj1nmyot&e=1&dl=0 
    Crooped clips can be found at https://www.dropbox.com/scl/fi/97r0kjz65wb6xf0mjpcd0/video_clips.tar?rlkey=flcqqd91lyxtm6nlsh4vjzvkq&e=1&dl=0 

    Args:
        video_name (str): name of the full video (e.g. OP01-R01-PastaSalad)
        sort_temporally (bool): if True, sorts all clips temporally based on frame number in original video. Helpful for tasks like predicting the next action from a series of action clips
    """
    clips_dir = Path(SCRATCH + '/egtea_gaze/cropped_clips/' + video_name).glob('*.mp4')
    all_clips = [clip for clip in clips_dir]
    if sort_temporally:
        all_clips.sort()
    return all_clips