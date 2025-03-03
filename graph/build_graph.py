import os
import cv2
import torch
import torchvision as tv
from collections import defaultdict, Counter
from transformers import CLIPProcessor, CLIPModel

from graph.node import Node
from graph.utils import update_graph, print_levels  #TODO rename config
from egtea_gaze.utils import SCRATCH, EGTEA_DIR
from egtea_gaze.gaze_data.gaze_io_sample import parse_gtea_gaze

### Helpers ###
class Record:
    """
    Copied over from egotpo repo: https://github.com/facebookresearch/ego-topo/blob/main/anticipation/anticipation/datasets/epic_utils.py/
    for reproducibility purposes.
    All credit to original authors
    """
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def start_frame(self):
        return int(self._data[1])

    @property
    def end_frame(self):
        return int(self._data[2])

    @property
    def label(self):
        return [int(x) for x in self._data[3:]]

    @property
    def num_frames(self):
        return self.end_frame - self.start_frame + 1


def get_roi(image, roi_center, roi_size):
    #TODO add support for CHW and HWC
    _, H, W, = image.shape
    x, y = roi_center

    # Define the ROI bounds (Region Of Interest)
    roi_half = roi_size // 2
    roi_y1 = max(0, y - roi_half)
    roi_y2 = min(H, y + roi_half)
    roi_x1 = max(0, x - roi_half)
    roi_x2 = min(W, x + roi_half)

    # Extract the ROI
    bbox = ((roi_x1, roi_y1), (roi_x2, roi_y2))
    roi = image[:, roi_y1:roi_y2, roi_x1:roi_x2]

    return roi, bbox

def run_CLIP(model, processor, frames, CLIP_labels, obj_labels, device):
    """
    Computes one forward pass of CLIP on a (batch of) images
    """
    input = processor(
        text=CLIP_labels,
        images=frames,
        return_tensors='pt',
        padding=True
    ).to(device)

    output = model(**input)
    probs = output.logits_per_image.softmax(dim=1)
    label = obj_labels[
        probs.argmax(dim=1).item()
    ]
    return label

def get_future_action_labels(records: list[Record], t: int, action_to_class: dict[tuple[int, int], int]):
    """
    Args:
        records (list[Record]): the action clips we pull labels from
        t (int): the frame after which we look for future action labels
        action_to_class (dict[tuple[int], int]): a map from action (verb, noun) -> class number
    Returns:
        future_action_labels (Tensor): a multiclass binary target vector where the target is 1 if this action is observed in the future
    """
    num_action_classes = len(action_to_class)

    past_records = [record for record in records if record.end_frame <= t]
    future_records = [record for record in records if record.start_frame > t]
    if len(past_records)< 3 or len(future_records) < 3:
        return torch.tensor([])
    
    observed_future_actions = set([
        action_to_class[(record.label[0], record.label[1])]
        for record in future_records if (record.label[0], record.label[1]) in action_to_class
    ])

    future_action_labels = torch.zeros(1, num_action_classes)
    future_action_labels[0, list(observed_future_actions)] = 1
    return future_action_labels


### MAIN ###
def build_graph(video_list, ann_file, timestamp_ratios, num_action_classes=106, print_graph=False):
    # Initialize CLIP model
    # object labels are pulled from noun_idx.txt, maps class number to string
    obj_labels, labels_to_int = {}, {}
    with open(EGTEA_DIR + '/action_annotation/noun_idx.txt') as f:
        for line in f:
            line = line.split(' ')
            class_idx, label = int(line[1]) - 1, line[0]
            obj_labels[class_idx] = label
            labels_to_int[label] = class_idx
        # obj_labels[label + 1] = 'kitchen_background'
    CLIP_labels = [f"a picture of a {obj}" for obj in obj_labels.values()]

    model_id = "openai/clip-vit-base-patch16"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Collect data
    vid_lengths = open(ann_file.replace('.csv', '_nframes.csv')).read().strip().split('\n')
    vid_lengths = [line.split('\t') for line in vid_lengths]
    vid_lengths = {k:int(v) for k,v in vid_lengths}

    records = [Record(x.strip().split('\t')) for x in open(ann_file)]
    records_by_vid = defaultdict(list)
    for record in records:
        records_by_vid[record.path].append(record)

    #NOTE I think the annotation files are such that one will get the same 106 actions for train and test, but the way the Egotopo people wrote it is bug-prone
    int_counts = [(record.label[0], record.label[1]) for record in records]
    int_counts = Counter(int_counts).items()
    int_counts = sorted(int_counts, key=lambda x: -x[1])[:num_action_classes]    # top actions
    int_to_idx = {interact:idx for idx, (interact, count) in enumerate(int_counts)}

    all_node_data = []
    all_edge_data = []
    all_edge_indices = []
    all_labels = []

    for video_name in video_list:
        records_for_vid = sorted(records_by_vid[video_name], key=lambda record: record.end_frame)
        vid_length = vid_lengths[video_name]

        timestamps = [int(frac*vid_length) for frac in sorted(timestamp_ratios)]
       
        gaze = parse_gtea_gaze(EGTEA_DIR + '/gaze_data/gaze_data/' + video_name + '.txt')
        
        # Build graph
        # ===========
        # for every frame
        # if gaze is 1 (fixation), run CLIP, increase counter, append keypoints
        # if gaze is 2 (saccade), create new Node, link, reset stuff
        #   linking = check duplicates (BFS + filter by label & keypoints) & label edge via gaze shift
        stream = tv.io.VideoReader(SCRATCH + '/egtea_gaze/raw_videos/' + video_name + '.mp4', 'video')

        SIFT = cv2.SIFT_create()    #TODO other options that one can toggle

        prev_x, prev_y = -1,-1  # needed to calculate how the position shifted
        potential_labels = defaultdict(int)  # counts of labels returned by CLIP (we take highest count per period of fixation)
        kps, descs = [], []  # keypoints, descriptors from SIFT
        visit = []  # the first and last frame gaze was fixated on a given object
        frame_num, relative_frame_num = 0, 0  # first is wrt actual video, second is for graph purposes / avoiding black frames

        num_nodes = [0]  # aka the graph size. Allows us to properly index new nodes in the adjacency list
        node_data = {}  # used to update node features in real time instead of searching through graph each timestamp: keys = Node.id, values = feature tensor
        edge_data = []  # used to update edge features in real time
        edge_index = [[],[]] # essentially PyG compatible version of an adjacency list

        # Root is used to clean up the loop. 
        # To handle it in downstream tasks (e.g. BFS), filter via label == 'root'
        root = curr_node = Node(id=-1, object_label='root')
        while True:
            curr_frame = next(stream, 'EOS') 

            if curr_frame == 'EOS':
                # it's a little ugly to write the loop this way instead of checking EOS in the while condition, 
                # but we have to handle the case of fixation ending the video
                if potential_labels:
                    visit.append(relative_frame_num - 1)
                    curr_node = update_graph(
                        curr_node,
                        potential_labels,
                        visit, 
                        kps,
                        descs,
                        (prev_x, prev_y),
                        (x, y),
                        edge_data,
                        edge_index,
                        num_nodes
                    )

                break

            # Save graph states dynamically at each timestamp if a graph exists
            if (frame_num in timestamps or frame_num >= len(gaze)) and edge_data:
                action_labels_t = get_future_action_labels(records_for_vid, frame_num, int_to_idx)
                if action_labels_t.numel() == 0:   # insufficient data
                    continue 

                node_data_t = torch.stack(list(node_data.values()))
                # normalize 1st, 2nd column and update 5th column
                node_data_t[:,0] /= relative_frame_num
                node_data_t[:,1] /= node_data_t[:,1].max()
                node_data_t[:,4] = timestamp_ratios[timestamps.index(frame_num)] if frame_num < len(gaze) else frame_num / vid_length

                edge_data_t = torch.stack(edge_data)
                edge_index_t = torch.tensor(edge_index, dtype=torch.long)
    
                all_node_data.append(node_data_t)
                all_labels.append(action_labels_t)
                all_edge_data.append(edge_data_t)
                all_edge_indices.append(edge_index_t)

                if frame_num == timestamps[-1] or frame_num >= len(gaze): # save ourselves compute
                    break
                
            # EGTEA can have starting + ending black frames, so skip these
            frame = curr_frame['data']
            if frame.count_nonzero().item() == 0:
                frame_num += 1
                continue

            # Main logic as described above
            gaze_type = gaze[frame_num, 2]
            if gaze_type == 1:
                if not visit:   # add start of visit
                    visit.append(relative_frame_num)
                    x, y = gaze[frame_num, :2]

                patch, _ = get_roi(frame, (int(x),int(y)), 256)
                label = run_CLIP(model, processor, patch, CLIP_labels, obj_labels, device)
                potential_labels[label] += 1

                frame = cv2.cvtColor(
                    frame.permute(1,2,0).numpy(),
                    cv2.COLOR_RGB2GRAY
                )
                kp, desc = SIFT.detectAndCompute(frame, None)
                kps.append(kp)
                descs.append(desc)
            
            elif gaze_type == 2:
                if potential_labels:  # not empty aka we've started tracking
                    visit.append(relative_frame_num - 1)

                    # update graph (create new nodes or merge to existing) and edge features
                    curr_node = update_graph(
                        curr_node,
                        potential_labels,
                        visit, 
                        kps,
                        descs,
                        (prev_x, prev_y),
                        (x, y),
                        edge_data,
                        edge_index,
                        num_nodes,
                    )

                    # update node features 
                    id = curr_node.id
                    num_visits = len(curr_node.visits)
                    total_frames_visited = sum([visit[1] - visit[0] + 1 for visit in curr_node.visits])
                    first_frame = curr_node.visits[0][0] / (vid_length - frame_num + relative_frame_num)    # normalized + accounts for black frames
                    last_frame = curr_node.visits[-1][-1] / (vid_length - frame_num + relative_frame_num)  

                    if id in node_data:
                        node_data[id][:4] = torch.tensor([
                            total_frames_visited,  # this has to be normalized wrt timestamp
                            num_visits, # this has to be normalized at the end wrt max of column
                            first_frame,
                            last_frame
                        ])
                    else:
                        one_hot = torch.zeros(len(obj_labels))
                        c = labels_to_int[curr_node.object_label]
                        one_hot[c] = 1
                        node_data[id] = torch.cat([
                            torch.tensor([total_frames_visited, num_visits, first_frame, last_frame, -1]), # -1 = placeholder for timestamp fraction
                            one_hot
                        ])

                    visit = []
                    kps, descs = [], []
                    prev_x, prev_y = x, y
                    potential_labels = defaultdict(int)

            relative_frame_num += 1
            frame_num += 1

        if print_graph:
            if num_nodes[0] > 0:
                start_node = root.neighbors[0][0]
                print_levels(start_node)    # TODO print this to an out file
            else:
                print('Something went wrong... No nodes were added to the graph.\n Maybe your video is empty or no fixations occured')

    full_dataset = {
        'x': all_node_data,
        'edge_index': all_edge_indices,
        'edge_attr': all_edge_data,
        'y': all_labels
    }
    return full_dataset