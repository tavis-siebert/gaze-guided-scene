import json
import os
import multiprocessing as mp
import torch
from itertools import islice

from egtea_gaze.utils import SCRATCH, EGTEA_DIR
from graph.build_graph import build_graph

#TODO make these argparsed
ann_file_train = SCRATCH + '/ego-topo/data/gtea/split/train_S1.csv'
ann_file_val = SCRATCH + '/ego-topo/data/gtea/split/val_S1.csv'
train_timestamps = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
val_timestamps = [0.25, 0.5, 0.75]
num_action_classes = 106
save_dir = os.path.dirname(os.path.realpath(__file__))

def split_list(lst, n):
    """Splits a list into n roughly equal parts."""
    avg = len(lst) // n
    remainder = len(lst) % n
    split_sizes = [avg + (1 if i < remainder else 0) for i in range(n)]
    it = iter(lst)
    return [list(islice(it, size)) for size in split_sizes]

def build_dataset(train_vids, val_vids, gpu_id, result_queue):
    torch.cuda.set_device(gpu_id)
    train_data = build_graph(train_vids, ann_file_train, train_timestamps, num_action_classes)
    val_data = build_graph(val_vids, ann_file_val, val_timestamps, num_action_classes)
    data = {
        'train': train_data,
        'val': val_data
    }

    # save to disk because pickling in process kills it
    out_file = save_dir + f'/data_subset_{gpu_id}.pth'
    torch.save(data, out_file)

    result_queue.put(out_file)

if __name__ == '__main__':
    with open(SCRATCH + '/ego-topo/data/gtea/train_test_splits.json') as f:
        split = json.load(f)

    train_videos = split['train_vids']
    val_videos = split['val_vids']
    
    num_gpus = torch.cuda.device_count()
    train_splits = split_list(train_videos, num_gpus)
    val_splits = split_list(val_videos, num_gpus)

    processes, result_queue = [], mp.Queue()
    for gpu_id in range(num_gpus):
        train_subset = train_splits[gpu_id]
        val_subset = val_splits[gpu_id]
        
        p = mp.Process(target=build_dataset, args=(train_subset, val_subset, gpu_id, result_queue))
        p.start()
        processes.append(p)

    saved_subsets = [result_queue.get() for _ in range(num_gpus)]

    for p in processes:
        p.join()

    # merge 
    dataset = {
        'train': {'x': [],  'edge_index': [], 'edge_attr': [], 'y': []},
        'test': {'x': [],  'edge_index': [], 'edge_attr': [], 'y': []}
    }

    for subset_path in saved_subsets:
        data_subset = torch.load(subset_path, map_location='cpu')
        for split, tensor_dict in data_subset.items():
            for data_type, tensor_list in tensor_dict.items():
                dataset[split][data_type].extend(tensor_list)

    # save
    torch.save(dataset, save_dir + '/dataset.pth')