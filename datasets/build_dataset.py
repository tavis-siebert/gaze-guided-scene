import json
import os
import multiprocessing as mp
import torch
from itertools import islice
from pathlib import Path
from tqdm import tqdm

from graph.build_graph import build_graph
from egtea_gaze.constants import NUM_ACTION_CLASSES
from config.config_utils import DotDict
from logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

def split_list(lst, n):
    """Splits a list into n roughly equal parts."""
    avg = len(lst) // n
    remainder = len(lst) % n
    split_sizes = [avg + (1 if i < remainder else 0) for i in range(n)]
    it = iter(lst)
    return [list(islice(it, size)) for size in split_sizes]

def build_dataset_subset(train_vids, val_vids, device_id, config: DotDict, result_queue, use_gpu: bool = False):
    """Build dataset subset using specified device (GPU or CPU)."""
    # Get a logger for the subprocess
    subprocess_logger = get_logger(f"{__name__}.device{device_id}")
    
    if use_gpu:
        torch.cuda.set_device(device_id)
    
    device_name = f"GPU {device_id}" if use_gpu else f"CPU {device_id}"
    subprocess_logger.info(f"Starting dataset building on {device_name}")
    
    # Create progress bars for each split
    train_data = build_graph(
        video_list=train_vids,
        config=config,
        split='train',
        desc=f"{device_name} - Training"
    )
    
    val_data = build_graph(
        video_list=val_vids,
        config=config,
        split='val',
        desc=f"{device_name} - Validation"
    )
    
    data = {
        'train': train_data,
        'val': val_data
    }

    # Save to disk because pickling in process kills it
    out_file = Path(config.dataset.output.subset_prefix + f"{device_id}.pth")
    torch.save(data, out_file)
    subprocess_logger.info(f"Saved dataset subset to {out_file}")
    result_queue.put(str(out_file))

def build_dataset(config: DotDict, debug: bool = False):
    """Build dataset using all available GPUs or CPU. Set debug=True to process only one video per split."""
    logger.info("Starting dataset building process...")
    
    with open(config.dataset.ego_topo.splits.train_test) as f:
        split = json.load(f)

    # In debug mode, take only one video from each split
    train_videos = split['train_vids'][:1] if debug else split['train_vids']
    val_videos = split['val_vids'][:1] if debug else split['val_vids']
    
    # In debug mode, force single CPU processing
    if debug:
        num_devices = 1
        use_gpu = False
        device_type = "CPU"
        logger.debug("Debug mode enabled: using single CPU and processing only one video per split")
    else:
        use_gpu = torch.cuda.is_available()
        num_devices = torch.cuda.device_count() if use_gpu else config.processing.n_cores
        device_type = "GPU" if use_gpu else "CPU"
    
    logger.info(f"Using {num_devices} {device_type}(s) for dataset building")
    logger.info(f"Total videos to process - Train: {len(train_videos)}, Val: {len(val_videos)}")
    
    train_splits = split_list(train_videos, num_devices)
    val_splits = split_list(val_videos, num_devices)

    processes, result_queue = [], mp.Queue()
    
    with tqdm(total=num_devices, desc="Launching processes") as pbar:
        for device_id in range(num_devices):
            train_subset = train_splits[device_id]
            val_subset = val_splits[device_id]
            
            p = mp.Process(
                target=build_dataset_subset, 
                args=(train_subset, val_subset, device_id, config, result_queue, use_gpu)
            )
            p.start()
            processes.append(p)
            pbar.update(1)

    saved_subsets = []
    with tqdm(total=num_devices, desc="Collecting results") as pbar:
        for _ in range(num_devices):
            saved_subsets.append(result_queue.get())
            pbar.update(1)

    for p in processes:
        p.join()

    # Merge subsets
    dataset = {
        'train': {'x': [], 'edge_index': [], 'edge_attr': [], 'y': []},
        'val': {'x': [], 'edge_index': [], 'edge_attr': [], 'y': []}
    }

    logger.info("Merging dataset subsets...")
    for subset_path in tqdm(saved_subsets, desc="Merging subsets"):
        data_subset = torch.load(subset_path, map_location='cpu')
        for split, tensor_dict in data_subset.items():
            for data_type, tensor_list in tensor_dict.items():
                dataset[split][data_type].extend(tensor_list)

    # Save final dataset
    save_path = Path(config.dataset.output.dataset_file)
    logger.info(f"Saving complete dataset to {save_path}")
    torch.save(dataset, save_path)
    
    logger.info("Dataset building completed successfully!")
    return dataset