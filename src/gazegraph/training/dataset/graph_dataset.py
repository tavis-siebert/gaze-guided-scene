import random
from pathlib import Path
from typing import List, Optional, Tuple, Literal
from torch_geometric.data import Data, Dataset, InMemoryDataset
from gazegraph.training.dataset.graph_assembler import create_graph_assembler
from tqdm import tqdm
import numpy as np
from bisect import bisect_right

from gazegraph.graph.checkpoint_manager import GraphCheckpoint, CheckpointManager
from gazegraph.datasets.egtea_gaze.video_metadata import VideoMetadata
from gazegraph.training.dataset.augmentations import node_dropping
from gazegraph.training.dataset.sampling import get_samples
from gazegraph.logger import get_logger
from gazegraph.config.config_utils import DotDict

logger = get_logger(__name__)


class GraphDataset(Dataset):
    """Dataset for loading graph checkpoints and creating PyG data objects."""
    
    def __init__(
        self,
        config: DotDict,
        root_dir: str,
        split: str = "train",
        task_mode: str = "future_actions",
        node_drop_p: float = 0.0,
        max_droppable: int = 0,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        object_node_feature: str = "roi-embeddings",
        action_node_feature: str = "action-label-embedding",
        device: str = "cuda",
        graph_type: Literal["object-graph", "action-graph", "action-object-graph"] = "object-graph"
    ):
        self.root_dir = Path(root_dir) / split  # use `/ (split + "_small")` if you want to test
        self.split = split
        self.config = config
        self.metadata = VideoMetadata(config)

        if self.split == "val":
            if not config or not hasattr(config, 'training') or not hasattr(config.training, 'val_timestamps'):
                raise ValueError("Config or config.training.val_timestamps missing")
            self.val_timestamps = config.training.val_timestamps
            if self.val_timestamps is None:
                raise ValueError("No validation timestamps provided")

        self.task_mode = task_mode
        self.node_drop_p = node_drop_p
        self.max_droppable = max_droppable
        self.device = device

        self.object_node_feature = object_node_feature
        self.action_node_feature = action_node_feature
        self.graph_type = graph_type
        self.checkpoint_files = list(self.root_dir.glob("*_graph.pth"))
        if not hasattr(self, 'sample_tuples'): # Exists if loaded from cache
            self.sample_tuples : List[Tuple[GraphCheckpoint, dict]] = []

        self._load_and_collect_samples()
        # Create the appropriate graph assembler based on the graph type
        self._assembler = create_graph_assembler(
            graph_type=self.graph_type,
            config=self.config,
            device=self.device,
            object_node_feature=self.object_node_feature,
            action_node_feature=self.action_node_feature,
            split=self.split    # primarily for loading caches in heterogeneous graphs
        )
        self._data_cache = {}
        super().__init__(root=str(self.root_dir), transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

    def _load_and_collect_samples(self):
        if self.sample_tuples:
            logger.info(f"Using cached samples for {self.split} split")
            return
        logger.info(f"Collecting checkpoints for {self.split} split")
        for file_path in tqdm(self.checkpoint_files, desc=f"Loading {self.split} checkpoints"):
            video_name = Path(file_path).stem.split('_')[0]
            checkpoints = CheckpointManager.load_checkpoints(str(file_path))
            if self.split == "val":
                self._add_val_samples(checkpoints)
            else:
                self._add_train_samples(checkpoints, video_name)

    def _add_train_samples(self, checkpoints: List[GraphCheckpoint], video_name: str):
        sampling_cfg = None
        if self.config and hasattr(self.config, 'dataset') and hasattr(self.config.dataset, 'sampling'):
            sampling_cfg = self.config.dataset.sampling
        else:
            raise ValueError("Sampling configuration not found in config")
        if getattr(sampling_cfg, 'random_seed', None) is not None:
            random.seed(sampling_cfg.random_seed)
            np.random.seed(sampling_cfg.random_seed)
        samples = get_samples(
            checkpoints=checkpoints,
            video_name=video_name,
            strategy=sampling_cfg.strategy,
            samples_per_video=sampling_cfg.samples_per_video,
            allow_duplicates=sampling_cfg.allow_duplicates,
            oversampling=sampling_cfg.oversampling,
            metadata=self.metadata
        )
        self.sample_tuples.extend(samples)

    def _add_val_samples(self, checkpoints: List[GraphCheckpoint]):
        if not checkpoints:
            return
        video_length = checkpoints[0].video_length
        timestamp_frames = [int(r * video_length) for r in self.val_timestamps]
        frame_to_cp = {cp.frame_number: cp for cp in checkpoints}
        frames_sorted = sorted(frame_to_cp.keys())
        for target in timestamp_frames:
            idx = bisect_right(frames_sorted, target)
            closest = frame_to_cp[frames_sorted[0]] if idx == 0 else frame_to_cp[frames_sorted[idx - 1]]
            labels = closest.get_future_action_labels(closest.frame_number, self.metadata)
            if labels is not None:
                self.sample_tuples.append((closest, labels))

    def len(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.sample_tuples)

    def get(self, idx: int) -> Data:
        """Get a single graph data object, using cache and assembler."""
        if idx in self._data_cache:
            return self._data_cache[idx]
        checkpoint, action_labels = self.sample_tuples[idx]
        y = action_labels[self.task_mode]
        data = self._assembler.assemble(checkpoint, y)
        if self.graph_type != "action-object-graph":    #TODO handle aug for hetero
            data = self._apply_augmentations(data)
        self._data_cache[idx] = data
        return data

    def _apply_augmentations(self, data: Data) -> Data:
        if self.node_drop_p > 0 and random.random() < self.node_drop_p:
            if data.x is not None and data.edge_index is not None and data.edge_attr is not None:
                augmented_data = node_dropping(data.x, data.edge_index, data.edge_attr, self.max_droppable)
            else:
                augmented_data = None
            if augmented_data is not None:
                data = augmented_data
        return data
