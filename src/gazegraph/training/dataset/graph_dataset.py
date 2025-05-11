import random
import torch
from pathlib import Path
from typing import List, Optional, Tuple
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
import numpy as np
from bisect import bisect_right

from gazegraph.graph.checkpoint_manager import GraphCheckpoint, CheckpointManager
from gazegraph.datasets.egtea_gaze.video_metadata import VideoMetadata
from gazegraph.training.dataset.augmentations import node_dropping
from gazegraph.training.dataset.sampling import get_samples
from gazegraph.training.dataset.node_features import get_node_feature_extractor
from gazegraph.graph.graph_tracer import GraphTracer
from gazegraph.datasets.egtea_gaze.video_processor import Video
from gazegraph.logger import get_logger

logger = get_logger(__name__)


class GraphDataset(Dataset):
    """Dataset for loading graph checkpoints and creating PyG data objects."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        val_timestamps: Optional[List[float]] = None,
        task_mode: str = "future_actions",
        node_drop_p: float = 0.0,
        max_droppable: int = 0,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        config=None,
        node_feature_type: str = "one-hot",
        device: str = "cuda"
    ):
        self.root_dir = Path(root_dir) / split
        self.split = split
        self.config = config
        self.metadata = VideoMetadata(config)
        if not config or not hasattr(config, 'training') or not hasattr(config.training, 'val_timestamps'):
            raise ValueError("Config or config.training.val_timestamps missing")
        self.val_timestamps = config.training.val_timestamps
        if self.val_timestamps is None:
            raise ValueError("No validation timestamps provided")
        self.task_mode = task_mode
        self.node_drop_p = node_drop_p
        self.max_droppable = max_droppable
        self.device = device
        self.node_feature_type = node_feature_type
        self.node_feature_extractor = get_node_feature_extractor(node_feature_type, device=device)
        self.checkpoint_files = list(self.root_dir.glob("*_graph.pth"))
        if not hasattr(self, 'sample_tuples'): # Exists if loaded from cache
            self.sample_tuples : List[Tuple[GraphCheckpoint, dict]] = []
        self._load_and_collect_samples()
        super().__init__(root=str(self.root_dir), transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

    def _load_and_collect_samples(self):
        if self.sample_tuples:
            logger.info(f"Using cached samples for {self.split} split")
            return
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
        if sampling_cfg:
            if getattr(sampling_cfg, 'random_seed', None) is not None:
                random.seed(sampling_cfg.random_seed)
                np.random.seed(sampling_cfg.random_seed)
            samples = get_samples(
                checkpoints, video_name, sampling_cfg.strategy, sampling_cfg.samples_per_video,
                sampling_cfg.allow_duplicates, sampling_cfg.oversampling, self.metadata
            )
            self.sample_tuples.extend(samples)
        else:
            for cp in checkpoints:
                labels = cp.get_future_action_labels(cp.frame_number, self.metadata)
                if labels is not None:
                    self.sample_tuples.append((cp, labels))

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
        """Get a single graph data object."""
        checkpoint, action_labels = self.sample_tuples[idx]
        node_features = self._extract_node_features(checkpoint)
        edge_index, edge_attr = self._extract_edge_features(checkpoint)
        y = action_labels[self.task_mode]
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y)
        if self.node_drop_p > 0 and random.random() < self.node_drop_p:
            if data.x is not None and data.edge_index is not None and data.edge_attr is not None:
                augmented_data = node_dropping(data.x, data.edge_index, data.edge_attr, self.max_droppable)
            else:
                augmented_data = None
            if augmented_data is not None:
                data = augmented_data
        return data

    def _get_tracer_for_checkpoint(self, checkpoint: GraphCheckpoint) -> GraphTracer | None:
        video_name = checkpoint.video_name
        if not self.config or not hasattr(self.config, 'directories') or not hasattr(self.config.directories, 'traces'):
            logger.warning("Config or directories.traces missing; cannot load trace file.")
            return None
        trace_path = Path(self.config.directories.traces) / f"{video_name}_trace.jsonl"
        if not trace_path.exists():
            logger.warning(f"Trace file not found at {trace_path}. ROI embeddings may not work correctly.")
            return None
        tracer = GraphTracer(trace_path.parent, video_name, enabled=False)
        return tracer

    def _get_video_for_checkpoint(self, checkpoint: GraphCheckpoint) -> Video | None:
        video_name = checkpoint.video_name
        if not self.config or not hasattr(self.config, 'dataset') or not hasattr(self.config.dataset, 'egtea') or not hasattr(self.config.dataset.egtea, 'raw_videos'):
            logger.warning("Config or dataset.egtea.raw_videos missing; cannot load video file.")
            return None
        video_path = Path(self.config.dataset.egtea.raw_videos) / f"{video_name}.mp4"
        if not video_path.exists():
            logger.warning(f"Video file not found at {video_path}. ROI embeddings may not work correctly.")
            return None
        video = Video(video_name)
        return video

    def _extract_node_features(self, checkpoint: GraphCheckpoint) -> torch.Tensor:
        if self.node_feature_type == "roi-embeddings" and hasattr(self.node_feature_extractor, "set_context"):
            tracer = self._get_tracer_for_checkpoint(checkpoint)
            video = self._get_video_for_checkpoint(checkpoint)
            if tracer and video:
                self.node_feature_extractor.set_context(tracer=tracer, video=video)
            else:
                logger.warning(f"Could not set ROI embedding context for checkpoint {checkpoint.video_name}")
        return self.node_feature_extractor.extract_features(checkpoint)

    def _extract_edge_features(self, checkpoint: GraphCheckpoint) -> Tuple[torch.Tensor, torch.Tensor]:
        if not checkpoint.edges:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1))
        edge_list = [(e["source_id"], e["target_id"]) for e in checkpoint.edges]
        edge_attrs = [[e.get("angle", 0.0)] for e in checkpoint.edges]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        if edge_attr.shape[0] > 0 and edge_attr.max() > 0:
            edge_attr = edge_attr / (edge_attr.max() + 1e-8)
        return edge_index, edge_attr
