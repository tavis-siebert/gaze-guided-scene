import random
import torch
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
import numpy as np
from bisect import bisect_right

from gazegraph.graph.checkpoint_manager import GraphCheckpoint, CheckpointManager
from gazegraph.datasets.egtea_gaze.video_metadata import VideoMetadata
from gazegraph.training.dataset.augmentations import node_dropping
from gazegraph.training.dataset.sampling import get_samples
from gazegraph.training.dataset.node_features import NodeFeatureExtractor, get_node_feature_extractor
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
        val_timestamps: List[float] = None,
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
        self.val_timestamps = getattr(config.training, 'val_timestamps', None)
        if self.val_timestamps is None:
            raise ValueError("No validation timestamps provided")
        self.task_mode = task_mode
        self.node_drop_p = node_drop_p
        self.max_droppable = max_droppable
        self.device = device
        self.node_feature_type = node_feature_type
        self.node_feature_extractor = get_node_feature_extractor(node_feature_type, device=device)

        self.checkpoint_files = list(self.root_dir.glob("*_graph.pth"))
        self.sample_tuples: List[Tuple[GraphCheckpoint, dict]] = []
        for file_path in tqdm(self.checkpoint_files, desc=f"Loading {self.split} checkpoints"):
            self._load_and_collect_samples(file_path)
        super().__init__(root=str(self.root_dir), transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
    
    def _load_and_collect_samples(self, file_path: Path):
        video_name = Path(file_path).stem.split('_')[0]
        checkpoints = self._load_checkpoints(file_path, video_name)
        if self.split == "val":
            self._add_val_samples(checkpoints)
        else:
            self._add_train_samples(checkpoints, video_name)

    def _load_checkpoints(self, file_path: Path, video_name: str) -> List[GraphCheckpoint]:
        checkpoints = CheckpointManager.load_checkpoints(str(file_path))
        for cp in checkpoints:
            if not hasattr(cp, 'video_length') or cp.video_length is None:
                cp.video_length = self.metadata.get_video_length(video_name)
            if not hasattr(cp, 'video_name') or cp.video_name is None:
                cp.video_name = video_name
        return checkpoints

    def _add_train_samples(self, checkpoints: List[GraphCheckpoint], video_name: str):
        sampling_cfg = getattr(self.config.dataset, 'sampling', None) if self.config else None
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
            augmented_data = node_dropping(data.x, data.edge_index, data.edge_attr, self.max_droppable)
            if augmented_data is not None:
                data = augmented_data
        return data

    def _get_tracer_for_checkpoint(self, checkpoint: GraphCheckpoint):
        video_name = checkpoint.video_name
        trace_path = Path(self.config.directories.traces) / f"{video_name}_trace.jsonl"
        if not trace_path.exists():
            logger.warning(f"Trace file not found at {trace_path}. ROI embeddings may not work correctly.")
            return None
        tracer = GraphTracer(trace_path.parent, video_name, enabled=False)
        return tracer

    def _get_video_for_checkpoint(self, checkpoint: GraphCheckpoint):
        video_name = checkpoint.video_name
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
                self.node_feature_extractor.set_context(tracer, video)
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