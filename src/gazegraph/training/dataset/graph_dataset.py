import random
import torch
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
import numpy as np

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
        """Initialize the dataset.
        
        Args:
            root_dir: Root directory containing graph checkpoints
            split: Dataset split ("train" or "val")
            val_timestamps: Timestamps to sample for validation set (as fractions of video length)
                            If None, will use config.training.val_timestamps
            task_mode: Task mode ("future_actions", "future_actions_ordered", or "next_action")
            node_drop_p: Probability of node dropping augmentation
            max_droppable: Maximum number of nodes to drop
            transform: PyG transform to apply to each data object
            pre_transform: PyG pre-transform to apply to each data object
            pre_filter: PyG pre-filter to apply to each data object
            config: Configuration object to pass to VideoMetadata
        """
        self.root_dir = Path(root_dir) / split
        self.split = split
        self.config = config
        
        self.metadata = VideoMetadata(config)
        
        if self.config and hasattr(self.config.training, 'val_timestamps'):
            self.val_timestamps = self.config.training.val_timestamps
        else:
            raise ValueError("No validation timestamps provided")
            
        self.task_mode = task_mode
        self.node_drop_p = node_drop_p
        self.max_droppable = max_droppable
        self.device = device
        
        # Initialize node feature extractor
        self.node_feature_type = node_feature_type
        self.node_feature_extractor = get_node_feature_extractor(node_feature_type, device=device)
        
        # For ROI embeddings, we need to set up additional context
        self.tracer_cache = {}
        self.video_cache = {}
        
        # We don't initialize video and tracer objects here
        # They will be initialized on-demand when processing each checkpoint
        
        # Find all graph checkpoint files
        self.checkpoint_files = list(self.root_dir.glob("*_graph.pth"))
        
        # Load each video's checkpoints and build sample tuples
        self.sample_tuples: List[Tuple[GraphCheckpoint, dict]] = []
        
        for file_path in tqdm(self.checkpoint_files, desc=f"Loading {self.split} checkpoints"):
            if self.split == "val":
                # For validation, sample tuples are populated in filter method
                self._load_and_filter_checkpoints(file_path)
            else:
                # For training, sampling populates sample_tuples
                checkpoints = self._load_checkpoints(file_path)
                video_name = Path(file_path).stem.split('_')[0]
                self._process_training_checkpoints(checkpoints, video_name)
        
        # Initialize PyG Dataset
        super().__init__(root=str(self.root_dir), transform=transform, 
                         pre_transform=pre_transform, pre_filter=pre_filter)
    
    def _process_training_checkpoints(self, checkpoints: List[GraphCheckpoint], video_name: str):
        """Process training checkpoints with or without oversampling.
        
        Args:
            checkpoints: List of checkpoints to process
            video_name: Name of the video
        """
        # Set action labels for each checkpoint
        processed_checkpoints = []
        for checkpoint in checkpoints:
            if not hasattr(checkpoint, 'video_name') or checkpoint.video_name is None:
                checkpoint.video_name = video_name
            processed_checkpoints.append(checkpoint)
        
        # Apply sampling if configured
        if self.config and getattr(self.config.dataset, 'sampling', None):
            sampling_cfg = self.config.dataset.sampling
            strategy = sampling_cfg.strategy
            k = sampling_cfg.samples_per_video
            allow_dup = sampling_cfg.allow_duplicates
            oversampling = sampling_cfg.oversampling
            
            # Set seed for reproducibility if provided
            seed = sampling_cfg.random_seed
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
            
            # Sample according to strategy
            samples = get_samples(
                processed_checkpoints,
                video_name,
                strategy,
                k,
                allow_dup,
                oversampling,
                self.metadata
            )
            
            self.sample_tuples.extend(samples)
        else:
            # No sampling, use all checkpoints but still filter for valid action labels
            for checkpoint in processed_checkpoints:
                if (labels := checkpoint.get_future_action_labels(checkpoint.frame_number, self.metadata)) is not None:
                    self.sample_tuples.append((checkpoint, labels))
    
    def _load_checkpoints(self, file_path: Path) -> List[GraphCheckpoint]:
        """Load all checkpoints from file without filtering.
        
        Args:
            file_path: Path to checkpoint file
            
        Returns:
            List of all checkpoints
        """
        # Extract video name for looking up records
        video_name = Path(file_path).stem.split('_')[0]  # Assuming format: video_name_graph.pth
        
        # Use the CheckpointManager to load checkpoints
        checkpoints = CheckpointManager.load_checkpoints(str(file_path))
        
        # Add video_length if not present
        for checkpoint in checkpoints:
            if not hasattr(checkpoint, 'video_length') or checkpoint.video_length is None:
                checkpoint.video_length = self.metadata.get_video_length(video_name)
                
            # Ensure video_name is set
            if not hasattr(checkpoint, 'video_name') or checkpoint.video_name is None:
                checkpoint.video_name = video_name
                
        return checkpoints
    
    def _load_and_filter_checkpoints(self, file_path: Path):
        """Load checkpoints from file and filter based on split.
        
        Args:
            file_path: Path to checkpoint file
        """
        # Extract video name for looking up records
        video_name = Path(file_path).stem.split('_')[0]  # Assuming format: video_name_graph.pth
        
        # Use the CheckpointManager to load checkpoints
        all_checkpoints = CheckpointManager.load_checkpoints(str(file_path))
        
        # Add video_length if not present
        for checkpoint in all_checkpoints:
            if not hasattr(checkpoint, 'video_length') or checkpoint.video_length is None:
                checkpoint.video_length = self.metadata.get_video_length(video_name)
                
            # Ensure video_name is set
            if not hasattr(checkpoint, 'video_name') or checkpoint.video_name is None:
                checkpoint.video_name = video_name
        
        # In train mode, keep all checkpoints
        if self.split == "train":
            for checkpoint in all_checkpoints:
                if (labels := checkpoint.get_future_action_labels(checkpoint.frame_number, self.metadata)) is not None:
                    self.sample_tuples.append((checkpoint, labels))
        
        # In val mode, sample checkpoints at specific timestamps
        elif self.split == "val":
            # Group by video
            if not all_checkpoints:
                return
                
            # Get video info from first checkpoint
            video_length = all_checkpoints[0].video_length
            
            # Calculate frame numbers from timestamp ratios
            timestamp_frames = [int(ratio * video_length) for ratio in self.val_timestamps]
            
            # Find the closest checkpoint to each target frame
            selected_checkpoints = []

            # Create a dictionary of checkpoints for quick lookup
            checkpoint_dict = {cp.frame_number: cp for cp in all_checkpoints}

            # Get sorted list of checkpoint frames
            checkpoint_frames = sorted(checkpoint_dict.keys())

            # For each target frame, get closest checkpoint <= target frame
            for target_frame in timestamp_frames:
                # Find the index of the first checkpoint frame greater than target frame
                idx = bisect_right(checkpoint_frames, target_frame)
                if idx == 0:
                    closest = checkpoint_dict[checkpoint_frames[0]]
                else:
                    closest = checkpoint_dict[checkpoint_frames[idx - 1]]
                
                if (labels := closest.get_future_action_labels(closest.frame_number, self.metadata)) is not None:
                    selected_checkpoints.append(closest)
                    self.sample_tuples.append((closest, labels))

        # Ensure sample tuples are sorted by increasing frame number
        self.sample_tuples.sort(key=lambda x: x[0].frame_number)
    
    def len(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.sample_tuples)
    
    def get(self, idx: int) -> Data:
        """Get a single graph data object.
        
        Args:
            idx: Index of the sample
            
        Returns:
            PyG Data object
        """
        checkpoint, action_labels = self.sample_tuples[idx]

        # Get node features
        node_features = self._extract_node_features(checkpoint)
        
        # Get edge indices and attributes
        edge_index, edge_attr = self._extract_edge_features(checkpoint)
        
        # Get label for the specified task mode
        y = action_labels[self.task_mode]
        
        # Create PyG data object
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
        # Apply node dropping augmentation if enabled
        if self.node_drop_p > 0 and random.random() < self.node_drop_p:
            augmented_data = node_dropping(data.x, data.edge_index, data.edge_attr, self.max_droppable)
            if augmented_data is not None:
                data = augmented_data
                
        return data
    
    def _get_tracer_for_checkpoint(self, checkpoint: GraphCheckpoint):
        """Get the appropriate GraphTracer for a checkpoint.
        
        Args:
            checkpoint: GraphCheckpoint object
            
        Returns:
            GraphTracer object for the checkpoint
        """
        from graph.graph_tracer import GraphTracer
        from pathlib import Path
        
        video_name = checkpoint.video_name
        
        # Return cached tracer if available
        if video_name in self.tracer_cache:
            return self.tracer_cache[video_name]
        
        # Initialize a new tracer
        trace_path = Path(self.config.directories.traces) / f"{video_name}_trace.jsonl"
        if not trace_path.exists():
            self.logger.warning(f"Trace file not found at {trace_path}. ROI embeddings may not work correctly.")
            return None
            
        tracer = GraphTracer(trace_path.parent, video_name, enabled=False)
        
        # Cache the tracer
        self.tracer_cache[video_name] = tracer
        return tracer
        
    def _get_video_for_checkpoint(self, checkpoint: GraphCheckpoint):
        """Get the appropriate Video processor for a checkpoint.
        
        Args:
            checkpoint: GraphCheckpoint object
            
        Returns:
            Video object for the checkpoint
        """
        from datasets.egtea_gaze.video_processor import Video
        from pathlib import Path
        
        video_name = checkpoint.video_name
        
        # Return cached video if available
        if video_name in self.video_cache:
            return self.video_cache[video_name]
        
        # Initialize a new video processor
        video_path = Path(self.config.dataset.egtea.raw_videos) / f"{video_name}.mp4"
        if not video_path.exists():
            self.logger.warning(f"Video file not found at {video_path}. ROI embeddings may not work correctly.")
            return None
            
        video = Video(video_name)
        
        # Cache the video
        self.video_cache[video_name] = video
        return video
    
    def _extract_node_features(self, checkpoint: GraphCheckpoint) -> torch.Tensor:
        """Extract node features from a checkpoint.
        
        Args:
            checkpoint: GraphCheckpoint object
            
        Returns:
            Tensor of node features
        """
        # If using ROI embeddings, set the context for the current checkpoint
        if self.node_feature_type == "roi-embeddings" and hasattr(self.node_feature_extractor, "set_context"):
            tracer = self._get_tracer_for_checkpoint(checkpoint)
            video = self._get_video_for_checkpoint(checkpoint)
            
            if tracer and video:
                self.node_feature_extractor.set_context(tracer, video)
            else:
                self.logger.warning(f"Could not set ROI embedding context for checkpoint {checkpoint.video_name}")
        
        # Use the node feature extractor to get the features
        return self.node_feature_extractor.extract_features(checkpoint)
    
    def _extract_edge_features(self, checkpoint: GraphCheckpoint) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract edge indices and attributes from a checkpoint.
        
        Args:
            checkpoint: GraphCheckpoint object
            
        Returns:
            Tuple of (edge_indices, edge_attributes)
        """
        if not checkpoint.edges:
            # No edges, return empty tensors
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1))
            
        # Collect edge data
        edge_list = []
        edge_attrs = []
        
        for edge in checkpoint.edges:
            edge_list.append((edge["source_id"], edge["target_id"]))
            
            # Edge attribute is angle
            edge_attrs.append([edge.get("angle", 0.0)])
            
        # Convert to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        # Normalize edge attributes if needed
        if edge_attr.shape[0] > 0 and edge_attr.max() > 0:
            edge_attr = edge_attr / (edge_attr.max() + 1e-8)
            
        return edge_index, edge_attr 