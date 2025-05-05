import random
import torch
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from graph.checkpoint_manager import GraphCheckpoint, CheckpointManager
from datasets.egtea_gaze.action_record import ActionRecord
from datasets.egtea_gaze.video_metadata import VideoMetadata
from training.dataset.augmentations import node_dropping
from training.dataset.sampling import get_samples


class GraphDataset(Dataset):
    """Dataset for loading graph checkpoints and creating PyG data objects."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        val_timestamps: List[float] = [0.25, 0.5, 0.75],
        task_mode: str = "future_actions",
        node_drop_p: float = 0.0,
        max_droppable: int = 0,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        config=None
    ):
        """Initialize the dataset.
        
        Args:
            root_dir: Root directory containing graph checkpoints
            split: Dataset split ("train" or "val")
            val_timestamps: Timestamps to sample for validation set (as fractions of video length)
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
        self.val_timestamps = val_timestamps
        self.task_mode = task_mode
        self.node_drop_p = node_drop_p
        self.max_droppable = max_droppable
        self.config = config
        
        # Initialize video metadata
        self.metadata = VideoMetadata(config)
        
        # Find all graph checkpoint files
        self.checkpoint_files = list(self.root_dir.glob("*_graph.pth"))
        
        # Load each video's checkpoints and select only the frames we want
        self.processed_checkpoints = []
        self.sample_tuples = []  # List of (checkpoint, frame_number) tuples
        
        for file_path in tqdm(self.checkpoint_files, desc=f"Loading {self.split} checkpoints"):
            if self.split == "val":
                # For validation, we use the specified timestamps
                checkpoints = self._load_and_filter_checkpoints(file_path)
                self.processed_checkpoints.extend(checkpoints)
                self.sample_tuples.extend([(cp, cp.frame_number) for cp in checkpoints])
            else:
                # For training, we need the raw checkpoints for sampling
                checkpoints = self._load_checkpoints(file_path)
                video_name = Path(file_path).stem.split('_')[0]  # Assuming format: video_name_graph.pth
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
            # No sampling, use all checkpoints
            self.processed_checkpoints.extend(processed_checkpoints)
            self.sample_tuples.extend([(cp, cp.frame_number) for cp in processed_checkpoints])
    
    def _load_checkpoints(self, file_path: Path) -> List[GraphCheckpoint]:
        """Load all checkpoints from file without filtering.
        
        Args:
            file_path: Path to checkpoint file
            
        Returns:
            List of all checkpoints
        """
        # Use the CheckpointManager to load checkpoints
        return CheckpointManager.load_checkpoints(str(file_path))
    
    def _load_and_filter_checkpoints(self, file_path: Path) -> List[GraphCheckpoint]:
        """Load checkpoints from file and filter based on split.
        
        Args:
            file_path: Path to checkpoint file
            
        Returns:
            List of filtered checkpoints with added action labels
        """
        # Extract video name for looking up records
        video_name = Path(file_path).stem.split('_')[0]  # Assuming format: video_name_graph.pth
        
        # Use the CheckpointManager to load checkpoints
        all_checkpoints = CheckpointManager.load_checkpoints(str(file_path))
        
        # In train mode, keep all checkpoints
        if self.split == "train":
            return all_checkpoints
        
        # In val mode, sample checkpoints at specific timestamps
        elif self.split == "val":
            # Group by video
            if not all_checkpoints:
                return []
                
            # Get video info from first checkpoint
            video_length = all_checkpoints[0].video_length
            
            # Calculate frame numbers from timestamp ratios
            timestamp_frames = [int(ratio * video_length) for ratio in self.val_timestamps]
            
            # Find the closest checkpoint to each target frame
            selected_checkpoints = []
            for target_frame in timestamp_frames:
                closest = min(all_checkpoints, 
                              key=lambda cp: abs(cp.frame_number - target_frame))
                selected_checkpoints.append(closest)
                
            return selected_checkpoints
    
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
        checkpoint, frame_number = self.sample_tuples[idx]
        
        # Get future action labels for the specific frame
        action_labels = checkpoint.get_future_action_labels(frame_number, self.metadata)
        
        # Skip if no action labels available (should not happen due to sampling)
        if action_labels is None:
            # Fallback to next sample
            return self.get((idx + 1) % len(self.sample_tuples))
        
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
    
    def _extract_node_features(self, checkpoint: GraphCheckpoint) -> torch.Tensor:
        """Extract node features from a checkpoint.
        
        Args:
            checkpoint: GraphCheckpoint object
            
        Returns:
            Tensor of node features
        """
        features_list = []
        for node_id, node_data in checkpoint.nodes.items():
            # Extract basic node information
            total_frames_visited = sum(end - start for start, end in node_data["visits"])
            num_visits = len(node_data["visits"])
            
            first_visit_frame = node_data["visits"][0][0] if node_data["visits"] else 0
            last_visit_frame = node_data["visits"][-1][1] if node_data["visits"] else 0
            
            # Normalize temporal features
            first_frame_normalized = first_visit_frame / checkpoint.non_black_frame_count
            last_frame_normalized = last_visit_frame / checkpoint.non_black_frame_count
            frame_fraction = checkpoint.frame_number / checkpoint.video_length
            
            # Create temporal features tensor
            temporal_features = torch.tensor([
                total_frames_visited,
                num_visits,
                first_frame_normalized,
                last_frame_normalized,
                frame_fraction
            ])
            
            # Normalize first feature (total frames visited)
            if checkpoint.non_black_frame_count > 0:
                temporal_features[0] /= checkpoint.non_black_frame_count
                
            # Create one-hot encoding for object class
            class_idx = checkpoint.labels_to_int.get(node_data["object_label"], 0)
            one_hot = torch.zeros(checkpoint.num_object_classes)
            one_hot[class_idx] = 1
            
            # Combine features
            node_features = torch.cat([temporal_features, one_hot])
            features_list.append(node_features)
        
        if not features_list:
            return torch.tensor([])
            
        # Stack all node features
        node_features_tensor = torch.stack(features_list)
        
        # Further normalization for visit count if needed
        if node_features_tensor[:, 1].max() > 0:
            node_features_tensor[:, 1] /= node_features_tensor[:, 1].max()
            
        return node_features_tensor
    
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