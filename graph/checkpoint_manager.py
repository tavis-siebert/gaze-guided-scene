from typing import Dict, List, Optional, Tuple, Any, Callable
import torch
from dataclasses import dataclass

from graph.graph import Graph
from graph.action_utils import ActionUtils
from logger import get_logger

logger = get_logger(__name__)

@dataclass
class GraphCheckpoint:
    """Encapsulates graph state at a specific timestamp."""
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    action_labels: Dict[str, torch.Tensor]


class CheckpointManager:
    """Manages the creation and storage of graph checkpoints."""
    
    def __init__(
        self, 
        graph: Graph,
        checkpoint_frames: List[int] = None,
        timestamps: List[int] = None,
        timestamp_ratios: List[float] = None,
        records: List[Any] = None,
        action_to_idx: Dict[Any, int] = None,
        gaze_data_length: int = None
    ):
        """Initialize the checkpoint manager.
        
        Args:
            graph: The graph to checkpoint
            checkpoint_frames: Specific frames at which to create checkpoints
            timestamps: List of predefined checkpoint frame numbers
            timestamp_ratios: Corresponding ratios for each timestamp
            records: List of action records
            action_to_idx: Mapping from actions to indices
            gaze_data_length: Length of gaze data
        """
        self.graph = graph
        self.checkpoint_frames = checkpoint_frames or []
        self.timestamps = timestamps or []
        self.timestamp_ratios = timestamp_ratios or []
        self.records = records or []
        self.action_to_idx = action_to_idx or {}
        self.gaze_data_length = gaze_data_length
        self.last_checkpoint_frame = -1
    
    def _should_checkpoint(self, frame_num: int) -> bool:
        """Determine if a checkpoint should be created at the current frame."""
        if self.gaze_data_length and frame_num >= self.gaze_data_length:
            return True
        if frame_num in self.checkpoint_frames:
            return True
        return False
    
    def create_checkpoint(
        self,
        frame_num: int,
        non_black_frame_count: int
    ) -> Optional[GraphCheckpoint]:
        """Create a checkpoint of the current graph state.
        
        Args:
            frame_num: Current frame number
            non_black_frame_count: Number of non-black frames processed
            
        Returns:
            GraphCheckpoint object or None if checkpoint creation failed
        """
        if not self.graph.edges:
            logger.info(f"Skipping checkpoint - no edges in graph")
            return None
        
        if frame_num == self.last_checkpoint_frame:
            logger.info(f"Skipping checkpoint - already created at frame {frame_num}")
            return None
            
        logger.info(f"\n[Frame {frame_num}] Creating graph checkpoint")
        
        action_labels = ActionUtils.get_future_action_labels(
            self.records, 
            frame_num, 
            self.action_to_idx
        )
            
        if action_labels is None:
            logger.info(f"Skipping checkpoint - insufficient action data")
            return None
        
        node_features, edge_index, edge_attr = self.graph.get_feature_tensor(
            video_length=self.graph.video_length,
            current_frame=frame_num,
            non_black_frame_count=non_black_frame_count,
            timestamps=self.timestamps,
            timestamp_ratios=self.timestamp_ratios,
            gaze_data_length=self.gaze_data_length,
            labels_to_int=self.graph.labels_to_int,
            num_object_classes=self.graph.num_object_classes
        )
        
        if node_features.numel() == 0:
            logger.info(f"Skipping checkpoint - no valid node features")
            return None
        
        checkpoint = GraphCheckpoint(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            action_labels=action_labels
        )
        
        self.graph.checkpoints.append(checkpoint)
        self.last_checkpoint_frame = frame_num
        
        logger.info(f"Checkpoint created:")
        logger.info(f"- Nodes: {self.graph.num_nodes}")
        logger.info(f"- Edges: {len(self.graph.edges)}")
        
        return checkpoint
    
    def checkpoint_if_needed(
        self,
        frame_num: int,
        non_black_frame_count: int
    ) -> Optional[GraphCheckpoint]:
        """Check if a checkpoint is needed and create one if necessary.
        
        Args:
            frame_num: Current frame number
            non_black_frame_count: Number of non-black frames processed
            
        Returns:
            Created checkpoint or None
        """
        if not self._should_checkpoint(frame_num):
            return None
        
        return self.create_checkpoint(
            frame_num,
            non_black_frame_count
        )
    
    @staticmethod
    def build_dataset_from_graphs(graphs: List[Graph]) -> Dict:
        """Convert a list of graphs to a dataset ready for model training.
        
        Args:
            graphs: List of Graph objects with checkpoints
            
        Returns:
            Dictionary with x, edge_index, edge_attr, and y keys for all videos
        """
        dataset = {
            'x': [],
            'edge_index': [],
            'edge_attr': [],
            'y': []
        }
        
        for graph in graphs:
            for checkpoint in graph.get_checkpoints():
                dataset['x'].append(checkpoint.node_features)
                dataset['edge_index'].append(checkpoint.edge_index)
                dataset['edge_attr'].append(checkpoint.edge_attr)
                dataset['y'].append(checkpoint.action_labels)
        
        return dataset 