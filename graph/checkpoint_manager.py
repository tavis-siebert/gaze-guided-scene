from typing import Dict, List, Optional, Tuple, Any, Callable
import torch
from dataclasses import dataclass
import json
import numpy as np
from collections import defaultdict
import os
from pathlib import Path

from graph.graph import Graph
from graph.graph_tracer import GraphTracer
from logger import get_logger

logger = get_logger(__name__)

@dataclass
class GraphCheckpoint:
    """Encapsulates graph state at a specific timestamp."""
    # Graph structure
    nodes: Dict[int, Dict]
    edges: List[Dict]
    adjacency: Dict[int, List[int]]
    
    # Metadata per checkpoint
    frame_number: int
    non_black_frame_count: int
    
    # Shared video context - only needed for deserialization
    video_name: Optional[str] = None
    labels_to_int: Optional[Dict[str, int]] = None
    num_object_classes: Optional[int] = None
    video_length: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert checkpoint to serializable dictionary without shared context."""
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "adjacency": self.adjacency,
            "frame_number": self.frame_number,
            "non_black_frame_count": self.non_black_frame_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict, context: Dict = None) -> 'GraphCheckpoint':
        """Create checkpoint from dictionary with optional shared context.
        
        Args:
            data: Dictionary with checkpoint data
            context: Optional shared context data (video_name, labels_to_int, etc.)
        """
        context = context or {}
        return cls(
            nodes=data["nodes"],
            edges=data["edges"],
            adjacency=data["adjacency"],
            frame_number=data["frame_number"],
            non_black_frame_count=data["non_black_frame_count"],
            video_name=context.get("video_name"),
            labels_to_int=context.get("labels_to_int"),
            num_object_classes=context.get("num_object_classes"),
            video_length=context.get("video_length")
        )


class CheckpointManager:
    """Manages the creation and storage of graph checkpoints."""
    
    def __init__(
        self, 
        graph: Graph,
        gaze_data_length: int = None,
        video_name: str = "",
        output_dir: Optional[str] = None,
        split: str = "train"
    ):
        """Initialize the checkpoint manager.
        
        Args:
            graph: The graph to checkpoint
            gaze_data_length: Length of gaze data
            video_name: Name of the video being processed
            output_dir: Directory to save checkpoints to
            split: Dataset split ('train' or 'val')
        """
        self.graph = graph
        self.gaze_data_length = gaze_data_length
        self.last_checkpoint_frame = -1
        self.video_name = video_name
        self.split = split
        self.checkpoints = []
        
        # Setup output directory if provided
        if output_dir:
            self.output_dir = Path(output_dir) / split
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None
    
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
            
        # Serialize node data
        nodes_data = {}
        for node_id, node in self.graph.nodes.items():
            if node_id >= 0:  # Skip root node
                nodes_data[node_id] = {
                    "id": node.id,
                    "object_label": node.object_label,
                    "visits": node.visits
                }
        
        # Serialize edge data
        edges_data = []
        for edge in self.graph.edges:
            if edge.source_id >= 0 and edge.target_id >= 0:  # Skip edges connected to root
                edges_data.append({
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "angle": edge.angle,
                    "prev_gaze_pos": edge.prev_gaze_pos,
                    "curr_gaze_pos": edge.curr_gaze_pos
                })
        
        # Convert adjacency to serializable format
        adjacency_data = {k: v for k, v in self.graph.adjacency.items()}
        
        checkpoint = GraphCheckpoint(
            nodes=nodes_data,
            edges=edges_data,
            adjacency=adjacency_data,
            frame_number=frame_num,
            non_black_frame_count=non_black_frame_count,
            # Include context for the full checkpoint object
            video_name=self.video_name,
            labels_to_int=self.graph.labels_to_int,
            num_object_classes=self.graph.num_object_classes,
            video_length=self.graph.video_length
        )
        
        self.checkpoints.append(checkpoint)
        self.last_checkpoint_frame = frame_num
        
        # Log checkpoint creation
        self.graph.tracer.log_checkpoint_created(
            frame_num,
            self.graph.num_nodes,
            len(self.graph.edges)
        )
        
        logger.info(f"\n[Frame {frame_num}] Created graph checkpoint")
        logger.debug(f"Checkpoint created:")
        logger.debug(f"- Nodes: {self.graph.num_nodes}")
        logger.debug(f"- Edges: {len(self.graph.edges)}")
        
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
        # Skip if no edges in the graph
        if not self.graph.edges:
            return None
            
        # Serialize current graph state
        current_nodes = {}
        for node_id, node in self.graph.nodes.items():
            if node_id >= 0:  # Skip root node
                current_nodes[node_id] = {
                    "id": node.id,
                    "object_label": node.object_label,
                    "visits": node.visits
                }
                
        current_edges = []
        for edge in self.graph.edges:
            if edge.source_id >= 0 and edge.target_id >= 0:  # Skip edges connected to root
                current_edges.append({
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "angle": edge.angle,
                    "prev_gaze_pos": edge.prev_gaze_pos,
                    "curr_gaze_pos": edge.curr_gaze_pos
                })
                
        current_adjacency = {k: v for k, v in self.graph.adjacency.items()}
        
        # Current serialized state
        current_state = {
            "nodes": current_nodes,
            "edges": current_edges,
            "adjacency": current_adjacency
        }
        
        # Always create the first checkpoint
        if not self.checkpoints:
            return self.create_checkpoint(frame_num, non_black_frame_count)
        
        # Get the last checkpoint for comparison
        last_checkpoint = self.checkpoints[-1]
        
        # Last checkpoint serialized state
        last_state = {
            "nodes": last_checkpoint.nodes,
            "edges": last_checkpoint.edges,
            "adjacency": last_checkpoint.adjacency
        }
        
        # Directly compare serialized representations
        if current_state == last_state:
            # No changes detected, skip creating a checkpoint
            logger.debug(f"[Frame {frame_num}] No graph changes, skipping checkpoint")
            return None
        else:
            # Graph changed, create a new checkpoint
            return self.create_checkpoint(frame_num, non_black_frame_count)
    
    def save_checkpoints(self) -> Optional[str]:
        """Save all checkpoints to disk using a more portable format.
        
        Returns:
            Path to the saved file or None if saving failed
        """
        if not self.output_dir or not self.checkpoints:
            return None
        
        output_file = self.output_dir / f"{self.video_name}_graph.pth"
        logger.info(f"Saving {len(self.checkpoints)} checkpoints to {output_file}")
        
        # Extract shared context data
        if self.checkpoints:
            first_checkpoint = self.checkpoints[0]
            context = {
                "video_name": self.video_name,
                "labels_to_int": first_checkpoint.labels_to_int,
                "num_object_classes": first_checkpoint.num_object_classes,
                "video_length": first_checkpoint.video_length
            }
        else:
            context = {}
        
        # Create the portable dictionary format
        checkpoint_data = {
            "context": context,
            "checkpoints": [cp.to_dict() for cp in self.checkpoints]
        }
        
        torch.save(checkpoint_data, output_file)
        return str(output_file)
    
    @staticmethod
    def load_checkpoints(file_path: str) -> List[GraphCheckpoint]:
        """Load checkpoints from disk.
        
        Args:
            file_path: Path to the checkpoint file
            
        Returns:
            List of GraphCheckpoint objects
        """
        with torch.serialization.safe_globals([GraphCheckpoint]):
            data = torch.load(file_path, weights_only=False)
            
            if isinstance(data, dict) and "checkpoints" in data:
                context = data.get("context", {})
                return [GraphCheckpoint.from_dict(cp, context) for cp in data["checkpoints"]]
            else:
                logger.error(f"Unsupported checkpoint format in {file_path}. Use the conversion script.")
                return [] 