"""
Action recognition sampling functionality for graph datasets.

This module provides specialized sampling strategies for action recognition tasks,
focusing on sampling at action completion with object relevance filtering.
"""

from typing import Dict, List, Tuple, Optional

from gazegraph.graph.checkpoint_manager import GraphCheckpoint
from gazegraph.datasets.egtea_gaze.video_metadata import VideoMetadata
from gazegraph.datasets.egtea_gaze.action_record import ActionRecord
from gazegraph.logger import get_logger

logger = get_logger(__name__)


class ActionRecognitionSampler:
    """Specialized sampler for action recognition tasks."""

    def __init__(self, metadata: VideoMetadata):
        self.metadata = metadata

    def get_action_recognition_samples(
        self,
        checkpoints: List[GraphCheckpoint],
        video_name: str,
        samples_per_action: int = 1,
        action_completion_ratio: float = 1.0,
        min_nodes_threshold: int = 1,
        visit_lookback_frames: int = 0,
    ) -> List[Tuple[GraphCheckpoint, Dict]]:
        """Sample checkpoints for action recognition at action completion.

        Args:
            checkpoints: List of available checkpoints
            video_name: Name of the video
            samples_per_action: Number of samples per action instance
            action_completion_ratio: Ratio of action completion for sampling (1.0 = at action end)
            min_nodes_threshold: Minimum number of nodes required after filtering
            visit_lookback_frames: Frames before action start to include visits

        Returns:
            List of (checkpoint, action_labels) tuples for action recognition
        """
        if not checkpoints:
            return []

        checkpoints_sorted = sorted(checkpoints, key=lambda x: x.frame_number)
        action_records = self.metadata.get_records_for_video(video_name)

        if not action_records:
            logger.warning(f"No action records found for video {video_name}")
            return []

        samples = []

        for action_record in action_records:
            if action_record.action_idx is None:
                continue

            # Sample at action completion point
            completion_frame = int(
                action_record.start_frame
                + action_completion_ratio
                * (action_record.end_frame - action_record.start_frame)
            )

            # Find best checkpoint for completion frame
            best_checkpoint = self._find_best_checkpoint(
                checkpoints_sorted, completion_frame
            )
            if best_checkpoint is None:
                continue

            # Apply object filtering based on action context
            filtered_checkpoint = self._filter_checkpoint_for_action(
                best_checkpoint,
                action_record,
                min_nodes_threshold,
                visit_lookback_frames,
            )

            if filtered_checkpoint is None:
                continue

            # Create action label
            action_label = {"action_recognition": action_record.action_idx}
            samples.append((filtered_checkpoint, action_label))

        logger.info(
            f"Generated {len(samples)} action recognition samples for video {video_name}"
        )
        return samples

    def _find_best_checkpoint(
        self, checkpoints: List[GraphCheckpoint], target_frame: int
    ) -> Optional[GraphCheckpoint]:
        """Find the best checkpoint for a target frame."""
        best_checkpoint = None
        best_distance = float("inf")

        for checkpoint in checkpoints:
            if checkpoint.frame_number <= target_frame:
                distance = target_frame - checkpoint.frame_number
                if distance < best_distance:
                    best_distance = distance
                    best_checkpoint = checkpoint

        return best_checkpoint

    def _filter_checkpoint_for_action(
        self,
        checkpoint: GraphCheckpoint,
        action_record: ActionRecord,
        min_nodes_threshold: int,
        visit_lookback_frames: int,
    ) -> Optional[GraphCheckpoint]:
        """Filter checkpoint nodes based on action relevance."""
        action_start = action_record.start_frame
        action_end = action_record.end_frame

        # Filter nodes to only include those with visits relevant to the action
        filtered_nodes = {}
        for node_id, node_data in checkpoint.nodes.items():
            if node_id < 0:  # Keep root node
                filtered_nodes[node_id] = node_data
                continue

            # Check if node has visits that overlap with or precede the action
            relevant_visits = []
            for visit in node_data.get("visits", []):
                visit_start, visit_end = visit
                # Include if visit overlaps with action or occurs within lookback window
                if visit_end >= action_start - visit_lookback_frames:
                    # Prune visit to only include frames before action end
                    pruned_visit = [visit_start, min(visit_end, action_end)]
                    if pruned_visit[1] >= pruned_visit[0]:  # Valid visit
                        relevant_visits.append(pruned_visit)

            if relevant_visits:
                # Update node with filtered visits
                filtered_node = node_data.copy()
                filtered_node["visits"] = relevant_visits
                filtered_nodes[node_id] = filtered_node

        # Check if we have enough nodes after filtering
        non_root_nodes = len([nid for nid in filtered_nodes.keys() if nid >= 0])
        if non_root_nodes < min_nodes_threshold:
            return None

        # Filter edges to only include edges between remaining nodes
        filtered_edges = []
        remaining_node_ids = set(filtered_nodes.keys())

        for edge in checkpoint.edges:
            if (
                edge["source_id"] in remaining_node_ids
                and edge["target_id"] in remaining_node_ids
            ):
                filtered_edges.append(edge)

        # Update adjacency
        filtered_adjacency = {}
        for source_id in remaining_node_ids:
            filtered_adjacency[source_id] = [
                target_id
                for target_id in checkpoint.adjacency.get(source_id, [])
                if target_id in remaining_node_ids
            ]

        # Create filtered checkpoint
        filtered_checkpoint = GraphCheckpoint(
            nodes=filtered_nodes,
            edges=filtered_edges,
            adjacency=filtered_adjacency,
            frame_number=checkpoint.frame_number,
            non_black_frame_count=checkpoint.non_black_frame_count,
            video_name=checkpoint.video_name,
            object_label_to_id=checkpoint.object_label_to_id,
            video_length=checkpoint.video_length,
        )

        return filtered_checkpoint


def get_action_recognition_samples(
    checkpoints: List[GraphCheckpoint],
    video_name: str,
    samples_per_action: int,
    metadata: VideoMetadata,
    **kwargs,
) -> List[Tuple[GraphCheckpoint, Dict]]:
    """Main entry point for action recognition sampling.

    Args:
        checkpoints: List of checkpoints to sample from
        video_name: Name of the video
        samples_per_action: Number of samples per action
        metadata: VideoMetadata object
        **kwargs: Additional sampling parameters

    Returns:
        List of (checkpoint, action_labels) tuples
    """
    sampler = ActionRecognitionSampler(metadata)
    return sampler.get_action_recognition_samples(
        checkpoints=checkpoints,
        video_name=video_name,
        samples_per_action=samples_per_action,
        **kwargs,
    )
