"""
Sampling functionality for graph datasets.

This module provides sampling strategies for graph datasets, including:
- Uniform sampling from checkpoints
- Random sampling from checkpoints
- Oversampling from all frames, using the latest checkpoint
"""

import random
from typing import Dict, List, Tuple
import numpy as np

from gazegraph.graph.checkpoint_manager import GraphCheckpoint
from gazegraph.datasets.egtea_gaze.video_metadata import VideoMetadata


def get_samples(
    checkpoints: List[GraphCheckpoint],
    video_name: str,
    strategy: str,
    samples_per_video: int,
    allow_duplicates: bool,
    oversampling: bool,
    metadata: VideoMetadata
) -> List[Tuple[GraphCheckpoint, Dict]]:
    """Sample checkpoints according to the specified strategy.
    
    Args:
        checkpoints: List of checkpoints to sample from
        video_name: Name of the video
        strategy: Sampling strategy ('all', 'uniform', or 'random')
        samples_per_video: Number of samples per video
        allow_duplicates: Whether to allow duplicate samples
        oversampling: Whether to sample from all frames (not just checkpoint frames)
        metadata: VideoMetadata object to get valid frame ranges
    
    Returns:
        List of tuples (checkpoint, action_labels), where action_labels is pre-computed future action labels.
    
    Raises:
        ValueError: If the strategy is not recognized
    """
    if not checkpoints:
        return []
    
    # Sort checkpoints by frame number
    checkpoints_sorted = sorted(checkpoints, key=lambda x: x.frame_number)
    
    # If not oversampling, the frame number is the same as the checkpoint frame number
    if not oversampling:
        return _sample_from_checkpoints(
            checkpoints_sorted, strategy, samples_per_video, allow_duplicates, metadata
        )
    else:
        return _sample_with_oversampling(
            checkpoints_sorted, video_name, strategy, samples_per_video, 
            allow_duplicates, metadata
        )


def _sample_from_checkpoints(
    checkpoints: List[GraphCheckpoint],
    strategy: str,
    samples_per_video: int,
    allow_duplicates: bool,
    metadata: VideoMetadata
) -> List[Tuple[GraphCheckpoint, Dict]]:
    """Sample from available checkpoint frames without oversampling.
    
    Args:
        checkpoints: Sorted list of checkpoints to sample from
        strategy: Sampling strategy ('all', 'uniform', or 'random')
        samples_per_video: Number of samples per video
        allow_duplicates: Whether to allow duplicate samples
        metadata: VideoMetadata object used to get future action labels
    
    Returns:
        List of tuples (checkpoint, action_labels)
    
    Raises:
        ValueError: If the strategy is not recognized
    """
    # Build all valid (checkpoint, labels) pairs
    potential = []
    for cp in checkpoints:
        labels = cp.get_future_action_labels(cp.frame_number, metadata)
        if labels is not None:
            potential.append((cp, labels))
    if not potential:
        return []
    
    # 'all' strategy: return all valid checkpoints
    if strategy == 'all' or samples_per_video <= 0:
        return potential
    
    # 'uniform' strategy: sample evenly spaced checkpoints
    if strategy == 'uniform':
        if samples_per_video >= len(potential):
            return random.choices(potential, k=samples_per_video) if allow_duplicates else potential
        indices = np.linspace(0, len(potential) - 1, samples_per_video, dtype=int).tolist()
        return [potential[i] for i in indices]
    
    # 'random' strategy: sample random checkpoints
    elif strategy == 'random':
        if samples_per_video >= len(potential):
            return random.choices(potential, k=samples_per_video) if allow_duplicates else potential
        return random.choices(potential, k=samples_per_video) if allow_duplicates else random.sample(potential, samples_per_video)
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")


def _sample_with_oversampling(
    checkpoints: List[GraphCheckpoint],
    video_name: str,
    strategy: str,
    samples_per_video: int,
    allow_duplicates: bool,
    metadata: VideoMetadata
) -> List[Tuple[GraphCheckpoint, Dict]]:
    """Sample from all valid frames using oversampling.
    
    Args:
        checkpoints: Sorted list of checkpoints to sample from
        video_name: Name of the video
        strategy: Sampling strategy ('all', 'uniform', or 'random')
        samples_per_video: Number of samples per video
        allow_duplicates: Whether to allow duplicate samples
        metadata: VideoMetadata object to get valid frame ranges
    
    Returns:
        List of tuples (checkpoint, action_labels)
    
    Raises:
        ValueError: If the strategy is not recognized
    """
    # Get valid frame range for the video
    try:
        start_frame, end_frame = metadata.get_action_frame_range(video_name)
    except ValueError:
        # Fall back to regular sampling if no action records are found
        return _sample_from_checkpoints(checkpoints, strategy, samples_per_video, allow_duplicates, metadata)
    
    # All valid frames for sampling
    valid_frames = list(range(start_frame, end_frame + 1))
    
    # Build all valid (checkpoint, labels) pairs for oversampling
    potential = []
    for frame in valid_frames:
        suitable = [cp for cp in checkpoints if cp.frame_number <= frame]
        if not suitable:
            continue
        cp = suitable[-1]
        labels = cp.get_future_action_labels(frame, metadata)
        if labels is not None:
            potential.append((cp, labels))
    if not potential:
        return []
    
    # 'all' strategy: return all valid frames
    if strategy == 'all' or samples_per_video <= 0:
        return potential
    
    # 'uniform' strategy: sample evenly spaced frames
    elif strategy == 'uniform':
        if samples_per_video >= len(potential):
            return random.choices(potential, k=samples_per_video) if allow_duplicates else potential
        indices = np.linspace(0, len(potential) - 1, samples_per_video, dtype=int).tolist()
        return [potential[i] for i in indices]
    
    # 'random' strategy: sample random frames
    elif strategy == 'random':
        if samples_per_video >= len(potential):
            return random.choices(potential, k=samples_per_video) if allow_duplicates else potential
        return random.choices(potential, k=samples_per_video) if allow_duplicates else random.sample(potential, samples_per_video)
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}") 