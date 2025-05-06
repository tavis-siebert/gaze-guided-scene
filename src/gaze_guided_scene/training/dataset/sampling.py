"""
Sampling functionality for graph datasets.

This module provides sampling strategies for graph datasets, including:
- Uniform sampling from checkpoints
- Random sampling from checkpoints
- Oversampling from all frames, using the latest checkpoint
"""

import random
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path

from graph.checkpoint_manager import GraphCheckpoint
from datasets.egtea_gaze.video_metadata import VideoMetadata


def get_samples(
    checkpoints: List[GraphCheckpoint],
    video_name: str,
    strategy: str,
    samples_per_video: int,
    allow_duplicates: bool,
    oversampling: bool,
    metadata: VideoMetadata
) -> List[Tuple[GraphCheckpoint, int, Dict]]:
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
        List of tuples (checkpoint, frame_number, action_labels), where frame_number is the actual 
        frame to use for label generation and action_labels is pre-computed future action labels.
        When oversampling is False, frame_number will be the same as checkpoint.frame_number.
    
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
) -> List[Tuple[GraphCheckpoint, int, Dict]]:
    """Sample from available checkpoint frames without oversampling.
    
    Args:
        checkpoints: Sorted list of checkpoints to sample from
        strategy: Sampling strategy ('all', 'uniform', or 'random')
        samples_per_video: Number of samples per video
        allow_duplicates: Whether to allow duplicate samples
        metadata: VideoMetadata object used to get future action labels
    
    Returns:
        List of tuples (checkpoint, frame_number, action_labels)
    
    Raises:
        ValueError: If the strategy is not recognized
    """
    # First prepare all potential samples with their frames
    potential_samples = []
    for cp in checkpoints:
        action_labels = cp.get_future_action_labels(cp.frame_number, metadata)
        if action_labels is not None:
            potential_samples.append((cp, cp.frame_number, action_labels))
    
    n = len(potential_samples)
    if n == 0:
        return []
    
    # 'all' strategy: use all valid checkpoints
    if strategy == 'all' or samples_per_video <= 0:
        return potential_samples
    
    # 'uniform' strategy: sample evenly spaced checkpoints
    if strategy == 'uniform':
        if samples_per_video >= n:
            samples_to_use = random.choices(potential_samples, k=samples_per_video) if allow_duplicates else potential_samples
        else:
            indices = np.linspace(0, n - 1, samples_per_video, dtype=int).tolist()
            samples_to_use = [potential_samples[i] for i in indices]
    
    # 'random' strategy: sample random checkpoints
    elif strategy == 'random':
        if samples_per_video >= n:
            samples_to_use = random.choices(potential_samples, k=samples_per_video) if allow_duplicates else potential_samples
        else:
            samples_to_use = random.choices(potential_samples, k=samples_per_video) if allow_duplicates else random.sample(potential_samples, samples_per_video)
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    return samples_to_use


def _sample_with_oversampling(
    checkpoints: List[GraphCheckpoint],
    video_name: str,
    strategy: str,
    samples_per_video: int,
    allow_duplicates: bool,
    metadata: VideoMetadata
) -> List[Tuple[GraphCheckpoint, int, Dict]]:
    """Sample from all valid frames using oversampling.
    
    Args:
        checkpoints: Sorted list of checkpoints to sample from
        video_name: Name of the video
        strategy: Sampling strategy ('all', 'uniform', or 'random')
        samples_per_video: Number of samples per video
        allow_duplicates: Whether to allow duplicate samples
        metadata: VideoMetadata object to get valid frame ranges
    
    Returns:
        List of tuples (checkpoint, frame_number, action_labels)
    
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
    
    # First filter out frames with no future actions
    potential_frames = []
    for frame in valid_frames:
        # Find the most recent checkpoint that is at or before the current frame
        suitable_checkpoints = [cp for cp in checkpoints if cp.frame_number <= frame]
        if not suitable_checkpoints:
            continue
        
        checkpoint = suitable_checkpoints[-1]
        action_labels = checkpoint.get_future_action_labels(frame, metadata)
        if action_labels is not None:
            potential_frames.append((frame, checkpoint, action_labels))
    
    if not potential_frames:
        return []
    
    n_frames = len(potential_frames)
    
    # 'all' strategy: use all valid frames
    if strategy == 'all' or samples_per_video <= 0:
        frames_to_sample = potential_frames
    
    # 'uniform' strategy: sample evenly spaced frames
    elif strategy == 'uniform':
        if samples_per_video >= n_frames:
            frames_to_sample = random.choices(potential_frames, k=samples_per_video) if allow_duplicates else potential_frames
        else:
            indices = np.linspace(0, n_frames - 1, samples_per_video, dtype=int).tolist()
            frames_to_sample = [potential_frames[i] for i in indices]
    
    # 'random' strategy: sample random frames
    elif strategy == 'random':
        if samples_per_video >= n_frames:
            frames_to_sample = random.choices(potential_frames, k=samples_per_video) if allow_duplicates else potential_frames
        else:
            frames_to_sample = random.choices(potential_frames, k=samples_per_video) if allow_duplicates else random.sample(potential_frames, samples_per_video)
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    # Convert to expected format (checkpoint, frame_number, action_labels)
    result = [(checkpoint, frame, action_labels) for frame, checkpoint, action_labels in frames_to_sample]
    
    return result 