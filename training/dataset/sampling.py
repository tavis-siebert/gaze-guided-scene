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
) -> List[Tuple[GraphCheckpoint, int]]:
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
        List of tuples (checkpoint, frame_number), where frame_number is the actual frame to use
        for label generation. When oversampling is False, this will be the same as checkpoint.frame_number
    
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
            checkpoints_sorted, strategy, samples_per_video, allow_duplicates
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
    allow_duplicates: bool
) -> List[Tuple[GraphCheckpoint, int]]:
    """Sample from available checkpoint frames without oversampling.
    
    Args:
        checkpoints: Sorted list of checkpoints to sample from
        strategy: Sampling strategy ('all', 'uniform', or 'random')
        samples_per_video: Number of samples per video
        allow_duplicates: Whether to allow duplicate samples
    
    Returns:
        List of tuples (checkpoint, frame_number)
    
    Raises:
        ValueError: If the strategy is not recognized
    """
    n = len(checkpoints)
    
    # 'all' strategy: use all checkpoints
    if strategy == 'all' or samples_per_video <= 0:
        return [(cp, cp.frame_number) for cp in checkpoints]
    
    # 'uniform' strategy: sample evenly spaced checkpoints
    if strategy == 'uniform':
        if samples_per_video >= n:
            checkpoints_to_use = random.choices(checkpoints, k=samples_per_video) if allow_duplicates else checkpoints
        else:
            indices = np.linspace(0, n - 1, samples_per_video, dtype=int).tolist()
            checkpoints_to_use = [checkpoints[i] for i in indices]
    
    # 'random' strategy: sample random checkpoints
    elif strategy == 'random':
        if samples_per_video >= n:
            checkpoints_to_use = random.choices(checkpoints, k=samples_per_video) if allow_duplicates else checkpoints
        else:
            checkpoints_to_use = random.choices(checkpoints, k=samples_per_video) if allow_duplicates else random.sample(checkpoints, samples_per_video)
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    return [(cp, cp.frame_number) for cp in checkpoints_to_use]


def _sample_with_oversampling(
    checkpoints: List[GraphCheckpoint],
    video_name: str,
    strategy: str,
    samples_per_video: int,
    allow_duplicates: bool,
    metadata: VideoMetadata
) -> List[Tuple[GraphCheckpoint, int]]:
    """Sample from all valid frames using oversampling.
    
    Args:
        checkpoints: Sorted list of checkpoints to sample from
        video_name: Name of the video
        strategy: Sampling strategy ('all', 'uniform', or 'random')
        samples_per_video: Number of samples per video
        allow_duplicates: Whether to allow duplicate samples
        metadata: VideoMetadata object to get valid frame ranges
    
    Returns:
        List of tuples (checkpoint, frame_number)
    
    Raises:
        ValueError: If the strategy is not recognized
    """
    # Get valid frame range for the video
    try:
        start_frame, end_frame = metadata.get_action_frame_range(video_name)
    except ValueError:
        # Fall back to regular sampling if no action records are found
        return _sample_from_checkpoints(checkpoints, strategy, samples_per_video, allow_duplicates)
    
    # All valid frames for sampling
    valid_frames = list(range(start_frame, end_frame + 1))
    
    # 'all' strategy: use all valid frames
    if strategy == 'all' or samples_per_video <= 0:
        frames_to_sample = valid_frames
    
    # 'uniform' strategy: sample evenly spaced frames
    elif strategy == 'uniform':
        n_frames = len(valid_frames)
        if samples_per_video >= n_frames:
            frames_to_sample = random.choices(valid_frames, k=samples_per_video) if allow_duplicates else valid_frames
        else:
            indices = np.linspace(0, n_frames - 1, samples_per_video, dtype=int).tolist()
            frames_to_sample = [valid_frames[i] for i in indices]
    
    # 'random' strategy: sample random frames
    elif strategy == 'random':
        n_frames = len(valid_frames)
        if samples_per_video >= n_frames:
            frames_to_sample = random.choices(valid_frames, k=samples_per_video) if allow_duplicates else valid_frames
        else:
            frames_to_sample = random.choices(valid_frames, k=samples_per_video) if allow_duplicates else random.sample(valid_frames, samples_per_video)
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    # Find the best checkpoint for each frame (the latest checkpoint before the frame)
    result = []
    for frame in frames_to_sample:
        # Find the most recent checkpoint that is at or before the current frame
        suitable_checkpoints = [cp for cp in checkpoints if cp.frame_number <= frame]
        if not suitable_checkpoints:
            # If no suitable checkpoint is found, skip this frame
            continue
        
        # Get the most recent checkpoint
        checkpoint = suitable_checkpoints[-1]
        result.append((checkpoint, frame))
    
    return result 