"""
Sampling functionality for graph datasets.

This module provides sampling strategies for graph datasets, including:
- Uniform sampling from checkpoints
- Random sampling from checkpoints
- Oversampling from all frames, using the latest checkpoint
- Action recognition sampling with temporal alignment
"""

import random
from typing import Dict, List, Tuple
import numpy as np

from gazegraph.graph.checkpoint_manager import GraphCheckpoint
from gazegraph.datasets.egtea_gaze.video_metadata import VideoMetadata
from gazegraph.training.dataset.action_recognition_sampling import (
    get_action_recognition_samples,
)
from gazegraph.logger import get_logger

logger = get_logger(__name__)


def get_samples(
    checkpoints: List[GraphCheckpoint],
    video_name: str,
    strategy: str,
    samples_per_video: int,
    allow_duplicates: bool,
    oversampling: bool,
    metadata: VideoMetadata,
    task_mode: str = "future_actions",
    **kwargs,
) -> List[Tuple[GraphCheckpoint, Dict]]:
    """Sample checkpoints according to the specified strategy.

    Args:
        checkpoints: List of checkpoints to sample from
        video_name: Name of the video
        strategy: Sampling strategy ('all', 'uniform', 'random', or action recognition strategies)
        samples_per_video: Number of samples per video
        allow_duplicates: Whether to allow duplicate samples
        oversampling: Whether to sample from all frames (not just checkpoint frames)
        metadata: VideoMetadata object to get valid frame ranges
        task_mode: Task mode for determining sampling approach
        **kwargs: Additional parameters for specialized sampling

    Returns:
        List of tuples (checkpoint, action_labels), where action_labels is pre-computed labels.

    Raises:
        ValueError: If the strategy is not recognized
    """
    if not checkpoints:
        return []

    checkpoints_sorted = sorted(checkpoints, key=lambda x: x.frame_number)

    # Handle action recognition sampling
    if task_mode == "action_recognition":
        return get_action_recognition_samples(
            checkpoints=checkpoints_sorted,
            video_name=video_name,
            samples_per_action=samples_per_video,
            metadata=metadata,
            **kwargs,
        )

    # Handle standard future/next action sampling
    if oversampling:
        logger.info("Sampling with oversampling")
        try:
            start_frame, end_frame = metadata.get_action_frame_range(video_name)
            valid_frames = range(start_frame, end_frame + 1)
            potential = _potential_from_oversampling(
                checkpoints_sorted, valid_frames, metadata
            )
        except ValueError:
            logger.info("Falling back to checkpoint sampling due to metadata error.")
            potential = _potential_from_checkpoints(checkpoints_sorted, metadata)
    else:
        logger.info("Sampling without oversampling")
        potential = _potential_from_checkpoints(checkpoints_sorted, metadata)

    return _sample_potential(potential, strategy, samples_per_video, allow_duplicates)


def _potential_from_checkpoints(
    checkpoints: List[GraphCheckpoint], metadata: VideoMetadata
):
    """Build all valid (checkpoint, labels) pairs from checkpoints."""
    return [
        (cp, labels)
        for cp in checkpoints
        if (labels := cp.get_future_action_labels(cp.frame_number, metadata)) and labels
    ]


def _potential_from_oversampling(
    checkpoints: List[GraphCheckpoint], valid_frames, metadata: VideoMetadata
):
    """Build all valid (checkpoint, labels) pairs for oversampling frames."""
    potential = []
    for frame in valid_frames:
        suitable = [cp for cp in checkpoints if cp.frame_number <= frame]
        if not suitable:
            continue
        cp = suitable[-1]
        labels = cp.get_future_action_labels(frame, metadata)
        if labels:
            potential.append((cp, labels))
    return potential


def _sample_potential(potential, strategy, samples_per_video, allow_duplicates):
    """Sample from a list of (checkpoint, labels) pairs according to strategy."""
    if not potential:
        return []
    if strategy == "all" or samples_per_video <= 0:
        return potential
    if strategy == "uniform":
        if samples_per_video >= len(potential):
            return (
                random.choices(potential, k=samples_per_video)
                if allow_duplicates
                else potential
            )
        indices = np.linspace(
            0, len(potential) - 1, samples_per_video, dtype=int
        ).tolist()
        return [potential[i] for i in indices]
    if strategy == "random":
        if samples_per_video >= len(potential):
            return (
                random.choices(potential, k=samples_per_video)
                if allow_duplicates
                else potential
            )
        return (
            random.choices(potential, k=samples_per_video)
            if allow_duplicates
            else random.sample(potential, samples_per_video)
        )
    raise ValueError(f"Unknown sampling strategy: {strategy}")
