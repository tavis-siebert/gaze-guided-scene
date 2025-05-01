#!/usr/bin/env python3
"""
Aligns graph snapshots from the dataset to their corresponding video frames
by matching future action sequences to the groundtruth action annotations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class SuffixNode:
    """Node in the suffix graph representing an action in a sequence."""
    action_id: int
    clips: Dict[str, List[Tuple[int, int, str, Optional[str]]]] = field(default_factory=dict)
    next_actions: Dict[int, 'SuffixNode'] = field(default_factory=dict)
    
    
class MatchResult(NamedTuple):
    """Result of matching a graph to a suffix."""
    clip_name: str
    frame_lower: int
    frame_upper: int
    time_lower: Optional[str]
    time_upper: Optional[str]
    is_ambiguous: bool


def load_action_data(train_csv_path: str, test_csv_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load and preprocess clip action data from train and test CSVs.
    
    Args:
        train_csv_path: Path to the training split CSV file
        test_csv_path: Path to the test split CSV file
        
    Returns:
        Dictionary containing DataFrames for each unique clip
    """
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    # Combine train and test data
    combined_df = pd.concat([train_df, test_df])
    
    # Group by clip_name
    clip_dfs = {}
    for clip_name, clip_df in combined_df.groupby('clip_name'):
        # Sort by start_frame to ensure actions are in chronological order
        clip_dfs[clip_name] = clip_df.sort_values('start_frame')
    
    return clip_dfs


def build_suffix_graph(clip_dfs: Dict[str, pd.DataFrame]) -> Dict[int, SuffixNode]:
    """
    Build a suffix graph for all clips by traversing each clip from the back.
    
    Args:
        clip_dfs: Dictionary of DataFrames containing action data for each clip
    
    Returns:
        Dictionary mapping action IDs to root nodes in the suffix graph
    """
    root_nodes = {}  # Maps action_id to SuffixNode for the root (last) actions
    
    for clip_name, clip_df in clip_dfs.items():
        # Sort by start_frame in descending order to process from the end of the clip
        clip_records = clip_df.sort_values('start_frame', ascending=False).to_dict('records')
        
        if not clip_records:
            continue
            
        # For each action in the clip (from last to first)
        prev_node = None
        prev_record = None
        
        for i, record in enumerate(clip_records):
            action_id = record['action_id']
            frame_upper = record['start_frame']
            time_upper = record['start_time_fmt']
            
            # Get or create the node for this action
            if action_id not in root_nodes:
                root_nodes[action_id] = SuffixNode(action_id=action_id)
            
            current_node = root_nodes[action_id]
            
            # Set frame bounds
            if i == 0:  # Last action in clip
                frame_lower = frame_upper
                time_lower = time_upper
            else:  # Actions in the middle
                frame_lower = prev_record['start_frame']
                time_lower = prev_record['start_time_fmt']
            
            # Add clip info to this node
            if clip_name not in current_node.clips:
                current_node.clips[clip_name] = []
            current_node.clips[clip_name].append((frame_lower, frame_upper, time_lower, time_upper))
            
            # Connect to the previous node (if this isn't the last action)
            if prev_node:
                current_node.next_actions[prev_record['action_id']] = prev_node
            
            # Update previous node for next iteration
            prev_node = current_node
            prev_record = record
    
    return root_nodes


def match_graph_to_suffix(
    graph_future_actions: List[int],
    suffix_graph: Dict[int, SuffixNode]
) -> List[MatchResult]:
    """
    Find clips and frame ranges that match a graph's future actions by traversing the suffix graph.
    
    Args:
        graph_future_actions: List of action IDs in the future actions sequence
        suffix_graph: Dictionary mapping action IDs to root nodes in the suffix graph
    
    Returns:
        List of MatchResults for matching clips and frames
    """
    if not graph_future_actions:
        return []
    
    # Start with the last action in the future actions list
    last_action_id = graph_future_actions[-1]
    
    if last_action_id not in suffix_graph:
        return []
    
    # Start from the root node for the last action
    current_node = suffix_graph[last_action_id]
    
    # Track possible paths through the graph
    paths = [(current_node, len(graph_future_actions) - 1)]
    complete_paths = []
    
    # Traverse the graph backward, following the future actions in reverse
    while paths:
        node, action_idx = paths.pop()
        
        # If we've reached the first action in the sequence
        if action_idx == 0:
            # Add all possible clip matches from this node
            for clip_name, frame_ranges in node.clips.items():
                for frame_lower, frame_upper, time_lower, time_upper in frame_ranges:
                    complete_paths.append((clip_name, frame_lower, frame_upper, time_lower, time_upper))
            continue
        
        # Try to follow the next action in the sequence (moving backward)
        next_action_id = graph_future_actions[action_idx - 1]
        
        if next_action_id in node.next_actions:
            # Follow this path
            next_node = node.next_actions[next_action_id]
            paths.append((next_node, action_idx - 1))
    
    # Convert paths to MatchResults, marking ambiguous results
    if not complete_paths:
        return []
    
    # Group by clip to detect ambiguity
    clips_count = defaultdict(int)
    for path in complete_paths:
        clips_count[path[0]] += 1
    
    matches = []
    for clip_name, frame_lower, frame_upper, time_lower, time_upper in complete_paths:
        is_ambiguous = len(complete_paths) > 1
        matches.append(MatchResult(
            clip_name=clip_name,
            frame_lower=frame_lower,
            frame_upper=frame_upper,
            time_lower=time_lower,
            time_upper=time_upper,
            is_ambiguous=is_ambiguous
        ))
    
    return matches


def align_graph_to_frames(
    graph_row: pd.Series,
    suffix_graph: Dict[int, SuffixNode]
) -> Optional[MatchResult]:
    """
    Find the clip and frame range that matches a graph's future actions.
    
    Args:
        graph_row: A row from the dataset summary DataFrame containing graph info
        suffix_graph: Dictionary mapping action IDs to root nodes in the suffix graph
    
    Returns:
        MatchResult or None if no match
    """
    # Extract future actions from the graph
    if pd.isna(graph_row['future_actions_ordered']):
        return None
    
    # Convert future_actions_ordered string to list of integers
    graph_future_actions = [int(x) for x in graph_row['future_actions_ordered'].split(',')]
    
    # Match the graph to the suffix graph
    matches = match_graph_to_suffix(graph_future_actions, suffix_graph)
    
    # Return the first match (if any)
    return matches[0] if matches else None


def align_dataset(
    dataset_summary_df: pd.DataFrame,
    train_csv_path: str,
    test_csv_path: str
) -> pd.DataFrame:
    """
    Align each graph in the dataset to its corresponding clip and frame range.
    
    Args:
        dataset_summary_df: DataFrame containing dataset summary
        train_csv_path: Path to the training split CSV file
        test_csv_path: Path to the test split CSV file
    
    Returns:
        DataFrame with additional columns for clip alignment information
    """
    print("Loading action data from CSV files...")
    clip_dfs = load_action_data(train_csv_path, test_csv_path)
    
    print("Building suffix graph from action data...")
    suffix_graph = build_suffix_graph(clip_dfs)
    
    print(f"Aligning {len(dataset_summary_df)} graphs to video frames...")
    
    # Create new columns for alignment information
    dataset_summary_df['aligned_clip'] = None
    dataset_summary_df['frame_lower'] = None
    dataset_summary_df['frame_upper'] = None
    dataset_summary_df['time_lower'] = None
    dataset_summary_df['time_upper'] = None
    dataset_summary_df['is_ambiguous'] = False
    
    # Process each graph in the dataset
    for idx, row in dataset_summary_df.iterrows():
        # Skip if no future actions
        if pd.isna(row['future_actions_ordered']):
            continue
            
        # Align the graph to a clip and frame range
        match_result = align_graph_to_frames(row, suffix_graph)
        
        if match_result:
            # Update the DataFrame with alignment information
            dataset_summary_df.at[idx, 'aligned_clip'] = match_result.clip_name
            dataset_summary_df.at[idx, 'frame_lower'] = match_result.frame_lower
            dataset_summary_df.at[idx, 'frame_upper'] = match_result.frame_upper
            dataset_summary_df.at[idx, 'time_lower'] = match_result.time_lower
            dataset_summary_df.at[idx, 'time_upper'] = match_result.time_upper
            dataset_summary_df.at[idx, 'is_ambiguous'] = match_result.is_ambiguous
    
    # Calculate alignment statistics
    aligned_count = dataset_summary_df['aligned_clip'].notna().sum()
    ambiguous_count = dataset_summary_df['is_ambiguous'].sum()
    total_count = len(dataset_summary_df)
    print(f"Aligned {aligned_count}/{total_count} graphs ({aligned_count/total_count*100:.2f}%)")
    print(f"Ambiguous alignments: {ambiguous_count}/{aligned_count} ({ambiguous_count/aligned_count*100:.2f}% of aligned)")
    
    return dataset_summary_df


def main(dataset_summary_path: str, output_path: str, train_csv_path: str, test_csv_path: str):
    """
    Main function to align dataset graphs to video frames.
    
    Args:
        dataset_summary_path: Path to the dataset summary CSV file
        output_path: Path to save the aligned dataset CSV file
        train_csv_path: Path to the training split CSV file
        test_csv_path: Path to the test split CSV file
    """
    print(f"Loading dataset summary from {dataset_summary_path}...")
    dataset_summary_df = pd.read_csv(dataset_summary_path)
    
    # Align the dataset
    aligned_df = align_dataset(dataset_summary_df, train_csv_path, test_csv_path)
    
    # Save the aligned dataset
    print(f"Saving aligned dataset to {output_path}...")
    aligned_df.to_csv(output_path, index=False)
    
    # Print alignment statistics
    print("\nAlignment Statistics:")
    print(f"Total graphs: {len(aligned_df)}")
    print(f"Aligned graphs: {aligned_df['aligned_clip'].notna().sum()}")
    print(f"Alignment rate: {aligned_df['aligned_clip'].notna().sum() / len(aligned_df) * 100:.2f}%")
    print(f"Ambiguous alignments: {aligned_df['is_ambiguous'].sum()}")
    print(f"Ambiguous rate: {aligned_df['is_ambiguous'].sum() / aligned_df['aligned_clip'].notna().sum() * 100:.2f}% of aligned")
    
    # For each split, print alignment stats
    for split in aligned_df['split'].unique():
        split_df = aligned_df[aligned_df['split'] == split]
        print(f"\n{split.capitalize()} split:")
        print(f"  Total graphs: {len(split_df)}")
        print(f"  Aligned graphs: {split_df['aligned_clip'].notna().sum()}")
        print(f"  Alignment rate: {split_df['aligned_clip'].notna().sum() / len(split_df) * 100:.2f}%")
        print(f"  Ambiguous alignments: {split_df['is_ambiguous'].sum()}")
        print(f"  Ambiguous rate: {split_df['is_ambiguous'].sum() / split_df['aligned_clip'].notna().sum() * 100:.2f}% of aligned")
    
    return aligned_df 