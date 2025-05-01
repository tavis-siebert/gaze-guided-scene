#!/usr/bin/env python3
"""
Aligns graph snapshots from the dataset to their corresponding video frames
by matching future action sequences to the groundtruth action annotations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set


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


def build_action_suffixes(clip_dfs: Dict[str, pd.DataFrame]) -> Dict[str, Dict[int, List[int]]]:
    """
    Build action suffixes for each frame in each clip.
    
    Args:
        clip_dfs: Dictionary of DataFrames containing action data for each clip
    
    Returns:
        Dictionary mapping clip names to another dictionary mapping frame numbers
        to lists of action IDs that occur after that frame
    """
    clip_suffixes = {}
    
    for clip_name, clip_df in clip_dfs.items():
        frame_to_suffix = {}
        frames = sorted(clip_df['start_frame'].unique())
        
        # For each frame in the clip
        for i, frame in enumerate(frames):
            # Get all actions that start after this frame
            future_actions = clip_df[clip_df['start_frame'] >= frame]
            
            # Create the suffix as a list of action IDs
            suffix = future_actions['action_id'].tolist()
            
            # Store the suffix for this frame
            frame_to_suffix[frame] = suffix
            
            # Also store suffixes for frames between the current action and the next one
            if i < len(frames) - 1:
                next_frame = frames[i + 1]
                # For all frames between current and next action start
                for intermediate_frame in range(frame + 1, next_frame):
                    frame_to_suffix[intermediate_frame] = suffix
        
        clip_suffixes[clip_name] = frame_to_suffix
    
    return clip_suffixes


def align_graph_to_frames(
    graph_row: pd.Series, 
    clip_suffixes: Dict[str, Dict[int, List[int]]],
    exact_match: bool = False
) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """
    Find the clip and frame range that matches a graph's future actions.
    
    Args:
        graph_row: A row from the dataset summary DataFrame containing graph info
        clip_suffixes: Dictionary mapping clip names to frame suffixes
        exact_match: Whether to require exact matches (default: False)
    
    Returns:
        Tuple of (clip_name, frame_lower, frame_upper) or (None, None, None) if no match
    """
    # Extract future actions from the graph
    if pd.isna(graph_row['future_actions_ordered']):
        return None, None, None
    
    # Convert future_actions_ordered string to list of integers
    graph_future_actions = [int(x) for x in graph_row['future_actions_ordered'].split(',')]
    
    best_match = None
    best_match_score = 0
    best_frame_range = (None, None)
    
    # Go through each clip
    for clip_name, frame_to_suffix in clip_suffixes.items():
        # Check each frame in the clip
        for frame, suffix in frame_to_suffix.items():
            # Skip if suffix is empty
            if not suffix:
                continue
            
            # For exact matching
            if exact_match:
                if graph_future_actions == suffix:
                    return clip_name, frame, frame
            
            # For fuzzy matching, we'll use the longest common subsequence approach
            else:
                # Calculate match score (length of common prefix)
                match_length = 0
                for i in range(min(len(graph_future_actions), len(suffix))):
                    if graph_future_actions[i] == suffix[i]:
                        match_length += 1
                    else:
                        break
                
                # If we found a better match
                if match_length > best_match_score:
                    best_match_score = match_length
                    best_match = clip_name
                    
                    # Find the upper bound for the frame range
                    # The upper bound is the frame where the sequence of future actions changes
                    frame_upper = frame
                    for next_frame in sorted(frame_to_suffix.keys()):
                        if next_frame <= frame:
                            continue
                            
                        # If the suffix changes, we've found our upper bound
                        if frame_to_suffix[next_frame][:match_length] != suffix[:match_length]:
                            frame_upper = next_frame - 1
                            break
                    
                    best_frame_range = (frame, frame_upper)
    
    # If we found a match with at least one matching action
    if best_match_score > 0:
        return best_match, best_frame_range[0], best_frame_range[1]
    
    return None, None, None


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
    
    print("Building action suffixes for each frame...")
    clip_suffixes = build_action_suffixes(clip_dfs)
    
    print(f"Aligning {len(dataset_summary_df)} graphs to video frames...")
    
    # Create new columns for alignment information
    dataset_summary_df['aligned_clip'] = None
    dataset_summary_df['frame_lower'] = None
    dataset_summary_df['frame_upper'] = None
    dataset_summary_df['time_lower'] = None
    dataset_summary_df['time_upper'] = None
    
    # Process each graph in the dataset
    for idx, row in dataset_summary_df.iterrows():
        # Skip if no future actions
        if pd.isna(row['future_actions_ordered']):
            continue
            
        # Align the graph to a clip and frame range
        clip_name, frame_lower, frame_upper = align_graph_to_frames(row, clip_suffixes)
        
        if clip_name:
            # Update the DataFrame with alignment information
            dataset_summary_df.at[idx, 'aligned_clip'] = clip_name
            dataset_summary_df.at[idx, 'frame_lower'] = frame_lower
            dataset_summary_df.at[idx, 'frame_upper'] = frame_upper
            
            # Add timestamp information
            if clip_name in clip_dfs:
                clip_df = clip_dfs[clip_name]
                
                # Find the closest frames in the clip data
                lower_frame_data = clip_df[clip_df['start_frame'] <= frame_lower].iloc[-1] if not clip_df[clip_df['start_frame'] <= frame_lower].empty else None
                upper_frame_data = clip_df[clip_df['start_frame'] >= frame_upper].iloc[0] if not clip_df[clip_df['start_frame'] >= frame_upper].empty else None
                
                # Extract time information
                if lower_frame_data is not None:
                    dataset_summary_df.at[idx, 'time_lower'] = lower_frame_data['start_time_fmt']
                
                if upper_frame_data is not None:
                    dataset_summary_df.at[idx, 'time_upper'] = upper_frame_data['start_time_fmt']
    
    # Calculate alignment statistics
    aligned_count = dataset_summary_df['aligned_clip'].notna().sum()
    total_count = len(dataset_summary_df)
    print(f"Aligned {aligned_count}/{total_count} graphs ({aligned_count/total_count*100:.2f}%)")
    
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
    
    # For each split, print alignment stats
    for split in aligned_df['split'].unique():
        split_df = aligned_df[aligned_df['split'] == split]
        print(f"\n{split.capitalize()} split:")
        print(f"  Total graphs: {len(split_df)}")
        print(f"  Aligned graphs: {split_df['aligned_clip'].notna().sum()}")
        print(f"  Alignment rate: {split_df['aligned_clip'].notna().sum() / len(split_df) * 100:.2f}%")
    
    return aligned_df 