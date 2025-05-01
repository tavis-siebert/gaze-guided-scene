#!/usr/bin/env python3
"""
Align dataset summary graphs to video frames using suffix mapping.
"""

import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional


def load_annotations(train_csv: str, test_csv: str) -> pd.DataFrame:
    """Load and concatenate train and test annotation CSVs."""
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    return pd.concat([df_train, df_test], ignore_index=True)


def build_suffix_map(ann_df: pd.DataFrame) -> Dict[Tuple[int, ...], List[Dict[str, Any]]]:
    """
    Build a mapping from suffix action sequences to video intervals.
    Key: tuple of future action_ids (ordered).
    Value: list of dicts with clip_name, frame_lower, frame_upper, time_lower, time_upper.
    """
    suffix_map: Dict[Tuple[int, ...], List[Dict[str, Any]]] = defaultdict(list)
    ann_df = ann_df.sort_values(['clip_name', 'start_frame']).reset_index(drop=True)
    for clip_name, group in ann_df.groupby('clip_name', sort=False):
        actions = group['action_id'].tolist()
        frames = group['start_frame'].tolist()
        times = group['start_time_fmt'].tolist()
        for idx in range(len(actions)):
            suffix = tuple(actions[idx:])
            if idx > 0:
                lower_frame = frames[idx-1]
                lower_time = times[idx-1]
            else:
                lower_frame = 0
                lower_time = None
            upper_frame = frames[idx]
            upper_time = times[idx]
            suffix_map[suffix].append({
                'clip_name': clip_name,
                'frame_lower': lower_frame,
                'frame_upper': upper_frame,
                'time_lower': lower_time,
                'time_upper': upper_time
            })
    return suffix_map


def align_dataset(
    dataset_df: pd.DataFrame,
    train_csv: str,
    test_csv: str
) -> pd.DataFrame:
    """
    Align each graph in dataset_df to its video frame interval.
    Adds columns: aligned_clip, frame_lower, frame_upper, time_lower,
                 time_upper, is_ambiguous.
    """
    ann_df = load_annotations(train_csv, test_csv)
    suffix_map = build_suffix_map(ann_df)

    # Initialize alignment columns
    for col in ('aligned_clip', 'frame_lower', 'frame_upper', 'time_lower', 'time_upper', 'is_ambiguous'):
        dataset_df[col] = None

    for idx, row in dataset_df.iterrows():
        suffix_str = row.get('future_actions_ordered')
        if pd.isna(suffix_str) or not suffix_str:
            continue
        suffix = tuple(int(x) for x in suffix_str.split(','))
        matches = suffix_map.get(suffix)
        if not matches:
            continue
        first = matches[0]
        dataset_df.at[idx, 'aligned_clip'] = first['clip_name']
        dataset_df.at[idx, 'frame_lower'] = first['frame_lower']
        dataset_df.at[idx, 'frame_upper'] = first['frame_upper']
        dataset_df.at[idx, 'time_lower'] = first['time_lower']
        dataset_df.at[idx, 'time_upper'] = first['time_upper']
        dataset_df.at[idx, 'is_ambiguous'] = len(matches) > 1

    return dataset_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Align dataset graphs to video frames')
    parser.add_argument('dataset_summary', help='CSV with dataset summary')
    parser.add_argument('aligned_output', help='Path to save aligned dataset CSV')
    parser.add_argument('--train-csv', required=True, help='Path to train annotation CSV')
    parser.add_argument('--test-csv', required=True, help='Path to test annotation CSV')
    args = parser.parse_args()
    df = pd.read_csv(args.dataset_summary)
    aligned = align_dataset(df, args.train_csv, args.test_csv)
    aligned.to_csv(args.aligned_output, index=False)
    print(f'Saved aligned dataset to {args.aligned_output}') 