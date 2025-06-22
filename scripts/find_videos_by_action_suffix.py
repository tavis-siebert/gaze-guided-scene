#!/usr/bin/env python
import sys
import csv
import argparse
import json
from collections import defaultdict


def load_action_mapping(file_path):
    """
    Load action mapping from CSV file.
    Returns a dictionary mapping (verb_id, noun_id) to action details.
    """
    action_map = {}
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            verb_id = int(row["verb_id"])
            noun_id = int(row["noun_id"])
            action_map[(verb_id, noun_id)] = {
                "verb_name": row["verb_name"],
                "noun_name": row["noun_name"],
                "action_description": row["action_description"],
                "ego_topo_action_id": int(row.get("ego_topo_action_id", -1)),
            }
    return action_map


def load_train_data(file_path):
    """
    Load training data from CSV file.
    Returns a dictionary mapping video names to lists of actions.
    """
    videos = defaultdict(list)
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            verb_id = int(row["label1"])
            noun_id = int(row["label2"])
            start_frame = int(row["start"])
            end_frame = int(row["end"])

            videos[name].append(
                {
                    "verb_id": verb_id,
                    "noun_id": noun_id,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "action_key": (verb_id, noun_id),
                }
            )

    # Sort actions by start frame for each video
    for name in videos:
        videos[name].sort(key=lambda x: x["start_frame"])

    return videos


def find_videos_with_action_suffix(videos, action_map, action_ids, suffix_length):
    """
    Find videos that contain the specified action suffix.

    Args:
        videos: Dictionary mapping video names to lists of actions
        action_map: Dictionary mapping (verb_id, noun_id) to action details
        action_ids: List of action IDs to match as a suffix
        suffix_length: Number of actions in the suffix to match

    Returns:
        List of matching video names and their matching action sequences
    """
    matches = []

    # Create a set of (verb_id, noun_id) tuples to match
    action_keys = []
    for action_id in action_ids:
        # Find the (verb_id, noun_id) for this action_id
        for key, details in action_map.items():
            if details.get("ego_topo_action_id") == action_id:
                action_keys.append(key)
                break

    # If we couldn't find all action keys, return empty list
    if len(action_keys) != len(action_ids):
        print(f"Warning: Could not find all action IDs in the mapping: {action_ids}")
        print(f"Found only: {action_keys}")
        return matches

    # Look for videos with matching action suffix
    for name, actions in videos.items():
        if len(actions) < suffix_length:
            continue

        # Check all possible ending positions
        for i in range(len(actions) - suffix_length + 1):
            sequence = actions[i : i + suffix_length]
            sequence_keys = [action["action_key"] for action in sequence]

            # Check if sequence matches our target action keys
            if action_keys == sequence_keys:
                matches.append(
                    {
                        "video_name": name,
                        "start_frame": sequence[0]["start_frame"],
                        "end_frame": sequence[-1]["end_frame"],
                        "actions": [
                            {
                                "verb_id": action["verb_id"],
                                "noun_id": action["noun_id"],
                                "verb_name": action_map[action["action_key"]][
                                    "verb_name"
                                ],
                                "noun_name": action_map[action["action_key"]][
                                    "noun_name"
                                ],
                                "description": action_map[action["action_key"]][
                                    "action_description"
                                ],
                                "start_frame": action["start_frame"],
                                "end_frame": action["end_frame"],
                            }
                            for action in sequence
                        ],
                    }
                )

    return matches


def main():
    parser = argparse.ArgumentParser(
        description="Find videos with specific action suffix"
    )
    parser.add_argument(
        "action_ids", type=int, nargs="+", help="List of action IDs to match as suffix"
    )
    parser.add_argument(
        "--suffix-length",
        type=int,
        default=None,
        help="Length of action suffix to match (default: length of action_ids)",
    )
    parser.add_argument(
        "--mapping-file",
        type=str,
        default="egtea_gaze/action_annotation/ego-topo-action-mapping.csv",
        help="Path to action mapping CSV file",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="train_S1.csv",
        help="Path to training data CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: print to stdout)",
    )
    args = parser.parse_args()

    # Use all provided action IDs if suffix length not specified
    suffix_length = (
        args.suffix_length if args.suffix_length is not None else len(args.action_ids)
    )

    # Load data
    try:
        action_map = load_action_mapping(args.mapping_file)
        videos = load_train_data(args.train_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the paths to the mapping and train files are correct.")
        sys.exit(1)

    # Find matching videos
    matches = find_videos_with_action_suffix(
        videos, action_map, args.action_ids, suffix_length
    )

    # Output results
    result = {
        "action_ids": args.action_ids,
        "suffix_length": suffix_length,
        "num_matches": len(matches),
        "matches": matches,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results written to {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
