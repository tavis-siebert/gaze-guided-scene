#!/usr/bin/env python3
import os
import sys
import csv
import torch
from pathlib import Path
from collections import Counter


def load_mappings(test_split_file, main_mapping_file):
    """Load the action mappings from CSV files"""
    # Load test split mapping (verb_id, noun_id) -> ego_topo_action_id
    test_mapping = {}
    test_id_to_action = {}
    with open(test_split_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["verb_id"]), int(row["noun_id"]))
            test_id = int(row["ego_topo_action_id"])
            test_mapping[key] = test_id
            test_id_to_action[test_id] = row["action_description"]

    # Load main mapping (verb_id, noun_id) -> ego_topo_action_id
    main_mapping = {}
    main_id_to_action = {}
    with open(main_mapping_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["verb_id"]), int(row["noun_id"]))
            main_id = int(row["ego_topo_action_id"])
            main_mapping[key] = main_id
            main_id_to_action[main_id] = row["action_description"]

    # Create remapping dictionary: test_id -> main_id
    id_remapping = {}
    for key, test_id in test_mapping.items():
        if key in main_mapping:
            main_id = main_mapping[key]
            id_remapping[test_id] = main_id
        else:
            print(f"Warning: Verb-noun pair {key} not found in main mapping")

    return id_remapping, test_id_to_action, main_id_to_action


def remap_action_ids(dataset_path):
    """Remap action IDs in the validation split"""
    # Define paths
    base_dir = Path(dataset_path).parent
    filename = Path(dataset_path).name
    output_path = base_dir / f"{Path(filename).stem}_remapped.pth"

    test_split_file = (
        "egtea_gaze/action_annotation/ego-topo-action-mapping-test-split.csv"
    )
    main_mapping_file = "egtea_gaze/action_annotation/ego-topo-action-mapping.csv"

    # Load the mappings
    id_remapping, test_id_to_action, main_id_to_action = load_mappings(
        test_split_file, main_mapping_file
    )

    # Print some remapping examples
    print("\nRemapping Examples:")
    examples = list(id_remapping.items())[:5]  # Show first 5 examples
    for test_id, main_id in examples:
        print(
            f"  {test_id} ({test_id_to_action.get(test_id, 'Unknown')}) -> {main_id} ({main_id_to_action.get(main_id, 'Unknown')})"
        )

    print(f"\nTotal remappable IDs: {len(id_remapping)}")

    # Load the dataset
    print(f"\nLoading dataset from {dataset_path}")
    data = torch.load(dataset_path)

    # Process validation split
    val_data = data["val"]
    y_list = val_data["y"]

    # Count statistics for validation
    remapped_count = 0
    total_count = 0
    next_action_stats = Counter()

    # Find the maximum action ID in the main mapping to determine vector size
    max_main_id = max(main_id_to_action.keys()) if main_id_to_action else 0

    # Process each sample in validation set
    print(f"\nProcessing {len(y_list)} validation samples")
    for i, y in enumerate(y_list):
        # Remap next_action
        if "next_action" in y:
            total_count += 1
            old_id = y["next_action"].item()
            next_action_stats[old_id] += 1
            if old_id in id_remapping:
                y["next_action"] = torch.tensor(
                    id_remapping[old_id], dtype=y["next_action"].dtype
                )
                remapped_count += 1
            else:
                print(f"Warning: Action ID {old_id} not found in remapping")

        # Remap future_actions (multi-hot vector)
        if "future_actions" in y:
            future_actions = y["future_actions"]
            # Create a new vector with zeros
            new_future_actions = torch.zeros_like(future_actions)

            # Find indices where the original vector has 1s
            for old_id in range(len(future_actions)):
                if future_actions[old_id] == 1 and old_id in id_remapping:
                    new_main_id = id_remapping[old_id]
                    if new_main_id < len(new_future_actions):
                        new_future_actions[new_main_id] = 1

            y["future_actions"] = new_future_actions

        # Remap future_actions_ordered (sequence of action IDs)
        if "future_actions_ordered" in y:
            future_actions_ordered = y["future_actions_ordered"]
            new_future_actions_ordered = future_actions_ordered.clone()

            # Remap each action ID in the sequence
            for j in range(len(future_actions_ordered)):
                old_id = future_actions_ordered[j].item()
                if old_id in id_remapping:
                    new_future_actions_ordered[j] = id_remapping[old_id]

            y["future_actions_ordered"] = new_future_actions_ordered

    # Save the remapped dataset
    print(f"\nSaving remapped dataset to {output_path}")
    torch.save(data, output_path)

    # Print summary statistics
    print("\nRemapping Summary:")
    print(
        f"  Remapped {remapped_count}/{total_count} next_action IDs in validation set ({remapped_count / total_count * 100:.1f}%)"
    )

    print("\nTop 5 next_action IDs:")
    for action_id, count in next_action_stats.most_common(5):
        action_desc = test_id_to_action.get(action_id, "Unknown")
        print(f"  ID {action_id} ({action_desc}): {count} occurrences")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python remap_action_ids.py <dataset.pth>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file {dataset_path} not found")
        sys.exit(1)

    remap_action_ids(dataset_path)
