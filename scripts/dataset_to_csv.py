#!/usr/bin/env python3
"""
Converts a PyTorch Geometric dataset file to CSV format with human-readable
labels for actions and objects from EGTEA Gaze+ dataset.
"""

import sys
import os
import torch
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

def load_mapping(file_path: str) -> Dict[int, str]:
    """Load index to name mapping from file."""
    mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Extract name and index from line
            parts = line.rsplit(' ', 1)
            if len(parts) == 2:
                name, idx_str = parts
                try:
                    idx = int(idx_str)
                    mapping[idx] = name
                except ValueError:
                    print(f"Warning: Invalid line format in {file_path}: {line}")
    return mapping

def extract_noun_indices(node_features: torch.Tensor, num_object_classes: int) -> Set[int]:
    """Extract unique noun indices from node features."""
    noun_indices = set()
    
    # Assuming node features format is [temporal_features, one_hot_encoding]
    # where temporal_features has length 5 and one_hot_encoding has length num_object_classes
    for node in node_features:
        one_hot = node[5:5+num_object_classes]
        if torch.any(one_hot > 0):
            class_idx = torch.argmax(one_hot).item()
            if class_idx > 0:  # Ignore background/unknown class (0)
                noun_indices.add(class_idx)
    
    return noun_indices

def is_bidirectional(edge_index: torch.Tensor) -> bool:
    """Check if all edges are bidirectional."""
    if edge_index.size(1) == 0:
        return True
        
    # Create a set of all edges as tuples
    edges = set()
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        edges.add((src, dst))
    
    # Check if all edges have their reverse
    for src, dst in edges:
        if (dst, src) not in edges:
            return False
    
    return True

def dataset_to_csv(dataset_path: str, output_path: str, noun_idx_path: str, action_idx_path: str):
    """
    Convert a PyTorch Geometric dataset to CSV format.
    
    Args:
        dataset_path: Path to the .pth dataset file
        output_path: Path to save the CSV output
        noun_idx_path: Path to the noun index mapping file
        action_idx_path: Path to the action index mapping file
    """
    print(f"Loading dataset from {dataset_path}...")
    dataset = torch.load(dataset_path)
    
    # Load mappings
    noun_mapping = load_mapping(noun_idx_path)
    action_mapping = load_mapping(action_idx_path)
    
    rows = []
    
    # Process each split
    for split in ['train', 'val']:
        if split not in dataset:
            print(f"Warning: '{split}' split not found in dataset")
            continue
            
        x_list = dataset[split]['x']
        edge_index_list = dataset[split]['edge_index']
        edge_attr_list = dataset[split]['edge_attr']
        y_list = dataset[split]['y']
        
        print(f"Processing {split} split with {len(x_list)} graphs...")
        
        # Determine num_object_classes from the first node tensor
        if len(x_list) > 0:
            # Assuming the temporal features are of length 5
            # The rest of the features are one-hot encoding of object classes
            num_object_classes = x_list[0].shape[1] - 5
        else:
            num_object_classes = 53  # Default based on noun_idx.txt
        
        # Process each graph
        for i, (x, edge_index, edge_attr, y) in enumerate(zip(x_list, edge_index_list, edge_attr_list, y_list)):
            # Extract basic graph stats
            num_nodes = x.shape[0]
            num_edges = edge_index.shape[1]
            bidirectional = is_bidirectional(edge_index)
            
            # Extract noun indices from node features
            noun_indices = extract_noun_indices(x, num_object_classes)
            
            # Convert noun indices to strings
            noun_idx_str = ",".join(str(idx) for idx in sorted(noun_indices))
            noun_names = ",".join(noun_mapping.get(idx, f"Unknown-{idx}") for idx in sorted(noun_indices))
            
            # Extract action information
            next_action_id = y["next_action"].item() if "next_action" in y else None
            next_action_name = action_mapping.get(next_action_id, f"Unknown-{next_action_id}") if next_action_id is not None else None
            
            # Handle future actions
            future_actions_str = ""
            future_actions_names = ""
            if "future_actions" in y:
                # Convert binary tensor to list of indices
                future_actions = torch.nonzero(y["future_actions"]).squeeze(1).tolist()
                future_actions_str = ",".join(str(idx) for idx in future_actions)
                future_actions_names = "|".join(action_mapping.get(idx, f"Unknown-{idx}") for idx in future_actions)
            
            # Handle ordered future actions
            future_actions_ordered_str = ""
            future_actions_ordered_names = ""
            if "future_actions_ordered" in y:
                ordered_actions = y["future_actions_ordered"].tolist()
                future_actions_ordered_str = ",".join(str(idx) for idx in ordered_actions)
                future_actions_ordered_names = "|".join(action_mapping.get(idx, f"Unknown-{idx}") for idx in ordered_actions)
            
            # Create row
            row = {
                "split": split,
                "id": i,
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "bidirectional": bidirectional,
                "noun_indices": noun_idx_str,
                "noun_names": noun_names,
                "next_action_id": next_action_id,
                "next_action_name": next_action_name,
                "future_actions": future_actions_str,
                "future_actions_names": future_actions_names,
                "future_actions_ordered": future_actions_ordered_str,
                "future_actions_ordered_names": future_actions_ordered_names
            }
            
            rows.append(row)
    
    # Write to CSV
    if rows:
        fieldnames = rows[0].keys()
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"Successfully wrote {len(rows)} records to {output_path}")
    else:
        print("No data to write")

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch Geometric dataset to CSV')
    parser.add_argument('dataset_path', help='Path to the .pth dataset file')
    parser.add_argument('-o', '--output', help='Output CSV file path (default: dataset_summary.csv)')
    parser.add_argument('--noun-mapping', help='Path to noun_idx.txt (default: egtea_gaze/action_annotation/noun_idx.txt)')
    parser.add_argument('--action-mapping', help='Path to action_idx.txt (default: egtea_gaze/action_annotation/action_idx.txt)')
    
    args = parser.parse_args()
    
    # Set defaults
    output_path = args.output or os.path.join(os.path.dirname(args.dataset_path), "dataset_summary.csv")
    noun_mapping = args.noun_mapping or "egtea_gaze/action_annotation/noun_idx.txt"
    action_mapping = args.action_mapping or "egtea_gaze/action_annotation/action_idx.txt"
    
    # Validate file paths
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset file not found: {args.dataset_path}")
        sys.exit(1)
    
    if not os.path.exists(noun_mapping):
        print(f"Error: Noun mapping file not found: {noun_mapping}")
        sys.exit(1)
        
    if not os.path.exists(action_mapping):
        print(f"Error: Action mapping file not found: {action_mapping}")
        sys.exit(1)
    
    # Convert dataset to CSV
    dataset_to_csv(args.dataset_path, output_path, noun_mapping, action_mapping)

if __name__ == "__main__":
    main() 