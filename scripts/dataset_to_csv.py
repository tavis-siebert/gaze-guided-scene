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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Counter
from collections import defaultdict

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
                # Convert to 1-indexed before adding to the set
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

def plot_distribution(data: Dict[str, Counter], title: str, output_path: str, 
                      top_n: int = 15, figsize: Tuple[int, int] = (12, 8)):
    """
    Plot distribution of categories.
    
    Args:
        data: Dictionary mapping split names to Counter objects
        title: Plot title
        output_path: Path to save the plot
        top_n: Number of top categories to show
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    num_splits = len(data)
    for i, (split_name, counter) in enumerate(data.items()):
        # Get the top N most common items
        top_items = counter.most_common(top_n)
        labels, values = zip(*top_items) if top_items else ([], [])
        
        # Create subplot for each split
        plt.subplot(num_splits, 1, i+1)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(labels))
        plt.barh(y_pos, values)
        plt.yticks(y_pos, labels)
        plt.title(f"{split_name.capitalize()} Split")
        plt.tight_layout()
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution plot to {output_path}")

def plot_edge_node_distribution(data: List[Dict], output_path: str):
    """
    Plot distribution of nodes and edges per graph.
    
    Args:
        data: List of dictionaries containing graph data
        output_path: Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Group data by split
    train_data = [d for d in data if d['split'] == 'train']
    val_data = [d for d in data if d['split'] == 'val']
    
    # Extract nodes and edges
    train_nodes = [d['num_nodes'] for d in train_data]
    train_edges = [d['num_edges'] for d in train_data]
    val_nodes = [d['num_nodes'] for d in val_data]
    val_edges = [d['num_edges'] for d in val_data]
    
    # Plot node distribution
    plt.subplot(2, 2, 1)
    sns.histplot(train_nodes, kde=True, color="blue", label="Train")
    plt.title("Node Distribution (Train)")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Frequency")
    
    plt.subplot(2, 2, 2)
    sns.histplot(val_nodes, kde=True, color="orange", label="Val")
    plt.title("Node Distribution (Val)")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Frequency")
    
    # Plot edge distribution
    plt.subplot(2, 2, 3)
    sns.histplot(train_edges, kde=True, color="blue", label="Train")
    plt.title("Edge Distribution (Train)")
    plt.xlabel("Number of Edges")
    plt.ylabel("Frequency")
    
    plt.subplot(2, 2, 4)
    sns.histplot(val_edges, kde=True, color="orange", label="Val")
    plt.title("Edge Distribution (Val)")
    plt.xlabel("Number of Edges")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved edge-node distribution plot to {output_path}")

def plot_edge_to_node_ratio(data: List[Dict], output_path: str):
    """
    Plot distribution of edge-to-node ratio.
    
    Args:
        data: List of dictionaries containing graph data
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Calculate edge-to-node ratios
    ratios = {}
    for split in ['train', 'val']:
        split_data = [d for d in data if d['split'] == split]
        ratios[split] = [d['num_edges'] / d['num_nodes'] if d['num_nodes'] > 0 else 0 
                         for d in split_data]
    
    # Plot density plot for each split
    for split, split_ratios in ratios.items():
        sns.kdeplot(split_ratios, label=split.capitalize(), fill=True)
    
    plt.title("Edge-to-Node Ratio Distribution")
    plt.xlabel("Edge-to-Node Ratio")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved edge-to-node ratio plot to {output_path}")

def dataset_to_csv(dataset_path: str, output_path: str, noun_idx_path: str, 
                   action_idx_path: str, analysis_dir: str):
    """
    Convert a PyTorch Geometric dataset to CSV format.
    
    Args:
        dataset_path: Path to the .pth dataset file
        output_path: Path to save the CSV output
        noun_idx_path: Path to the noun index mapping file
        action_idx_path: Path to the action index mapping file
        analysis_dir: Directory to save distribution plots
    """
    print(f"Loading dataset from {dataset_path}...")
    dataset = torch.load(dataset_path)
    
    # Load mappings
    noun_mapping = load_mapping(noun_idx_path)
    action_mapping = load_mapping(action_idx_path)
    
    rows = []
    
    # Counters for distributions
    noun_distribution = defaultdict(Counter)
    action_distribution = defaultdict(Counter)
    
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
            
            # Update noun distribution
            for idx in noun_indices:
                noun_name = noun_mapping.get(idx, f"Unknown-{idx}")
                noun_distribution[split][noun_name] += 1
            
            # Extract action information (convert from 0-indexed to 1-indexed)
            next_action_id = y["next_action"].item() if "next_action" in y else None
            next_action_name = None
            if next_action_id is not None:
                next_action_name = action_mapping.get(next_action_id, f"Unknown-{next_action_id}")
                # Update action distribution
                action_distribution[split][next_action_name] += 1
            
            # Handle future actions (convert from 0-indexed to 1-indexed)
            future_actions_str = ""
            future_actions_names = ""
            if "future_actions" in y:
                # Convert binary tensor to list of indices
                future_actions = torch.nonzero(y["future_actions"]).squeeze(1).tolist()
                future_actions_str = ",".join(str(idx) for idx in future_actions)
                future_actions_names = "|".join(action_mapping.get(idx, f"Unknown-{idx}") for idx in future_actions)
                
                # Update action distribution
                for idx in future_actions:
                    action_name = action_mapping.get(idx, f"Unknown-{idx}")
                    action_distribution[split][action_name] += 1
            
            # Handle ordered future actions (convert from 0-indexed to 1-indexed)
            future_actions_ordered_str = ""
            future_actions_ordered_names = ""
            if "future_actions_ordered" in y:
                ordered_actions = y["future_actions_ordered"].tolist()
                future_actions_ordered_str = ",".join(str(idx) for idx in ordered_actions)
                future_actions_ordered_names = "|".join(action_mapping.get(idx, f"Unknown-{idx}") for idx in ordered_actions)
            
            # Create row
            row = {
                "split": split,
                "id_in_split": i,
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "bidirectional": bidirectional,
                "node_noun_indices": noun_idx_str,
                "node_noun_labels": noun_names,
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
    
    # Generate distribution plots
    if analysis_dir:
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Plot noun distribution
        plot_distribution(
            noun_distribution, 
            "Noun Distribution in Graphs",
            os.path.join(analysis_dir, "noun_distribution.png")
        )
        
        # Plot action distribution
        plot_distribution(
            action_distribution, 
            "Action Distribution in Graphs",
            os.path.join(analysis_dir, "action_distribution.png"),
            top_n=20
        )
        
        # Plot node and edge distribution
        plot_edge_node_distribution(
            rows,
            os.path.join(analysis_dir, "node_edge_distribution.png")
        )
        
        # Plot edge-to-node ratio
        plot_edge_to_node_ratio(
            rows,
            os.path.join(analysis_dir, "edge_node_ratio.png")
        )
        
        print(f"All distribution plots saved to {analysis_dir}")

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch Geometric dataset to CSV')
    parser.add_argument('dataset_path', help='Path to the .pth dataset file')
    parser.add_argument('-o', '--output', help='Output CSV file path (default: out/{dataset_name}/dataset_summary.csv)')
    parser.add_argument('--noun-mapping', help='Path to noun_idx.txt (default: egtea_gaze/action_annotation/noun_idx.txt)')
    parser.add_argument('--action-mapping', help='Path to action_idx.txt (default: egtea_gaze/action_annotation/action_idx.txt)')
    parser.add_argument('--analysis-dir', help='Directory to save distribution plots (default: out/{dataset_name}/analysis)')
    
    args = parser.parse_args()
    
    # Extract dataset name from path
    dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
    
    # Create dataset-specific output directory
    dataset_output_dir = os.path.join("out", dataset_name)
    
    # Set defaults
    output_path = args.output or os.path.join(dataset_output_dir, "dataset_summary.csv")
    noun_mapping = args.noun_mapping or "egtea_gaze/action_annotation/noun_idx.txt"
    action_mapping = args.action_mapping or "egtea_gaze/action_annotation/action_idx.txt"
    analysis_dir = args.analysis_dir or os.path.join(dataset_output_dir, "analysis")
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
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
    dataset_to_csv(args.dataset_path, output_path, noun_mapping, action_mapping, analysis_dir)

if __name__ == "__main__":
    main() 