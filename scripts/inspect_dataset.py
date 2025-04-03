# load dataset from ../datasets/dataset.pth

import torch
import argparse
from typing import Any, Dict, List, Set, Tuple, Union
import numpy as np


def explore_dataset_structure(obj: Any, prefix: str = "", max_depth: int = 10, 
                             current_depth: int = 0, visited: Set[int] = None) -> None:
    """
    Recursively explore and pretty print the structure of a PyTorch dataset.
    
    Args:
        obj: The object to explore
        prefix: String prefix for the current exploration level
        max_depth: Maximum recursion depth to prevent infinite loops
        current_depth: Current recursion depth
        visited: Set of object ids already visited to prevent circular references
    """
    if visited is None:
        visited = set()
    
    # Prevent infinite recursion
    if current_depth > max_depth:
        print(f"{prefix}[MAX DEPTH REACHED]")
        return
    
    # Handle circular references
    obj_id = id(obj)
    if obj_id in visited:
        print(f"{prefix}[CIRCULAR REFERENCE]")
        return
    
    # Add current object to visited set for complex types
    if isinstance(obj, (dict, list, tuple, set, torch.Tensor, np.ndarray)):
        visited.add(obj_id)
    
    # Print information based on object type
    if obj is None:
        print(f"{prefix}None")
    elif isinstance(obj, (int, float, str, bool)):
        print(f"{prefix}Type: {type(obj).__name__}, Value: {obj}")
    elif isinstance(obj, dict):
        print(f"{prefix}Dict with {len(obj)} keys:")
        for key in obj:
            print(f"{prefix}  Key: {key}")
            explore_dataset_structure(obj[key], f"{prefix}    ", 
                                     max_depth, current_depth + 1, visited)
    elif isinstance(obj, (list, tuple)):
        container_type = type(obj).__name__
        print(f"{prefix}{container_type} with {len(obj)} items:")
        if len(obj) > 0:
            print(f"{prefix}  First item:")
            explore_dataset_structure(obj[0], f"{prefix}    ", 
                                     max_depth, current_depth + 1, visited)
            if len(obj) > 1:
                print(f"{prefix}  ... ({len(obj)-1} more items)")
    elif isinstance(obj, torch.Tensor):
        print(f"{prefix}Tensor: shape={obj.shape}, dtype={obj.dtype}")
        if obj.numel() > 0 and obj.numel() < 10:
            print(f"{prefix}  Value: {obj}")
        elif obj.numel() > 0:
            flat = obj.flatten()
            print(f"{prefix}  Sample values: {flat[:3].tolist()}...")
    elif isinstance(obj, np.ndarray):
        print(f"{prefix}NumPy Array: shape={obj.shape}, dtype={obj.dtype}")
        if obj.size > 0 and obj.size < 10:
            print(f"{prefix}  Value: {obj}")
        elif obj.size > 0:
            flat = obj.flatten()
            print(f"{prefix}  Sample values: {flat[:3].tolist()}...")
    else:
        print(f"{prefix}Object of type: {type(obj).__name__}")
        # Try to get additional info for unknown objects
        try:
            if hasattr(obj, "__dict__"):
                print(f"{prefix}  Attributes:")
                for key, value in obj.__dict__.items():
                    print(f"{prefix}    {key}:")
                    explore_dataset_structure(value, f"{prefix}      ", 
                                           max_depth, current_depth + 1, visited)
        except:
            pass


def analyze_dataset(dataset_path: str, max_depth: int = 5) -> None:
    """
    Load and analyze a PyTorch dataset structure.
    
    Args:
        dataset_path: Path to the .pth dataset file
        max_depth: Maximum recursion depth for exploration
    """
    print(f"Loading dataset from: {dataset_path}")
    try:
        dataset = torch.load(dataset_path)
        print("\nDataset Structure:")
        explore_dataset_structure(dataset, max_depth=max_depth)
    except Exception as e:
        print(f"Error loading dataset: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect PyTorch dataset structure")
    parser.add_argument("--path", type=str, default="./datasets/dataset_yolo_conf=0.35_iou=0.7_gaze-noise-filtering=false.pth",
                       help="Path to the .pth dataset file")
    parser.add_argument("--max-depth", type=int, default=5,
                       help="Maximum recursion depth")
    args = parser.parse_args()
    
    analyze_dataset(args.path, args.max_depth)