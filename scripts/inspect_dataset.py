# load dataset from ../datasets/dataset.pth

import torch
import argparse
from typing import Any, Set
import numpy as np


def explore_dataset_structure(
    obj: Any,
    prefix: str = "",
    max_depth: int = 10,
    current_depth: int = 0,
    visited: Set[int] = None,
) -> None:
    """
    Recursively explore and print the structure of a PyTorch dataset in a compact JSON-like format.

    Args:
        obj: The object to explore
        prefix: String prefix for the current exploration level
        max_depth: Maximum recursion depth to prevent infinite loops
        current_depth: Current recursion depth
        visited: Set of object ids already visited to prevent circular references
    """
    if visited is None:
        visited = set()

    if current_depth > max_depth:
        print(f"{prefix}[...]")
        return

    obj_id = id(obj)
    if obj_id in visited:
        print(f"{prefix}[circular]")
        return

    if isinstance(obj, (dict, list, tuple, set, torch.Tensor, np.ndarray)):
        visited.add(obj_id)

    if obj is None:
        print(f"{prefix}null")
    elif isinstance(obj, (int, float, str, bool)):
        print(f"{prefix}{type(obj).__name__}: {obj}")
    elif isinstance(obj, dict):
        print(f"{prefix}{{")
        for key in obj:
            print(f'{prefix}  "{key}": ', end="")
            explore_dataset_structure(
                obj[key], "", max_depth, current_depth + 1, visited
            )
        print(f"{prefix}}}")
    elif isinstance(obj, (list, tuple)):
        container_type = type(obj).__name__
        if len(obj) == 0:
            print(f"{prefix}{container_type}: []")
        else:
            print(f"{prefix}{container_type}[{len(obj)}]: [")
            print(f"{prefix}  ", end="")
            explore_dataset_structure(obj[0], "", max_depth, current_depth + 1, visited)
            if len(obj) > 1:
                print(f"{prefix}  ... ({len(obj) - 1} more)")
            print(f"{prefix}]")
    elif isinstance(obj, torch.Tensor):
        shape_str = "×".join(str(d) for d in obj.shape)
        if obj.numel() == 0:
            print(f"{prefix}Tensor({shape_str}, {obj.dtype}): []")
        elif obj.numel() < 5:
            print(f"{prefix}Tensor({shape_str}, {obj.dtype}): {obj.tolist()}")
        else:
            flat = obj.flatten()
            print(
                f"{prefix}Tensor({shape_str}, {obj.dtype}): [{flat[0].item()}, {flat[1].item()}, ... ]"
            )
    elif isinstance(obj, np.ndarray):
        shape_str = "×".join(str(d) for d in obj.shape)
        if obj.size == 0:
            print(f"{prefix}ndarray({shape_str}, {obj.dtype}): []")
        elif obj.size < 5:
            print(f"{prefix}ndarray({shape_str}, {obj.dtype}): {obj.tolist()}")
        else:
            flat = obj.flatten()
            print(
                f"{prefix}ndarray({shape_str}, {obj.dtype}): [{flat[0]}, {flat[1]}, ... ]"
            )
    else:
        attrs = {}
        try:
            if hasattr(obj, "__dict__"):
                print(f"{prefix}{type(obj).__name__}: {{")
                for key, value in obj.__dict__.items():
                    print(f'{prefix}  "{key}": ', end="")
                    explore_dataset_structure(
                        value, "", max_depth, current_depth + 1, visited
                    )
                print(f"{prefix}}}")
            else:
                print(f"{prefix}{type(obj).__name__}")
        except:
            print(f"{prefix}{type(obj).__name__}")


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
    parser.add_argument(
        "--path",
        type=str,
        default="./datasets/dataset_yolo_conf=0.35_iou=0.7_gaze-noise-filtering=false.pth",
        help="Path to the .pth dataset file",
    )
    parser.add_argument(
        "--max-depth", type=int, default=5, help="Maximum recursion depth"
    )
    args = parser.parse_args()

    analyze_dataset(args.path, args.max_depth)
