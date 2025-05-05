#!/usr/bin/env python3
"""
Convert graph checkpoint files from the old format (list of checkpoint objects)
to the new format (dictionary with context and checkpoints).
"""

import torch
import os
import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict
from tqdm import tqdm

# Add project root to Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from graph.checkpoint_manager import GraphCheckpoint
from logger import get_logger

logger = get_logger(__name__)

def convert_checkpoint_file(input_path: Path, output_path: Path) -> bool:
    """
    Convert a single checkpoint file from the old format to the new format,
    removing redundant checkpoints where the graph didn't change.
    
    Args:
        input_path: Path to the input checkpoint file
        output_path: Path to save the converted checkpoint file
        
    Returns:
        True if conversion was successful, False otherwise
    """
    try:
        # Load the old format checkpoint file
        with torch.serialization.safe_globals([GraphCheckpoint]):
            checkpoints = torch.load(input_path, weights_only=False)
            
        # Check if already in new format
        if not isinstance(checkpoints, list):
            logger.warning(f"File {input_path} is not in the old format, skipping.")
            return False
            
        # Extract video name from filename
        video_name = input_path.stem.split('_')[0]
        
        # Get shared context data from the first checkpoint
        if not checkpoints:
            logger.warning(f"Empty checkpoint file {input_path}, skipping.")
            return False
            
        first_checkpoint = checkpoints[0]
        context = {
            "video_name": video_name,
            "labels_to_int": first_checkpoint.labels_to_int,
            "num_object_classes": first_checkpoint.num_object_classes,
            "video_length": first_checkpoint.video_length
        }
        
        # Prune redundant checkpoints using equality comparison
        pruned_checkpoints = []
        prev_checkpoint = None
        for checkpoint in checkpoints:
            if prev_checkpoint is None or checkpoint != prev_checkpoint:
                pruned_checkpoints.append(checkpoint)
                prev_checkpoint = checkpoint
        
        # Create the new format dictionary with pruned checkpoints
        new_format = {
            "context": context,
            "checkpoints": [cp.to_dict() for cp in pruned_checkpoints]
        }
        
        # Calculate reduction percentage
        reduction_percent = 0 if not checkpoints else ((len(checkpoints) - len(pruned_checkpoints)) / len(checkpoints)) * 100
        
        # Save the new format checkpoint file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(new_format, output_path)
        
        logger.info(f"Successfully converted {input_path} to {output_path}")
        logger.info(f"Reduced checkpoints from {len(checkpoints)} to {len(pruned_checkpoints)} ({reduction_percent:.1f}% reduction)")
        return True
        
    except Exception as e:
        logger.error(f"Error converting {input_path}: {str(e)}")
        return False

def convert_all_graphs(input_dir: str, output_dir: str) -> Tuple[int, int, Dict[str, int]]:
    """
    Convert all graph checkpoint files in the input directory to the new format.
    
    Args:
        input_dir: Directory containing old format checkpoint files
        output_dir: Directory to save new format checkpoint files
        
    Returns:
        Tuple of (number of successful conversions, total number of files, stats dictionary)
    """
    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir)
    
    # Get all checkpoint files
    checkpoint_files = list(input_dir_path.glob("**/*_graph.pth"))
    
    successful = 0
    stats = {
        "total_original_checkpoints": 0,
        "total_pruned_checkpoints": 0
    }
    
    for file_path in tqdm(checkpoint_files, desc="Converting checkpoint files"):
        # Determine relative path to maintain directory structure
        rel_path = file_path.relative_to(input_dir_path)
        output_path = output_dir_path / rel_path
        
        # Count checkpoints before conversion
        try:
            with torch.serialization.safe_globals([GraphCheckpoint]):
                original_checkpoints = torch.load(file_path, weights_only=False)
                if isinstance(original_checkpoints, list):
                    stats["total_original_checkpoints"] += len(original_checkpoints)
        except Exception:
            pass
            
        # Convert the file
        if convert_checkpoint_file(file_path, output_path):
            successful += 1
            
            # Count pruned checkpoints
            try:
                with torch.serialization.safe_globals([GraphCheckpoint]):
                    pruned_data = torch.load(output_path, weights_only=False)
                    if isinstance(pruned_data, dict) and "checkpoints" in pruned_data:
                        stats["total_pruned_checkpoints"] += len(pruned_data["checkpoints"])
            except Exception:
                pass
            
    return successful, len(checkpoint_files), stats

def main():
    parser = argparse.ArgumentParser(
        description='Convert graph checkpoint files from old format to new format.'
    )
    parser.add_argument(
        '--input-dir', type=str, default='datasets/graphs',
        help='Directory containing old format checkpoint files'
    )
    parser.add_argument(
        '--output-dir', type=str, default='datasets/graphs_pruned',
        help='Directory to save new format checkpoint files'
    )
    args = parser.parse_args()
    
    # Convert all checkpoint files
    successful, total, stats = convert_all_graphs(args.input_dir, args.output_dir)
    
    logger.info("-" * 60)
    logger.info("Conversion Summary")
    logger.info("-" * 60)
    logger.info(f"Files processed: {total}")
    logger.info(f"Files successfully converted: {successful}")
    if successful < total:
        logger.warning(f"Failed conversions: {total - successful}")
    
    if stats["total_original_checkpoints"] > 0:
        reduction = ((stats["total_original_checkpoints"] - stats["total_pruned_checkpoints"]) / 
                     stats["total_original_checkpoints"]) * 100
        logger.info(f"Original checkpoints: {stats['total_original_checkpoints']}")
        logger.info(f"Pruned checkpoints: {stats['total_pruned_checkpoints']}")
        logger.info(f"Checkpoint reduction: {reduction:.1f}%")
        logger.info(f"Storage savings: Removed {stats['total_original_checkpoints'] - stats['total_pruned_checkpoints']} redundant checkpoints")
    
    logger.info("-" * 60)
    logger.info(f"Converted files saved to: {args.output_dir}")
    
if __name__ == "__main__":
    main() 