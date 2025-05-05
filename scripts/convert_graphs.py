#!/usr/bin/env python3
"""
Convert graph checkpoint files from the old format (list of checkpoint objects)
to the new format (dictionary with context and checkpoints).
"""

import torch
import os
import argparse
from pathlib import Path
from typing import Tuple
from tqdm import tqdm

from graph.checkpoint_manager import GraphCheckpoint
from logger import get_logger

logger = get_logger(__name__)

def convert_checkpoint_file(input_path: Path, output_path: Path) -> bool:
    """
    Convert a single checkpoint file from the old format to the new format.
    
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
        if checkpoints:
            first_checkpoint = checkpoints[0]
            context = {
                "video_name": video_name,
                "labels_to_int": first_checkpoint.labels_to_int,
                "num_object_classes": first_checkpoint.num_object_classes,
                "video_length": first_checkpoint.video_length
            }
        else:
            context = {"video_name": video_name}
            
        # Create the new format dictionary
        new_format = {
            "context": context,
            "checkpoints": [cp.to_dict() for cp in checkpoints]
        }
        
        # Save the new format checkpoint file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(new_format, output_path)
        
        logger.info(f"Successfully converted {input_path} to {output_path} ({len(checkpoints)} checkpoints)")
        return True
        
    except Exception as e:
        logger.error(f"Error converting {input_path}: {str(e)}")
        return False

def convert_all_graphs(input_dir: str, output_dir: str) -> Tuple[int, int]:
    """
    Convert all graph checkpoint files in the input directory to the new format.
    
    Args:
        input_dir: Directory containing old format checkpoint files
        output_dir: Directory to save new format checkpoint files
        
    Returns:
        Tuple of (number of successful conversions, total number of files)
    """
    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir)
    
    # Get all checkpoint files
    checkpoint_files = list(input_dir_path.glob("**/*_graph.pth"))
    
    successful = 0
    for file_path in tqdm(checkpoint_files, desc="Converting checkpoint files"):
        # Determine relative path to maintain directory structure
        rel_path = file_path.relative_to(input_dir_path)
        output_path = output_dir_path / rel_path
        
        # Convert the file
        if convert_checkpoint_file(file_path, output_path):
            successful += 1
            
    return successful, len(checkpoint_files)

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
    successful, total = convert_all_graphs(args.input_dir, args.output_dir)
    
    logger.info(f"Conversion summary: {successful}/{total} files successfully converted.")
    
    # Check for unconverted files
    if successful < total:
        logger.warning(f"Failed to convert {total - successful} files. Check the logs for details.")
    
if __name__ == "__main__":
    main() 