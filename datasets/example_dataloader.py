"""
Example script demonstrating how to use the graph dataset loading functionality.
"""

import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import os

from datasets.model_ready_dataset import GraphDataset, create_dataloader
from config.config_utils import load_config
from logger import get_logger

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Graph dataset loading example")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--batch-size", type=int, default=64, 
                        help="Batch size for data loader")
    parser.add_argument("--node-drop-p", type=float, default=0.0, 
                        help="Probability of node dropping (0.0 = no augmentation)")
    parser.add_argument("--max-droppable", type=int, default=0, 
                        help="Maximum number of nodes to drop in augmentation")
    parser.add_argument("--task-mode", type=str, default="future_actions",
                        choices=["future_actions", "future_actions_ordered", "next_action"],
                        help="Task mode for dataset")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val"],
                        help="Dataset split to load")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of worker processes for data loading")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Set up dataset paths
    graphs_dir = Path(config.directories.repo.datasets) / "graphs"
    
    if not graphs_dir.exists():
        logger.error(f"Graphs directory {graphs_dir} does not exist. Run build_graph.py first.")
        return
    
    # Get validation timestamps from config
    val_timestamps = config.training.val_timestamps
    
    # Create dataloader
    dataloader = create_dataloader(
        root_dir=str(graphs_dir),
        split=args.split,
        val_timestamps=val_timestamps,
        task_mode=args.task_mode,
        batch_size=args.batch_size,
        node_drop_p=args.node_drop_p,
        max_droppable=args.max_droppable,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Print dataset information
    dataset = dataloader.dataset
    logger.info(f"Loaded {args.split} dataset with {len(dataset)} samples")
    
    # Process a few batches as an example
    num_batches_to_show = min(3, len(dataloader))
    logger.info(f"Processing {num_batches_to_show} batches as example")
    
    for i, batch in enumerate(tqdm(dataloader, total=num_batches_to_show)):
        if i >= num_batches_to_show:
            break
            
        # Show batch information
        logger.info(f"Batch {i+1}:")
        logger.info(f"- Node features: {batch.x.shape}")
        logger.info(f"- Edge index: {batch.edge_index.shape}")
        logger.info(f"- Edge attributes: {batch.edge_attr.shape}")
        logger.info(f"- Labels: {batch.y.shape}")
        
        # You can now use this batch in your model
        # model(batch.x, batch.edge_index, batch.edge_attr)
    
    logger.info("Done!")

if __name__ == "__main__":
    main() 