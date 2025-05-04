import torch
import torch.nn as nn
import contextlib
import os
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from training.utils import get_optimizer
from training.data_loader import GraphDataset, create_dataloader
from models.gat_conv import GATForClassification
from logger import get_logger
from pathlib import Path

class BaseTask:
    def __init__(self, config, device, task_name):
        self.task = task_name
        self.device = device
        self.config = config
        self.num_classes = config.training.num_classes
        self.logger = get_logger(__name__)
        
        # Load data and setup loaders
        self._setup_data()
        
        # Initialize model
        self.model = GATForClassification(
            self.num_classes,
            self.input_dim, 
            self.hidden_dim,
            self.edge_dim, 
            self.num_heads, 
            self.num_layers, 
            self.res_connect
        )
        self.model.to(self.device)
        
        # Training parameters
        self.num_epochs = config.training.num_epochs
        self.optimizer = get_optimizer(config.training.optimizer, self.model, config.training.lr)
        
        # Initialize empty metrics dictionary (to be populated by subclasses)
        self.metrics = {}
        
        # Setup tensorboard writer
        log_dir = os.path.join('logs', f'{self.task}')
        self.writer = SummaryWriter(log_dir=log_dir)
    
    def _setup_data(self):
        """Setup datasets and data loaders"""
        # Get graphs directory path
        graphs_dir = Path(self.config.directories.repo.graphs)
        if not graphs_dir.exists():
            self.logger.error(f"Graphs directory {graphs_dir} not found. Run 'python main.py build' first.")
            raise FileNotFoundError(f"Graphs directory {graphs_dir} not found")
            
        # Get validation timestamps from config
        val_timestamps = self.config.training.val_timestamps
        
        # Create train loader
        self.train_loader = create_dataloader(
            root_dir=str(graphs_dir),
            split="train",
            val_timestamps=val_timestamps,
            task_mode=self.task,
            batch_size=self.config.training.batch_size,
            node_drop_p=self.config.training.node_drop_p,
            max_droppable=self.config.training.max_nodes_droppable,
            shuffle=True,
            num_workers=self.config.processing.dataloader_workers,
            config=self.config
        )
        
        # Create validation loader
        self.test_loader = create_dataloader(
            root_dir=str(graphs_dir),
            split="val",
            val_timestamps=val_timestamps,
            task_mode=self.task,
            batch_size=1,
            node_drop_p=0.0,  # No augmentation for validation
            max_droppable=0,
            shuffle=False,
            num_workers=self.config.processing.dataloader_workers,
            config=self.config
        )
        
        # Extract dimensions from data
        train_dataset = self.train_loader.dataset
        sample = train_dataset[0]
        self.input_dim = sample.x.shape[1]
        self.edge_dim = sample.edge_attr.shape[1]
        self.hidden_dim = self.config.training.hidden_dim
        self.num_heads = self.config.training.num_heads
        self.num_layers = self.config.training.num_layers
        self.res_connect = self.config.training.res_connect
        
        self.logger.info(f"Loaded train dataset with {len(train_dataset)} samples")
        self.logger.info(f"Loaded validation dataset with {len(self.test_loader.dataset)} samples")
        
    def _transfer_batch_to_device(self, data):
        """Transfer batch data to device"""
        x, edge_index, edge_attr, y, batch = data.x, data.edge_index, data.edge_attr, data.y, data.batch
        edge_attr = edge_attr.to(x.dtype)
        return (
            x.to(self.device),
            edge_index.to(self.device),
            edge_attr.to(self.device),
            y.to(self.device),
            batch.to(self.device)
        )
    
    @contextlib.contextmanager
    def evaluation_mode(self):
        """Context manager for model evaluation mode with no gradient tracking"""
        original_mode = self.model.training
        self.model.eval()
        with torch.no_grad():
            try:
                yield
            finally:
                self.model.train(original_mode)
    
    def log_metric(self, metric_name, value, epoch):
        """Add a metric value to the metrics dictionary and to tensorboard"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

        # Log to tensorboard
        self.writer.add_scalar(metric_name, value, epoch)
    
    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0
            num_samples = 0
            
            for data in self.train_loader:
                x, edge_index, edge_attr, y, batch = self._transfer_batch_to_device(data)
                
                self.optimizer.zero_grad()
                output = self.model(x, edge_index, edge_attr, batch)
                
                loss = self.compute_loss(output, y)
                epoch_loss += loss.item()
                num_samples += y.shape[0]
                
                loss.backward()
                self.optimizer.step()
            
            # Calculate metrics
            self.calculate_epoch_metrics(epoch, epoch_loss, num_samples)
            
            # Log progress at specified intervals
            if (epoch + 1) % 5 == 0 or epoch == 0:
                self.print_progress(epoch, epoch_loss, num_samples)
    
    def log_metric_row(self, label, value):
        """Log a formatted metric row"""
        self.logger.info(f"{label}: {value:.6f}")
    
    def log_separator(self):
        """Log a separator line"""
        self.logger.info('-' * 12)
    
    def compute_loss(self, output, y):
        """Compute loss - to be implemented by subclasses"""
        raise NotImplementedError
    
    def calculate_epoch_metrics(self, epoch, epoch_loss, num_samples):
        """Calculate and store metrics for epoch - to be implemented by subclasses"""
        raise NotImplementedError
    
    def print_progress(self, epoch, epoch_loss, num_samples):
        """Log training progress - to be implemented by subclasses"""
        raise NotImplementedError
    
    def test(self, dset):
        """Run evaluation - to be implemented by subclasses"""
        raise NotImplementedError
    
    def close(self):
        """Close tensorboard writer and free resources"""
        self.writer.close() 