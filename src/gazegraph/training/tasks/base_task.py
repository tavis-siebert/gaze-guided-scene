import torch
import torch.nn as nn
import contextlib
import os
import time
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from training.utils import get_optimizer
from training.dataset import GraphDataset, create_dataloader
from models.gat_conv import GATForClassification
from logger import get_logger
from pathlib import Path

class BaseTask:
    def __init__(self, config, device, task_name, node_feature_type="one-hot"):
        self.task = task_name
        self.device = device
        self.config = config
        self.num_classes = config.training.num_classes
        self.logger = get_logger(__name__)
        self.node_feature_type = node_feature_type
        
        self.logger.info(f"Using node feature type: {node_feature_type}")
        
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
        
        # Setup tensorboard writer with unique run directory
        self._setup_tensorboard_writer()
    
    def _setup_tensorboard_writer(self):
        """Setup tensorboard writer with unique run directory"""
        # Create base log directory for the task
        base_log_dir = os.path.join('logs', f'{self.task}')
        
        # Find the next available run directory
        run_dirs = [d for d in os.listdir(base_log_dir) if os.path.isdir(os.path.join(base_log_dir, d)) and d.startswith('run_')] if os.path.exists(base_log_dir) else []
        run_numbers = [int(d.split('_')[1]) for d in run_dirs if d.split('_')[1].isdigit()]
        next_run = max(run_numbers) + 1 if run_numbers else 1
        
        # Create the run directory with timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = f"run_{next_run}_{timestamp}"
        log_dir = os.path.join(base_log_dir, run_dir)
        
        # Create the writer
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.logger.info(f"Tensorboard logs will be written to {log_dir}")
    
    def _setup_data(self):
        """Setup datasets and data loaders"""
        # Get graphs directory path
        graphs_dir = Path(self.config.directories.graphs)
        if not graphs_dir.exists():
            self.logger.error(f"Graphs directory {graphs_dir} not found. Run 'python main.py build' first.")
            raise FileNotFoundError(f"Graphs directory {graphs_dir} not found")
            
        # Create train loader
        self.train_loader = create_dataloader(
            root_dir=str(graphs_dir),
            split="train",
            task_mode=self.task,
            config=self.config,
            node_feature_type=self.node_feature_type,
            device=self.device
        )
        
        # Create validation loader
        self.test_loader = create_dataloader(
            root_dir=str(graphs_dir),
            split="val",
            task_mode=self.task,
            config=self.config,
            node_feature_type=self.node_feature_type,
            device=self.device
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
    
    def log_per_class_metrics(self, metrics_dict, epoch, prefix=''):
        """Log per-class metrics to tensorboard.
        
        Args:
            metrics_dict: Dictionary with class indices as keys and metric values as values
            epoch: Current epoch
            prefix: Optional prefix for the metric name
        """
        for class_idx, value in metrics_dict.items():
            metric_name = f"{prefix}_{class_idx}" if prefix else f"class_{class_idx}"
            self.writer.add_scalar(metric_name, value, epoch)
    
    def log_confusion_matrix(self, conf_matrix, epoch, class_names=None):
        """Log confusion matrix as an image to tensorboard.
        
        Args:
            conf_matrix: Confusion matrix as numpy array
            epoch: Current epoch
            class_names: Optional list of class names for labeling
        """
        import matplotlib.pyplot as plt
        import io
        from PIL import Image
        
        fig, ax = plt.figure(figsize=(10, 10)), plt.axes()
        im = ax.imshow(conf_matrix, cmap='Blues')
        
        # Set labels if provided
        if class_names:
            ax.set_xticks(np.arange(len(class_names)))
            ax.set_yticks(np.arange(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.set_yticklabels(class_names)
        
        # Add colorbar
        plt.colorbar(im)
        
        # Add text annotations
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                text_color = 'white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black'
                ax.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color=text_color)
        
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        img_tensor = torch.tensor(np.array(img).transpose((2, 0, 1)))
        
        # Log to tensorboard
        self.writer.add_image('confusion_matrix', img_tensor, epoch)
        plt.close(fig)
    
    def log_histogram(self, values, epoch, name):
        """Log a histogram to tensorboard.
        
        Args:
            values: Tensor or numpy array of values
            epoch: Current epoch
            name: Name of the histogram
        """
        self.writer.add_histogram(name, values, epoch)
    
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