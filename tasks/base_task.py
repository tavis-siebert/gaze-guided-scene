import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from training.utils import get_optimizer
from datasets.model_ready_dataset import load_datasets, get_graph_dataset
from models.gat_conv import GATForClassification

class BaseTask:
    def __init__(self, config, device, task_name):
        self.task = task_name
        self.device = device
        self.num_classes = config.training.num_classes
        
        # Load data
        train_set, test_set = load_datasets(config.dataset.output.dataset_file)
        
        # Setup datasets and dataloaders
        train_data = get_graph_dataset(
            train_set, 
            self.task, 
            self.num_classes, 
            config.training.node_drop_p, 
            config.training.max_nodes_droppable
        )
        test_data = get_graph_dataset(test_set, self.task, self.num_classes)
        self.train_loader = DataLoader(train_data, batch_size=config.training.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
        
        # Extract dimensions from data
        x_sample, edge_attr_sample = train_data[0].x, train_data[0].edge_attr
        self.input_dim = x_sample.shape[1]
        self.edge_dim = edge_attr_sample.shape[1]
        self.hidden_dim = config.training.hidden_dim
        self.num_heads = config.training.num_heads
        self.num_layers = config.training.num_layers
        self.res_connect = config.training.res_connect
        
        # Initialize model
        self.model = GATForClassification(self.num_classes, self.input_dim, self.hidden_dim, 
                                         self.edge_dim, self.num_heads, self.num_layers, self.res_connect)
        self.model.to(self.device)
        
        # Training parameters
        self.num_epochs = config.training.num_epochs
        self.optimizer = get_optimizer(config.training.optimizer, self.model, config.training.lr)
        
        # Metrics dictionary is defined in child classes
    
    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0
            num_samples = 0
            
            for data in self.train_loader:
                x, edge_index, edge_attr, y, batch = data.x, data.edge_index, data.edge_attr, data.y, data.batch
                edge_attr = edge_attr.to(x.dtype)
                x, edge_index, edge_attr, y, batch = x.to(self.device), edge_index.to(self.device), edge_attr.to(self.device), y.to(self.device), batch.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(x, edge_index, edge_attr, batch)
                
                loss = self.compute_loss(output, y)
                epoch_loss += loss.item()
                num_samples += y.shape[0]
                
                loss.backward()
                self.optimizer.step()
            
            # Calculate metrics - these methods are implemented in subclasses
            self.calculate_epoch_metrics(epoch, epoch_loss, num_samples)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                self.print_progress(epoch, epoch_loss, num_samples)
    
    def compute_loss(self, output, y):
        """Compute loss - to be implemented by subclasses"""
        raise NotImplementedError
    
    def calculate_epoch_metrics(self, epoch, epoch_loss, num_samples):
        """Calculate and store metrics for epoch - to be implemented by subclasses"""
        raise NotImplementedError
    
    def print_progress(self, epoch, epoch_loss, num_samples):
        """Print training progress - to be implemented by subclasses"""
        raise NotImplementedError
    
    def test(self, dset):
        """Run evaluation - to be implemented by subclasses"""
        raise NotImplementedError 