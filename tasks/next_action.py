import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from training.utils import get_optimizer
from datasets.model_ready_dataset import load_datasets, get_graph_dataset
from models.gat_conv import GATForClassification
from evaluation.metrics import accuracy

class NextActionTask():
    def __init__(self, config, device):
        
        self.task = "next_action"

        self.device = device
        self.num_classes = config.training.num_classes
        
        train_set, test_set = load_datasets(config.dataset.output.dataset_file)
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

        x_sample, edge_attr_sample = train_data[0].x, train_data[0].edge_attr
        self.input_dim = x_sample.shape[1]
        self.edge_dim = edge_attr_sample.shape[1]
        self.hidden_dim = config.training.hidden_dim
        self.num_heads = config.training.num_heads
        self.num_layers = config.training.num_layers
        self.res_connect = config.training.res_connect

        self.model = GATForClassification(self.num_classes, self.input_dim, self.hidden_dim, self.edge_dim, self.num_heads, self.num_layers, self.res_connect)
        self.model.to(self.device)
        
        self.num_epochs = config.training.num_epochs
        self.optimizer = get_optimizer(config.training.optimizer, self.model, config.training.lr)
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum')

        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'test_acc': []
        }

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
                loss = self.criterion(output, y)
                epoch_loss += loss.item()
                num_samples += y.shape[0]

                loss.backward()
                self.optimizer.step()
            
            train_acc = self.test(self.train_loader)
            test_acc = self.test(self.test_loader)

            self.metrics['train_loss'].append(epoch_loss / num_samples)
            self.metrics['train_acc'].append(train_acc)
            self.metrics['test_acc'].append(test_acc)

            if (epoch + 1)%5 == 0 or epoch == 0:
                print(f'Epoch: {epoch+1}')
                print('------------')
                print(f'Train Loss: {epoch_loss / num_samples}')
                print(f'Train Acc: {train_acc}')
                print(f'Test Acc: {test_acc}')
                print('------------')
            
    def test(self, dset):
        self.model.eval()

        total_acc = 0
        total_samples = 0
        with torch.no_grad():
            for data in dset:
                x, edge_index, edge_attr, y, batch = data.x, data.edge_index, data.edge_attr, data.y, data.batch
                edge_attr = edge_attr.to(x.dtype)
                x, edge_index, edge_attr, y, batch = x.to(self.device), edge_index.to(self.device), edge_attr.to(self.device), y.to(self.device), batch.to(self.device)
                output = self.model(x, edge_index, edge_attr, batch)
                pred = output.argmax(dim=-1)

                # Ensure pred and y have the same size
                acc = accuracy(pred, y)
                total_acc += acc
                total_samples += 1

        return total_acc / total_samples
        
        
