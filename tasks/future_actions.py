import torch
import torch.nn as nn
import numpy as np
from models.gat_conv import GATForClassification
from torch_geometric.loader import DataLoader
from evaluation.metrics import confusion_matrix, mAP
from datasets.model_ready_dataset import load_datasets, get_graph_dataset
from training.utils import get_optimizer

class FutureActionsTask():
    def __init__(self, config, device):
        
        self.task = "future_actions"

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
        self.criterion = nn.CrossEntropyLoss()
        
        self.metrics = {
            "train_losses": [],
            "train_recalls": [],
            "train_precisions": [],
            "train_mAPs": [],
            "test_recalls": [],
            "test_precisions": [],
            "test_mAPs": []
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
                loss = self.criterion(output, y.float())
                epoch_loss += loss.item()
                num_samples += y.shape[0]

                loss.backward()
                self.optimizer.step()

            train_recall, train_precision, train_mAP = self.test(self.train_loader)
            test_recall, test_precision, test_mAP = self.test(self.test_loader)

            self.metrics["train_losses"].append(epoch_loss / num_samples)
            self.metrics["train_recalls"].append(train_recall)
            self.metrics["train_precisions"].append(train_precision)
            self.metrics["train_mAPs"].append(train_mAP)

            self.metrics["test_recalls"].append(test_recall)
            self.metrics["test_precisions"].append(test_precision)
            self.metrics["test_mAPs"].append(test_mAP)

            if (epoch + 1)%5 == 0 or epoch == 0:
                print(f'Epoch: {epoch+1}')
                print('------------')
                print(f'Train Loss: {epoch_loss / num_samples}')
                print(f'Test mAP: {test_mAP}')
                print(f'Test Recall: {test_recall}')
                print(f'Test Precision: {test_precision}')
                print('------------')

    def test(self, dset):
        self.model.eval()

        all_targets = []
        all_predictions = []

        total_samples = 0
        total_recall, total_precision = 0, 0
        with torch.no_grad():
            for data in dset:
                x, edge_index, edge_attr, y, batch = data.x, data.edge_index, data.edge_attr, data.y, data.batch
                edge_attr = edge_attr.to(x.dtype)
                x, edge_index, edge_attr, y, batch = x.to(self.device), edge_index.to(self.device), edge_attr.to(self.device), y.to(self.device), batch.to(self.device)

                output = self.model(x, edge_index, edge_attr, batch)
                pred = (output > 0.5).float()

                all_targets.append(y.detach().cpu().numpy())
                all_predictions.append(pred.detach().cpu().numpy())

                conf_mat = confusion_matrix(pred, y)
                tp, fp, fn, _ = conf_mat.flatten()
                if tp + fn != 0:
                    total_recall += tp / (tp + fn)
                if tp + fp != 0:
                    total_precision += tp / (tp + fp)

                total_samples += 1

            all_targets = np.vstack(all_targets)
            all_predictions = np.vstack(all_predictions)
            total_mAP = mAP(all_predictions, all_targets)

            return total_recall / total_samples, total_precision / total_samples, total_mAP