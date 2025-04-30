import torch
import torch.nn as nn
import numpy as np
from tasks.base_task import BaseTask
from evaluation.metrics import confusion_matrix, mAP

class FutureActionsTask(BaseTask):
    def __init__(self, config, device):
        super().__init__(config, device, "future_actions")
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

    def compute_loss(self, output, y):
        return self.criterion(output, y.float())
    
    def calculate_epoch_metrics(self, epoch, epoch_loss, num_samples):
        train_recall, train_precision, train_mAP = self.test(self.train_loader)
        test_recall, test_precision, test_mAP = self.test(self.test_loader)
        
        self.metrics["train_losses"].append(epoch_loss / num_samples)
        self.metrics["train_recalls"].append(train_recall)
        self.metrics["train_precisions"].append(train_precision)
        self.metrics["train_mAPs"].append(train_mAP)
        
        self.metrics["test_recalls"].append(test_recall)
        self.metrics["test_precisions"].append(test_precision)
        self.metrics["test_mAPs"].append(test_mAP)
    
    def print_progress(self, epoch, epoch_loss, num_samples):
        print(f'Epoch: {epoch+1}')
        print('------------')
        print(f'Train Loss: {epoch_loss / num_samples}')
        print(f'Test mAP: {self.metrics["test_mAPs"][-1]}')
        print(f'Test Recall: {self.metrics["test_recalls"][-1]}')
        print(f'Test Precision: {self.metrics["test_precisions"][-1]}')
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