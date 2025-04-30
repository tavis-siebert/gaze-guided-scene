import torch
import torch.nn as nn
import numpy as np
from tasks.base_task import BaseTask
from evaluation.metrics import confusion_matrix, mAP

class FutureActionsTask(BaseTask):
    def __init__(self, config, device):
        super().__init__(config, device, "future_actions")
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum')
    
    def compute_loss(self, output, y):
        return self.criterion(output, y.float())
    
    def calculate_epoch_metrics(self, epoch, epoch_loss, num_samples):
        train_recall, train_precision, train_mAP = self.test(self.train_loader)
        test_recall, test_precision, test_mAP = self.test(self.test_loader)
        
        self.log_metric('train_loss', epoch_loss / num_samples)
        self.log_metric('train_recall', train_recall)
        self.log_metric('train_precision', train_precision)
        self.log_metric('train_mAP', train_mAP)
        
        self.log_metric('test_recall', test_recall)
        self.log_metric('test_precision', test_precision)
        self.log_metric('test_mAP', test_mAP)
    
    def print_progress(self, epoch, epoch_loss, num_samples):
        self.logger.info(f'Epoch: {epoch+1}')
        self.log_separator()
        self.log_metric_row('Train Loss', epoch_loss / num_samples)
        self.log_metric_row('Test mAP', self.metrics["test_mAP"][-1])
        self.log_metric_row('Test Recall', self.metrics["test_recall"][-1])
        self.log_metric_row('Test Precision', self.metrics["test_precision"][-1])
        self.log_separator()
    
    def test(self, dset):
        all_targets = []
        all_predictions = []
        
        total_samples = 0
        total_recall, total_precision = 0, 0
        
        with self.evaluation_mode():
            for data in dset:
                x, edge_index, edge_attr, y, batch = self._transfer_batch_to_device(data)
                
                output = self.model(x, edge_index, edge_attr, batch)
                output_probs = torch.sigmoid(output)
                pred = (output_probs > 0.5).float()
                
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