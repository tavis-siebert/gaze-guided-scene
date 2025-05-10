import torch
import torch.nn as nn
from gazegraph.training.evaluation.metrics import accuracy
from gazegraph.training.tasks.base_task import BaseTask

class NextActionTask(BaseTask):
    def __init__(self, config, device, node_feature_type="one-hot"):
        super().__init__(config, device, "next_action", node_feature_type)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
    
    def compute_loss(self, output, y):
        return self.criterion(output, y.long())
    
    def calculate_epoch_metrics(self, epoch, epoch_loss, num_samples):
        train_acc = self.test(self.train_loader)
        test_acc = self.test(self.test_loader)
        
        self.log_metric('train_loss', epoch_loss / num_samples)
        self.log_metric('train_acc', train_acc)
        self.log_metric('test_acc', test_acc)
    
    def print_progress(self, epoch, epoch_loss, num_samples):
        self.logger.info(f'Epoch: {epoch+1}')
        self.log_separator()
        self.log_metric_row('Train Loss', epoch_loss / num_samples)
        self.log_metric_row('Train Acc', self.metrics["train_acc"][-1])
        self.log_metric_row('Test Acc', self.metrics["test_acc"][-1])
        self.log_separator()
    
    def test(self, dset):
        total_acc = 0
        total_samples = 0
        
        with self.evaluation_mode():
            for data in dset:
                x, edge_index, edge_attr, y, batch = self._transfer_batch_to_device(data)
                output = self.model(x, edge_index, edge_attr, batch)
                pred = output.argmax(dim=-1)

                acc = accuracy(pred, y)
                total_acc += acc
                total_samples += 1

        return total_acc / total_samples 