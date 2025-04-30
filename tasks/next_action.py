import torch
import torch.nn as nn
from evaluation.metrics import accuracy
from tasks.base_task import BaseTask

class NextActionTask(BaseTask):
    def __init__(self, config, device):
        super().__init__(config, device, "next_action")
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum')
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'test_acc': []
        }

    def compute_loss(self, output, y):
        return self.criterion(output, y)
    
    def calculate_epoch_metrics(self, epoch, epoch_loss, num_samples):
        train_acc = self.test(self.train_loader)
        test_acc = self.test(self.test_loader)
        
        self.metrics['train_loss'].append(epoch_loss / num_samples)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['test_acc'].append(test_acc)
    
    def print_progress(self, epoch, epoch_loss, num_samples):
        print(f'Epoch: {epoch+1}')
        print('------------')
        print(f'Train Loss: {epoch_loss / num_samples}')
        print(f'Train Acc: {self.metrics["train_acc"][-1]}')
        print(f'Test Acc: {self.metrics["test_acc"][-1]}')
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

                acc = accuracy(pred, y)
                total_acc += acc
                total_samples += 1

        return total_acc / total_samples 