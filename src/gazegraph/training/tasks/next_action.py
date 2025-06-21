import os
import torch
import torch.nn as nn
from gazegraph.training.evaluation.metrics import accuracy
from gazegraph.training.tasks.base_task import BaseTask

class NextActionTask(BaseTask):
    def __init__(self, config, device, task_name, **kwargs):
        super().__init__(config=config, device=device, task_name=task_name, **kwargs)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
    
    def compute_loss(self, output, y):
        return self.criterion(output, y.long())
    
    def calculate_epoch_metrics(self, epoch, epoch_loss, num_samples):
        train_acc = self.test(self.train_loader)
        test_acc = self.test(self.test_loader)
        
        self.log_metric('train_loss', epoch_loss / num_samples, epoch)
        self.log_metric('train_acc', train_acc, epoch)
        self.log_metric('test_acc', test_acc, epoch)

        if test_acc >= max(self.metrics['test_acc']):
            model_save_path = os.path.join(self.writer.file_writer.get_logdir(), "bestl_model.pt")
            state = {
                'config': self.config.to_dict(),
                'state_dict': self.model.state_dict()
            }
            torch.save(state, model_save_path)
            self.logger.info(f"Epoch {epoch}: Best model saved")

        # Log per-class metrics
        #TODO detailed logging every 5 epochs (e.g. using fine-grained noun and verb labels?)
    
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