import os
import torch
import torch.nn as nn
from gazegraph.training.tasks.base_task import BaseTask
from gazegraph.datasets.egtea_gaze.action_record import ActionRecord
from gazegraph.datasets.egtea_gaze.video_metadata import VideoMetadata


class NextActionTask(BaseTask):
    """Task for predicting the next action from object graph snapshots."""

    def __init__(self, config, device, task_name, **kwargs):
        super().__init__(config=config, device=device, task_name=task_name, **kwargs)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        self.metadata = VideoMetadata(config)
        self.action_names = ActionRecord.get_action_names()
        self.logger.info(
            f"Loaded {len(self.action_names)} action names for next action prediction"
        )

    def compute_loss(self, output, y):
        """Compute cross-entropy loss for multi-class next action prediction."""
        return self.criterion(output, y.long())

    def calculate_epoch_metrics(self, epoch, epoch_loss, num_samples):
        """Calculate and log epoch metrics."""
        train_acc, train_top5_acc, train_class_metrics, train_pred_dist = (
            self.test_recognition(self.train_loader)
        )
        test_acc, test_top5_acc, test_class_metrics, test_pred_dist = (
            self.test_recognition(self.test_loader)
        )

        # Log overall metrics
        self.log_metric("train_loss", epoch_loss / num_samples, epoch)
        self.log_metric("train_acc", train_acc, epoch)
        self.log_metric("train_top5_acc", train_top5_acc, epoch)
        self.log_metric("test_acc", test_acc, epoch)
        self.log_metric("test_top5_acc", test_top5_acc, epoch)
        
        # Save best model
        if test_acc >= max(self.metrics['test_acc']):
            model_save_path = os.path.join(self.writer.file_writer.get_logdir(), "bestl_model.pt")
            state = {
                'config': self.config.to_dict(),
                'state_dict': self.model.state_dict()
            }
            torch.save(state, model_save_path)
            self.logger.info(f"Epoch {epoch}: Best model saved")

        # Log per-class metrics every 5 epochs
        if epoch % 5 == 0:
            self.log_class_metrics_to_tensorboard(
                test_class_metrics, epoch, self.action_names
            )
            self.log_prediction_distribution(test_pred_dist, epoch, self.action_names)

    def print_progress(self, epoch, epoch_loss, num_samples):
        """Print training progress."""
        self.logger.info(f"Epoch: {epoch + 1}")
        self.log_separator()
        self.log_metric_row("Train Loss", epoch_loss / num_samples)
        self.log_metric_row("Train Acc", self.metrics["train_acc"][-1])
        self.log_metric_row("Train Top-5 Acc", self.metrics["train_top5_acc"][-1])
        self.log_metric_row("Test Acc", self.metrics["test_acc"][-1])
        self.log_metric_row("Test Top-5 Acc", self.metrics["test_top5_acc"][-1])
        self.log_separator()

    def test(self, dset):
        """Test the model and return accuracy, top-5 accuracy, and per-class metrics."""
        return self.test_recognition(dset)
