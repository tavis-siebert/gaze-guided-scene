import torch.nn as nn
import numpy as np
from gazegraph.training.tasks.base_task import BaseTask
from gazegraph.training.evaluation.metrics import accuracy, top_k_accuracy
from gazegraph.datasets.egtea_gaze.action_record import ActionRecord
from gazegraph.datasets.egtea_gaze.video_metadata import VideoMetadata


class ActionRecognitionTask(BaseTask):
    """Task for recognizing actions from object graph snapshots."""

    def __init__(self, config, device, task_name, **kwargs):
        super().__init__(config=config, device=device, task_name=task_name, **kwargs)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        self.metadata = VideoMetadata(config)
        self.action_names = ActionRecord.get_action_names()
        self.logger.info(
            f"Loaded {len(self.action_names)} action names for action recognition"
        )

    def compute_loss(self, output, y):
        """Compute cross-entropy loss for multi-class action recognition."""
        return self.criterion(output, y.long())

    def calculate_epoch_metrics(self, epoch, epoch_loss, num_samples):
        """Calculate and log epoch metrics."""
        train_acc, train_top5_acc, train_class_metrics = self.test(self.train_loader)
        test_acc, test_top5_acc, test_class_metrics = self.test(self.test_loader)

        # Log overall metrics
        self.log_metric("train_loss", epoch_loss / num_samples, epoch)
        self.log_metric("train_acc", train_acc, epoch)
        self.log_metric("train_top5_acc", train_top5_acc, epoch)
        self.log_metric("test_acc", test_acc, epoch)
        self.log_metric("test_top5_acc", test_top5_acc, epoch)

        # Log per-class metrics every 5 epochs
        if epoch % 5 == 0:
            for class_idx, metrics in test_class_metrics.items():
                action_name = self.action_names.get(class_idx, f"action_{class_idx}")
                action_tag = action_name.replace(" ", "_")

                self.writer.add_scalar(
                    f"class/{action_tag}/precision", metrics["precision"], epoch
                )
                self.writer.add_scalar(
                    f"class/{action_tag}/recall", metrics["recall"], epoch
                )
                self.writer.add_scalar(f"class/{action_tag}/f1", metrics["f1"], epoch)

            # Log prediction distribution
            if hasattr(self, "test_pred_distribution"):
                for class_idx, count in enumerate(self.test_pred_distribution):
                    if class_idx in self.action_names:
                        action_name = self.action_names.get(
                            class_idx, f"action_{class_idx}"
                        )
                        action_tag = action_name.replace(" ", "_")
                        self.writer.add_scalar(
                            f"distribution/{action_tag}", count, epoch
                        )

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
        all_targets = []
        all_predictions = []
        all_outputs = []
        total_acc = 0
        total_samples = 0

        # Initialize per-class metrics
        class_tp = np.zeros(self.num_classes)
        class_fp = np.zeros(self.num_classes)
        class_fn = np.zeros(self.num_classes)
        class_support = np.zeros(self.num_classes)

        with self.evaluation_mode():
            for data in dset:
                x, edge_index, edge_attr, y, batch = self._transfer_batch_to_device(
                    data
                )

                output = self.model(x, edge_index, edge_attr, batch)
                pred = output.argmax(dim=-1)

                all_targets.append(y.detach().cpu().numpy())
                all_predictions.append(pred.detach().cpu().numpy())
                all_outputs.append(output.detach().cpu().numpy())

                # Calculate accuracy
                acc = accuracy(pred, y)
                total_acc += acc
                total_samples += 1

                # Update per-class metrics
                for i in range(len(y)):
                    true_class = y[i].item()
                    pred_class = pred[i].item()

                    class_support[true_class] += 1

                    if true_class == pred_class:
                        class_tp[true_class] += 1
                    else:
                        class_fn[true_class] += 1
                        class_fp[pred_class] += 1

        # Calculate top-5 accuracy
        all_targets_flat = np.concatenate(all_targets)
        all_outputs_flat = np.vstack(all_outputs)
        top5_acc = top_k_accuracy(all_outputs_flat, all_targets_flat, k=5)

        # Calculate per-class metrics
        class_metrics = {}
        for c in range(self.num_classes):
            if class_support[c] > 0:
                recall = class_tp[c] / class_support[c] if class_support[c] > 0 else 0
                precision = (
                    class_tp[c] / (class_tp[c] + class_fp[c])
                    if (class_tp[c] + class_fp[c]) > 0
                    else 0
                )
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )

                class_metrics[c] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": class_support[c],
                }

        # Store prediction distribution
        all_predictions_flat = np.concatenate(all_predictions)
        pred_counts = np.bincount(all_predictions_flat, minlength=self.num_classes)
        self.test_pred_distribution = pred_counts

        return total_acc / total_samples, top5_acc, class_metrics
