import torch
import torch.nn as nn
import numpy as np
from gazegraph.training.tasks.base_task import BaseTask
from gazegraph.training.evaluation.metrics import confusion_matrix, mAP
from gazegraph.datasets.egtea_gaze.action_record import ActionRecord
from gazegraph.datasets.egtea_gaze.video_metadata import VideoMetadata


class FutureActionsTask(BaseTask):
    def __init__(self, config, device, task_name, **kwargs):
        super().__init__(config=config, device=device, task_name=task_name, **kwargs)
        self.criterion = nn.BCEWithLogitsLoss(reduction="sum")

        self.metadata = VideoMetadata(config)
        self.action_names = ActionRecord.get_action_names()
        self.logger.info(
            f"Loaded {len(self.action_names)} action names for metrics visualization"
        )

    def compute_loss(self, output, y):
        # Check if y is flattened and needs reshaping
        batch_size, num_classes = output.shape
        if len(y.shape) == 1 and y.shape[0] == batch_size * num_classes:
            y = y.view(batch_size, num_classes)

        return self.criterion(output, y.float())

    def calculate_epoch_metrics(self, epoch, epoch_loss, num_samples):
        train_recall, train_precision, train_mAP, train_class_metrics = self.test(
            self.train_loader
        )
        test_recall, test_precision, test_mAP, test_class_metrics = self.test(
            self.test_loader
        )

        # Log overall metrics
        self.log_metric("train_loss", epoch_loss / num_samples, epoch)
        self.log_metric("train_recall", train_recall, epoch)
        self.log_metric("train_precision", train_precision, epoch)
        self.log_metric("train_mAP", train_mAP, epoch)

        self.log_metric("test_recall", test_recall, epoch)
        self.log_metric("test_precision", test_precision, epoch)
        self.log_metric("test_mAP", test_mAP, epoch)

        # Log per-class metrics
        if epoch % 5 == 0:  # Log detailed metrics every 5 epochs to avoid cluttering
            # Log per-class metrics using human-readable action names
            for class_idx, metrics in test_class_metrics.items():
                action_name = self.action_names.get(class_idx, f"action_{class_idx}")
                # Remove spaces for tensorboard tag compatibility
                action_tag = action_name.replace(" ", "_")

                self.writer.add_scalar(
                    f"class/{action_tag}/precision", metrics["precision"], epoch
                )
                self.writer.add_scalar(
                    f"class/{action_tag}/recall", metrics["recall"], epoch
                )
                self.writer.add_scalar(f"class/{action_tag}/f1", metrics["f1"], epoch)

            # Log distribution of predicted actions
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

                # Log as histogram
                self.log_histogram(
                    self.test_pred_distribution, epoch, "action_distribution"
                )

    def print_progress(self, epoch, epoch_loss, num_samples):
        self.logger.info(f"Epoch: {epoch + 1}")
        self.log_separator()
        self.log_metric_row("Train Loss", epoch_loss / num_samples)
        self.log_metric_row("Test mAP", self.metrics["test_mAP"][-1])
        self.log_metric_row("Test Recall", self.metrics["test_recall"][-1])
        self.log_metric_row("Test Precision", self.metrics["test_precision"][-1])
        self.log_separator()

    def test(self, dset):
        all_targets = []
        all_predictions = []

        total_samples = 0
        total_recall, total_precision = 0, 0

        # Initialize per-class metrics
        class_tp = np.zeros(self.num_classes)
        class_fp = np.zeros(self.num_classes)
        class_fn = np.zeros(self.num_classes)

        with self.evaluation_mode():
            for data in dset:
                x, edge_index, edge_attr, y, batch = self._transfer_batch_to_device(
                    data
                )

                output = self.model(x, edge_index, edge_attr, batch)
                output_probs = torch.sigmoid(output)
                pred = (output_probs > 0.5).float()

                # Ensure y has same shape as output/pred
                batch_size, num_classes = pred.shape
                if len(y.shape) == 1 and y.shape[0] == batch_size * num_classes:
                    y = y.view(batch_size, num_classes)

                all_targets.append(y.detach().cpu().numpy())
                all_predictions.append(pred.detach().cpu().numpy())

                # Calculate overall metrics
                conf_mat = confusion_matrix(pred, y)
                tp, fp, fn, _ = conf_mat.flatten()
                if tp + fn != 0:
                    total_recall += tp / (tp + fn)
                if tp + fp != 0:
                    total_precision += tp / (tp + fp)

                # Update per-class metrics
                for c in range(self.num_classes):
                    class_y = y[:, c]
                    class_pred = pred[:, c]

                    # Calculate TP, FP, FN for this class
                    class_tp[c] += torch.sum((class_pred == 1) & (class_y == 1)).item()
                    class_fp[c] += torch.sum((class_pred == 1) & (class_y == 0)).item()
                    class_fn[c] += torch.sum((class_pred == 0) & (class_y == 1)).item()

                total_samples += 1

            # Calculate overall metrics
            all_targets = np.vstack(all_targets)
            all_predictions = np.vstack(all_predictions)
            total_mAP = mAP(all_predictions, all_targets)

            # Count distribution of predicted actions
            pred_counts = np.sum(all_predictions, axis=0)
            self.test_pred_distribution = pred_counts

            # Calculate per-class metrics
            class_metrics = {}
            for c in range(self.num_classes):
                if class_tp[c] + class_fn[c] > 0:
                    recall = class_tp[c] / (class_tp[c] + class_fn[c])
                else:
                    recall = 0

                if class_tp[c] + class_fp[c] > 0:
                    precision = class_tp[c] / (class_tp[c] + class_fp[c])
                else:
                    precision = 0

                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0

                class_metrics[c] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": class_tp[c] + class_fn[c],
                }

            # Create and log confusion matrix if this is the test dataset and we're at a logging epoch
            if (
                dset == self.test_loader and len(all_targets) < 100
            ):  # Only for manageable test sets
                # Create confusion matrix for top predicted classes
                cm = np.zeros((min(10, self.num_classes), min(10, self.num_classes)))
                top_classes = np.argsort(-np.sum(all_targets, axis=0))[
                    :10
                ]  # Top 10 most common classes

                for i, class_i in enumerate(top_classes):
                    for j, class_j in enumerate(top_classes):
                        # Count cases where class_i is actual and class_j is predicted
                        actual_i = all_targets[:, class_i] == 1
                        pred_j = all_predictions[:, class_j] == 1
                        cm[i, j] = np.sum(actual_i & pred_j)

                # Get human-readable names for top classes
                top_class_names = [
                    self.action_names.get(c, f"action_{c}") for c in top_classes
                ]

                # Log the confusion matrix
                self.log_confusion_matrix(cm, total_samples, top_class_names)

            return (
                total_recall / total_samples,
                total_precision / total_samples,
                total_mAP,
                class_metrics,
            )
