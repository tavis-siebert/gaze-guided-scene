"""
Test to reproduce KeyError in GraphDataset when using action_recognition task mode.

This test reproduces the issue where validation sampling uses get_future_action_labels
but action_recognition task mode expects action_labels["action_recognition"] key.
"""

import pytest
from unittest.mock import Mock, patch
import torch

from gazegraph.training.dataset.graph_dataset import GraphDataset
from gazegraph.graph.checkpoint_manager import GraphCheckpoint
from gazegraph.datasets.egtea_gaze.video_metadata import VideoMetadata
from gazegraph.config.config_utils import DotDict


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = DotDict(
        {
            "training": {"val_timestamps": [0.2, 0.5, 0.8]},
            "dataset": {
                "ego_topo": {
                    "splits": {
                        "train_video_lengths": "/fake/train_lengths.csv",
                        "val_video_lengths": "/fake/val_lengths.csv",
                    }
                },
                "egtea": {"raw_videos": "/fake/videos", "gaze_data": "/fake/gaze"},
                "sampling": {
                    "strategy": "all",
                    "samples_per_video": 1,
                    "allow_duplicates": False,
                    "oversampling": False,
                    "random_seed": 42,
                },
            },
        }
    )
    return config


@pytest.fixture
def mock_checkpoint():
    """Mock checkpoint for testing."""
    checkpoint = Mock(spec=GraphCheckpoint)
    checkpoint.frame_number = 100
    checkpoint.video_length = 1000
    checkpoint.video_name = "test_video"

    # Mock get_future_action_labels to return typical future action labels
    # This simulates what happens in validation sampling
    checkpoint.get_future_action_labels.return_value = {
        "next_action": torch.tensor(5),
        "future_actions": torch.zeros(106),
        "future_actions_ordered": torch.tensor([5, 10]),
    }

    return checkpoint


def test_action_recognition_keyerror_reproduction(
    mock_config, mock_checkpoint, tmp_path
):
    """Test that reproduces the KeyError when using action_recognition task mode."""

    # Create a temporary directory structure
    val_dir = tmp_path / "val"
    val_dir.mkdir()

    # Create a mock checkpoint file
    checkpoint_file = val_dir / "test_video_graph.pth"
    checkpoint_file.touch()

    # Mock the checkpoint loading
    with patch(
        "gazegraph.training.dataset.graph_dataset.CheckpointManager.load_checkpoints"
    ) as mock_load:
        mock_load.return_value = [mock_checkpoint]

        # Mock VideoMetadata to avoid file loading issues
        with patch("gazegraph.training.dataset.graph_dataset.VideoMetadata") as mock_vm:
            mock_metadata = Mock(spec=VideoMetadata)
            # Return empty action records to trigger fallback to old behavior
            mock_metadata.get_records_for_video.return_value = []
            mock_vm.return_value = mock_metadata

            # Mock the graph assembler creation
            with patch(
                "gazegraph.training.dataset.graph_dataset.create_graph_assembler"
            ) as mock_assembler:
                mock_assembler.return_value = Mock()

                # Mock get_samples to return empty list (simulating no action records found)
                # This will cause the validation sampling to fall back to the old behavior
                with patch(
                    "gazegraph.training.dataset.graph_dataset.get_samples"
                ) as mock_get_samples:
                    mock_get_samples.return_value = []

                    # Create the dataset with action_recognition task mode
                    dataset = GraphDataset(
                        config=mock_config,
                        root_dir=str(tmp_path),
                        split="val",
                        task_mode="action_recognition",  # This is the problematic task mode
                        object_node_feature="one-hot",
                        device="cpu",
                    )

                    # Since get_samples returned empty, no samples should be added
                    # and the dataset should be empty
                    assert len(dataset) == 0


def test_future_actions_works_correctly(mock_config, mock_checkpoint, tmp_path):
    """Test that future_actions task mode works correctly (no KeyError)."""

    # Create a temporary directory structure
    val_dir = tmp_path / "val"
    val_dir.mkdir()

    # Create a mock checkpoint file
    checkpoint_file = val_dir / "test_video_graph.pth"
    checkpoint_file.touch()

    # Mock the checkpoint loading
    with patch(
        "gazegraph.training.dataset.graph_dataset.CheckpointManager.load_checkpoints"
    ) as mock_load:
        mock_load.return_value = [mock_checkpoint]

        # Mock VideoMetadata to avoid file loading issues
        with patch("gazegraph.training.dataset.graph_dataset.VideoMetadata") as mock_vm:
            mock_metadata = Mock(spec=VideoMetadata)
            mock_vm.return_value = mock_metadata

            # Mock the graph assembler creation and assembly
            with patch(
                "gazegraph.training.dataset.graph_dataset.create_graph_assembler"
            ) as mock_assembler:
                mock_assembler_instance = Mock()
                mock_assembler_instance.assemble.return_value = Mock()
                mock_assembler.return_value = mock_assembler_instance

                # Create the dataset with future_actions task mode (default)
                dataset = GraphDataset(
                    config=mock_config,
                    root_dir=str(tmp_path),
                    split="val",
                    task_mode="future_actions",  # This should work fine
                    object_node_feature="one-hot",
                    device="cpu",
                )

                # This should work without KeyError
                _ = dataset[0]

                # Verify that the assembler was called with the correct label
                mock_assembler_instance.assemble.assert_called_once()
                args, kwargs = mock_assembler_instance.assemble.call_args
                checkpoint_arg, y_arg = args

                # The y argument should be the "future_actions" tensor
                assert torch.equal(y_arg, torch.zeros(106))


def test_action_recognition_fix_works(mock_config, mock_checkpoint, tmp_path):
    """Test that the fix works for action_recognition task mode."""

    # Create a temporary directory structure
    val_dir = tmp_path / "val"
    val_dir.mkdir()

    # Create a mock checkpoint file
    checkpoint_file = val_dir / "test_video_graph.pth"
    checkpoint_file.touch()

    # Mock the checkpoint loading
    with patch(
        "gazegraph.training.dataset.graph_dataset.CheckpointManager.load_checkpoints"
    ) as mock_load:
        mock_load.return_value = [mock_checkpoint]

        # Mock VideoMetadata to avoid file loading issues
        with patch("gazegraph.training.dataset.graph_dataset.VideoMetadata") as mock_vm:
            mock_metadata = Mock(spec=VideoMetadata)
            mock_vm.return_value = mock_metadata

            # Mock get_samples to return proper action recognition samples
            with patch(
                "gazegraph.training.dataset.graph_dataset.get_samples"
            ) as mock_get_samples:
                # Mock get_samples to return samples with action_recognition key
                mock_get_samples.return_value = [
                    (mock_checkpoint, {"action_recognition": torch.tensor(5)})
                ]

                # Mock the graph assembler creation and assembly
                with patch(
                    "gazegraph.training.dataset.graph_dataset.create_graph_assembler"
                ) as mock_assembler:
                    mock_assembler_instance = Mock()
                    mock_assembler_instance.assemble.return_value = Mock()
                    mock_assembler.return_value = mock_assembler_instance

                    # Create the dataset with action_recognition task mode
                    dataset = GraphDataset(
                        config=mock_config,
                        root_dir=str(tmp_path),
                        split="val",
                        task_mode="action_recognition",
                        object_node_feature="one-hot",
                        device="cpu",
                    )

                    # This should work without KeyError now
                    _ = dataset[0]

                    # Verify that get_samples was called with the correct parameters
                    mock_get_samples.assert_called_once()
                    args, kwargs = mock_get_samples.call_args

                    # Check that task_mode was passed correctly
                    assert kwargs["task_mode"] == "action_recognition"

                    # Verify that the assembler was called with the correct label
                    mock_assembler_instance.assemble.assert_called_once()
                    args, kwargs = mock_assembler_instance.assemble.call_args
                    checkpoint_arg, y_arg = args

                    # The y argument should be the action recognition tensor
                    assert torch.equal(y_arg, torch.tensor(5))
