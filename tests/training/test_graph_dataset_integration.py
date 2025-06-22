"""
Tests for graph dataset integration with action recognition.
"""

import pytest
from unittest.mock import Mock, patch
from gazegraph.training.dataset.graph_dataset import GraphDataset
from gazegraph.config.config_utils import DotDict
from pathlib import Path


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = DotDict(
        {
            "dataset": {
                "sampling": {
                    "strategy": "all",
                    "samples_per_video": 1,
                    "allow_duplicates": False,
                    "oversampling": False,
                    "random_seed": 42,
                    "action_completion_ratio": 0.8,
                    "min_nodes_threshold": 2,
                    "visit_lookback_frames": 30,
                }
            },
            "training": {"val_timestamps": [0.25, 0.5, 0.75]},
        }
    )
    return config


@pytest.fixture
def mock_root_dir(tmp_path):
    """Create mock root directory with checkpoint files."""
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    # Create mock checkpoint files
    for video in ["video1", "video2"]:
        checkpoint_file = train_dir / f"{video}_graph.pth"
        checkpoint_file.touch()

    return str(tmp_path)


class TestGraphDatasetIntegration:
    """Test cases for graph dataset integration."""

    @patch("gazegraph.training.dataset.graph_dataset.CheckpointManager")
    @patch("gazegraph.training.dataset.graph_dataset.VideoMetadata")
    @patch("gazegraph.training.dataset.graph_dataset.get_samples")
    @patch("gazegraph.training.dataset.graph_dataset.create_graph_assembler")
    def test_action_recognition_parameter_forwarding(
        self,
        mock_assembler,
        mock_get_samples,
        mock_metadata_class,
        mock_checkpoint_manager,
        mock_config,
        mock_root_dir,
    ):
        """Test that action recognition parameters are forwarded correctly."""
        # Setup mocks
        mock_metadata_class.return_value = Mock()
        mock_checkpoint_manager.load_checkpoints.return_value = []
        mock_get_samples.return_value = []
        mock_assembler.return_value = Mock()

        # Create dataset with action recognition task mode
        GraphDataset(
            config=mock_config,
            root_dir=mock_root_dir,
            split="train",
            task_mode="action_recognition",
        )

        # Verify get_samples was called with action recognition parameters
        expected_calls = mock_get_samples.call_args_list
        assert len(expected_calls) >= 1

        # Check that action recognition parameters were passed
        for call in expected_calls:
            kwargs = call.kwargs
            if kwargs.get("task_mode") == "action_recognition":
                assert "action_completion_ratio" in kwargs
                assert "min_nodes_threshold" in kwargs
                assert "visit_lookback_frames" in kwargs
                assert kwargs["action_completion_ratio"] == 0.8
                assert kwargs["min_nodes_threshold"] == 2
                assert kwargs["visit_lookback_frames"] == 30

    @patch("gazegraph.training.dataset.graph_dataset.CheckpointManager")
    @patch("gazegraph.training.dataset.graph_dataset.VideoMetadata")
    @patch("gazegraph.training.dataset.graph_dataset.get_samples")
    @patch("gazegraph.training.dataset.graph_dataset.create_graph_assembler")
    def test_future_actions_no_extra_params(
        self,
        mock_assembler,
        mock_get_samples,
        mock_metadata_class,
        mock_checkpoint_manager,
        mock_config,
        mock_root_dir,
    ):
        """Test that future actions mode doesn't add extra parameters."""
        # Setup mocks
        mock_metadata_class.return_value = Mock()
        mock_checkpoint_manager.load_checkpoints.return_value = []
        mock_get_samples.return_value = []
        mock_assembler.return_value = Mock()

        # Create dataset with future actions task mode
        GraphDataset(
            config=mock_config,
            root_dir=mock_root_dir,
            split="train",
            task_mode="future_actions",
        )

        # Verify get_samples was called without action recognition parameters
        expected_calls = mock_get_samples.call_args_list
        assert len(expected_calls) >= 1

        for call in expected_calls:
            kwargs = call.kwargs
            if kwargs.get("task_mode") == "future_actions":
                assert "action_completion_ratio" not in kwargs
                assert "min_nodes_threshold" not in kwargs
                assert "visit_lookback_frames" not in kwargs

    @patch("gazegraph.training.dataset.graph_dataset.CheckpointManager")
    @patch("gazegraph.training.dataset.graph_dataset.VideoMetadata")
    @patch("gazegraph.training.dataset.graph_dataset.get_samples")
    @patch("gazegraph.training.dataset.graph_dataset.create_graph_assembler")
    def test_default_parameter_values(
        self,
        mock_assembler,
        mock_get_samples,
        mock_metadata_class,
        mock_checkpoint_manager,
        mock_root_dir,
    ):
        """Test default parameter values when not specified in config."""
        # Config without action recognition parameters
        minimal_config = DotDict(
            {
                "dataset": {
                    "sampling": {
                        "strategy": "all",
                        "samples_per_video": 1,
                        "allow_duplicates": False,
                        "oversampling": False,
                    }
                },
                "training": {"val_timestamps": [0.25, 0.5, 0.75]},
            }
        )

        # Setup mocks
        mock_metadata_class.return_value = Mock()
        mock_checkpoint_manager.load_checkpoints.return_value = []
        mock_get_samples.return_value = []
        mock_assembler.return_value = Mock()

        # Create dataset with action recognition task mode
        GraphDataset(
            config=minimal_config,
            root_dir=mock_root_dir,
            split="train",
            task_mode="action_recognition",
        )

        # Verify default values were used
        expected_calls = mock_get_samples.call_args_list
        assert len(expected_calls) >= 1

        for call in expected_calls:
            kwargs = call.kwargs
            if kwargs.get("task_mode") == "action_recognition":
                assert kwargs.get("action_completion_ratio") == 1.0  # Default
                assert kwargs.get("min_nodes_threshold") == 1  # Default
                assert kwargs.get("visit_lookback_frames") == 0  # Default

    @patch("gazegraph.training.dataset.graph_dataset.CheckpointManager")
    @patch("gazegraph.training.dataset.graph_dataset.VideoMetadata")
    @patch("gazegraph.training.dataset.graph_dataset.get_samples")
    @patch("gazegraph.training.dataset.graph_dataset.create_graph_assembler")
    def test_validation_sampling_for_recognition_tasks(
        self,
        mock_assembler,
        mock_get_samples,
        mock_metadata_class,
        mock_checkpoint_manager,
        mock_config,
        mock_root_dir,
    ):
        """Test that validation sampling for recognition tasks uses all annotated actions."""
        # Create validation directory structure
        val_dir = Path(mock_root_dir) / "val"
        val_dir.mkdir(exist_ok=True)

        # Create mock checkpoint file
        checkpoint_file = val_dir / "test_video_graph.pth"
        checkpoint_file.touch()

        # Create mock checkpoint
        mock_checkpoint = Mock()
        mock_checkpoint.video_name = "test_video"
        mock_checkpoint.frame_number = 100
        mock_checkpoint.video_length = 1000

        # Setup mocks
        mock_metadata_class.return_value = Mock()
        mock_checkpoint_manager.load_checkpoints.return_value = [
            mock_checkpoint
        ]  # Provide checkpoints
        mock_get_samples.return_value = []
        mock_assembler.return_value = Mock()

        # Create validation dataset with action recognition task mode
        GraphDataset(
            config=mock_config,
            root_dir=mock_root_dir,
            split="val",  # Validation split
            task_mode="action_recognition",
        )

        # Verify get_samples was called for validation sampling
        expected_calls = mock_get_samples.call_args_list
        assert len(expected_calls) >= 1

        # Check that validation sampling uses the correct parameters
        validation_call = None
        for call in expected_calls:
            kwargs = call.kwargs
            if kwargs.get("task_mode") == "action_recognition":
                validation_call = kwargs
                break

        assert validation_call is not None
        # Validation should use "all" strategy to get all annotated actions
        assert validation_call["strategy"] == "all"
        assert validation_call["samples_per_video"] == 0  # Use all available
        assert not validation_call["allow_duplicates"]
        assert not validation_call["oversampling"]
        # Should include action recognition parameters
        assert "action_completion_ratio" in validation_call
        assert "min_nodes_threshold" in validation_call
        assert "visit_lookback_frames" in validation_call
