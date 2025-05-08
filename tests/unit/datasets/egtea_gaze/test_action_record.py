"""
Unit tests for the ActionRecord class in the gazegraph.datasets.egtea_gaze.action_record module.
"""

import pytest
import tempfile
import os
import torch
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path

from gazegraph.datasets.egtea_gaze.action_record import ActionRecord
from gazegraph.datasets.egtea_gaze.constants import NUM_ACTION_CLASSES

@pytest.fixture
def mock_config():
    """Mock configuration with test data paths."""
    config = MagicMock()
    config.dataset.egtea.action_annotations = "test_annotations"
    
    # Set up mock splits data
    config.dataset.ego_topo.splits = MagicMock()
    config.dataset.ego_topo.splits.train = "test_train_split.txt"
    config.dataset.ego_topo.splits.val = "test_val_split.txt"
    
    return config

@pytest.fixture
def mock_verb_index_file():
    """Create a temporary verb index file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("take 1\n")
        f.write("put 2\n")
        f.write("open 3\n")
        f.write("close 4\n")
        f.write("wash 5\n")
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)

@pytest.fixture
def mock_noun_index_file():
    """Create a temporary noun index file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("cup 1\n")
        f.write("bowl 2\n")
        f.write("knife 3\n")
        f.write("microwave 4\n")
        f.write("fridge 5\n")
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)

@pytest.fixture
def mock_train_split_file():
    """Create a temporary train split file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("video_1\t10\t30\t1\t3\n")  # take knife
        f.write("video_1\t40\t60\t2\t1\n")  # put cup
        f.write("video_2\t5\t25\t3\t4\n")   # open microwave
        f.write("video_2\t30\t50\t4\t4\n")  # close microwave
        f.write("video_3\t15\t35\t5\t2\n")  # wash bowl
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)

@pytest.fixture
def mock_val_split_file():
    """Create a temporary val split file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("video_4\t5\t25\t1\t1\n")   # take cup
        f.write("video_4\t30\t50\t4\t5\n")  # close fridge
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)

@pytest.fixture
def setup_mock_files(mock_verb_index_file, mock_noun_index_file, mock_train_split_file, mock_val_split_file, monkeypatch):
    """Set up mock files and directories for testing."""
    # Create a mock directory structure
    os.makedirs("test_annotations", exist_ok=True)
    
    # Copy mock files to test directory
    monkeypatch.setattr("os.path.exists", lambda path: True)
    
    # Mock the open function to return our test files
    original_open = open
    
    def mock_open(file, *args, **kwargs):
        if file == "test_annotations/verb_idx.txt":
            return original_open(mock_verb_index_file, *args, **kwargs)
        elif file == "test_annotations/noun_idx.txt":
            return original_open(mock_noun_index_file, *args, **kwargs)
        elif file == "test_train_split.txt":
            return original_open(mock_train_split_file, *args, **kwargs)
        elif file == "test_val_split.txt":
            return original_open(mock_val_split_file, *args, **kwargs)
        else:
            return original_open(file, *args, **kwargs)
    
    # Apply the patch
    monkeypatch.setattr("builtins.open", mock_open)
    
    yield
    
    # Clean up
    try:
        os.rmdir("test_annotations")
    except:
        pass

@pytest.mark.unit
class TestActionRecord:
    """Unit tests for the ActionRecord class."""
    
    def test_init_from_row(self):
        """Test initialization from a data row."""
        # Mock _ensure_initialized to avoid actual initialization
        with patch.object(ActionRecord, '_ensure_initialized'):
            record = ActionRecord(["video_1", "10", "30", "1", "3"])
            
            assert record.video_name == "video_1"
            assert record.start_frame == 10
            assert record.end_frame == 30
            assert record.label == [1, 3]
            assert record.verb_id == 1
            assert record.noun_id == 3
            assert record.num_frames == 21
            assert record.action_tuple == (1, 3)
    
    def test_name_properties(self):
        """Test verb and noun name properties."""
        with patch.object(ActionRecord, '_ensure_initialized'):
            # Set up mock name mappings
            ActionRecord._verb_id_to_name = {1: "take", 2: "put"}
            ActionRecord._noun_id_to_name = {1: "cup", 3: "knife"}
            
            record = ActionRecord(["video_1", "10", "30", "1", "3"])
            
            assert record.verb_name == "take"
            assert record.noun_name == "knife"
            assert record.action_name == "take knife"
            
            # Test with unmapped IDs
            record2 = ActionRecord(["video_1", "10", "30", "99", "99"])
            assert record2.verb_name is None
            assert record2.noun_name is None
            assert record2.action_name is None
    
    @pytest.mark.parametrize("verb_id,noun_id,expected_idx", [
        (1, 3, 0),  # take knife -> index 0
        (4, 4, 3),  # close microwave -> index 3
        (99, 99, None),  # unknown action -> None
    ])
    def test_action_idx(self, verb_id, noun_id, expected_idx):
        """Test action_idx property with various inputs."""
        with patch.object(ActionRecord, '_ensure_initialized'):
            # Set up mock action mapping
            ActionRecord._action_to_idx = {
                (1, 3): 0,  # take knife
                (2, 1): 1,  # put cup
                (3, 4): 2,  # open microwave
                (4, 4): 3,  # close microwave
                (5, 2): 4,  # wash bowl
            }
            
            record = ActionRecord(["video_1", "10", "30", str(verb_id), str(noun_id)])
            assert record.action_idx == expected_idx
    
    @patch("gazegraph.datasets.egtea_gaze.action_record.get_config")
    @patch.object(ActionRecord, '_load_name_mappings')
    @patch.object(ActionRecord, '_load_all_records')
    @patch.object(ActionRecord, '_compute_and_set_action_mapping')
    def test_ensure_initialized(self, mock_compute, mock_load_records, mock_load_mappings, mock_get_config, mock_config):
        """Test the _ensure_initialized method."""
        mock_get_config.return_value = mock_config
        mock_load_records.return_value = ({"video_1": []}, [])
        
        # Reset class variables
        ActionRecord._is_initialized = False
        
        # Call the method
        ActionRecord._ensure_initialized()
        
        # Verify method calls
        mock_load_mappings.assert_called_once()
        mock_load_records.assert_called_once()
        mock_compute.assert_called_once()
        
        # Verify class state
        assert ActionRecord._is_initialized is True
        
        # Call again - should not trigger any new calls
        mock_load_mappings.reset_mock()
        mock_load_records.reset_mock()
        mock_compute.reset_mock()
        
        ActionRecord._ensure_initialized()
        
        mock_load_mappings.assert_not_called()
        mock_load_records.assert_not_called()
        mock_compute.assert_not_called()
    
    def test_load_name_mappings(self, setup_mock_files, mock_config):
        """Test loading name mappings from files."""
        with patch("gazegraph.datasets.egtea_gaze.action_record.get_config", return_value=mock_config):
            # Reset class variables
            ActionRecord._verb_id_to_name = {}
            ActionRecord._noun_id_to_name = {}
            
            # Call the method
            ActionRecord._load_name_mappings()
            
            # Verify loaded data
            assert ActionRecord._verb_id_to_name == {1: "take", 2: "put", 3: "open", 4: "close", 5: "wash"}
            assert ActionRecord._noun_id_to_name == {1: "cup", 2: "bowl", 3: "knife", 4: "microwave", 5: "fridge"}
    
    def test_compute_action_mapping(self):
        """Test computation of action mapping from records."""
        with patch.object(ActionRecord, '_ensure_initialized'):
            # Create test records
            records = [
                ActionRecord(["video_1", "10", "30", "1", "3"]),  # take knife (occurs twice)
                ActionRecord(["video_2", "5", "25", "1", "3"]),   # take knife (occurs twice)
                ActionRecord(["video_1", "40", "60", "2", "1"]),  # put cup (occurs once)
                ActionRecord(["video_2", "30", "50", "4", "4"]),  # close microwave (occurs once)
            ]
            
            # Directly mock the Counter result instead of trying to set the property
            expected_mapping = {
                (1, 3): 0,  # take knife
                (2, 1): 1,  # put cup
                (4, 4): 2,  # close microwave
            }
            
            with patch('collections.Counter') as mock_counter:
                # Configure the Counter to return our desired counts
                counter_instance = mock_counter.return_value
                counter_instance.items.return_value = [
                    ((1, 3), 2),  # take knife occurs twice
                    ((2, 1), 1),  # put cup occurs once
                    ((4, 4), 1),  # close microwave occurs once
                ]
                
                # Call the method with test data
                mapping = ActionRecord._compute_action_mapping(records, num_classes=3)
                
                # Verify the mapping
                assert len(mapping) == 3  # Should have 3 actions
                assert mapping == expected_mapping  # Check the complete mapping
    
    @patch("gazegraph.datasets.egtea_gaze.action_record.get_config")
    def test_load_all_records(self, mock_get_config, setup_mock_files, mock_config):
        """Test loading records from split files."""
        mock_get_config.return_value = mock_config
        
        # Call the method
        records_by_video, train_records = ActionRecord._load_all_records()
        
        # Verify loaded data
        assert len(records_by_video) == 4  # 4 unique videos
        assert "video_1" in records_by_video
        assert "video_2" in records_by_video
        assert "video_3" in records_by_video
        assert "video_4" in records_by_video
        
        assert len(train_records) == 5  # 5 training records
        
        # Check that records are sorted by end_frame
        video_1_records = records_by_video["video_1"]
        assert len(video_1_records) == 2
        assert video_1_records[0].end_frame < video_1_records[1].end_frame
    
    def test_get_action_names(self):
        """Test getting action names from indices."""
        with patch.object(ActionRecord, '_ensure_initialized'):
            # Set up mock data
            ActionRecord._action_to_idx = {
                (1, 3): 0,  # take knife
                (2, 1): 1,  # put cup
            }
            ActionRecord._verb_id_to_name = {1: "take", 2: "put"}
            ActionRecord._noun_id_to_name = {1: "cup", 3: "knife"}
            
            # Call the method
            action_names = ActionRecord.get_action_names()
            
            # Verify results
            assert action_names == {
                0: "take knife",
                1: "put cup"
            }
    
    def test_create_future_action_labels(self):
        """Test creating future action labels."""
        with patch.object(ActionRecord, '_ensure_initialized'):
            # Set up mock data
            records = [
                ActionRecord(["video_1", "10", "30", "1", "3"]),  # take knife
                ActionRecord(["video_1", "40", "60", "2", "1"]),  # put cup
                ActionRecord(["video_1", "70", "90", "3", "4"]),  # open microwave
            ]
            
            # Instead of patching the property, mock get_action_mapping to return a specific mapping
            # that will naturally produce the action_idx values we want
            action_mapping = {
                (1, 3): 0,  # take knife
                (2, 1): 1,  # put cup
                (3, 4): 2,  # open microwave
            }
            
            # Set the class-level _action_to_idx directly
            ActionRecord._action_to_idx = action_mapping
            
            # Set up other class mocks
            ActionRecord.get_records_for_video = MagicMock(return_value=records)
            
            # Test case 1: Current frame before all actions
            result = ActionRecord.create_future_action_labels("video_1", 5, num_action_classes=3)
            assert result is not None
            assert result["next_action"].item() == 0  # take knife
            assert torch.equal(result["future_actions"], torch.tensor([1, 1, 1]))  # all actions in future
            assert torch.equal(result["future_actions_ordered"], torch.tensor([0, 1, 2]))  # order preserved
            
            # Test case 2: Current frame between actions
            result = ActionRecord.create_future_action_labels("video_1", 35, num_action_classes=3)
            assert result is not None
            assert result["next_action"].item() == 1  # put cup
            assert torch.equal(result["future_actions"], torch.tensor([0, 1, 1]))  # only put cup and open microwave
            assert torch.equal(result["future_actions_ordered"], torch.tensor([1, 2]))  # order preserved
            
            # Test case 3: Current frame after all actions
            result = ActionRecord.create_future_action_labels("video_1", 95, num_action_classes=3)
            assert result is None  # no future actions
    
    def test_str_representation(self):
        """Test string representation of action record."""
        with patch.object(ActionRecord, '_ensure_initialized'):
            # Set up mock name mappings
            ActionRecord._verb_id_to_name = {1: "take"}
            ActionRecord._noun_id_to_name = {3: "knife"}
            
            record = ActionRecord(["video_1", "10", "30", "1", "3"])
            
            # Check string representation with mapped names
            assert "take knife" in str(record)
            assert "video_1" in str(record)
            assert "start=10" in str(record)
            assert "end=30" in str(record)
            
            # Check string representation with unmapped IDs
            record2 = ActionRecord(["video_2", "5", "25", "99", "99"])
            assert "(99, 99)" in str(record2) 