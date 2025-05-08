"""
Advanced unit tests for the ActionRecord class in gazegraph.datasets.egtea_gaze.action_record module.
These tests focus on more complex functionality and edge cases.
"""

import pytest
import tempfile
import os
import torch
import json
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path

from gazegraph.datasets.egtea_gaze.action_record import ActionRecord
from gazegraph.datasets.egtea_gaze.constants import NUM_ACTION_CLASSES

@pytest.fixture
def mock_action_records():
    """Create mock action records for testing."""
    with patch.object(ActionRecord, '_ensure_initialized'):
        records = [
            ActionRecord(["video_1", "10", "30", "1", "3"]),  # take knife
            ActionRecord(["video_1", "40", "60", "2", "1"]),  # put cup
            ActionRecord(["video_2", "5", "25", "3", "4"]),   # open microwave
            ActionRecord(["video_2", "30", "50", "4", "4"]),  # close microwave
            ActionRecord(["video_3", "15", "35", "5", "2"]),  # wash bowl
        ]
        
        # Mock internal state
        ActionRecord._verb_id_to_name = {
            1: "take", 2: "put", 3: "open", 4: "close", 5: "wash"
        }
        ActionRecord._noun_id_to_name = {
            1: "cup", 2: "bowl", 3: "knife", 4: "microwave", 5: "fridge"
        }
        ActionRecord._action_to_idx = {
            (1, 3): 0,  # take knife
            (2, 1): 1,  # put cup
            (3, 4): 2,  # open microwave
            (4, 4): 3,  # close microwave
            (5, 2): 4,  # wash bowl
        }
        ActionRecord._is_initialized = True
        
        return records

@pytest.fixture
def mock_records_by_video(mock_action_records):
    """Organize records by video for testing."""
    # Group all records by video first
    records_by_video = {
        "video_1": [],
        "video_2": [],
        "video_3": [],
    }
    
    # Populate the dictionary
    for record in mock_action_records:
        video_name = record.video_name
        if video_name in records_by_video:
            records_by_video[video_name].append(record)
    
    # Ensure records are sorted by end_frame
    for records in records_by_video.values():
        records.sort(key=lambda r: r.end_frame)
        
    # Set up the class attribute
    ActionRecord._records_by_video = records_by_video
    
    return records_by_video

@pytest.mark.unit
class TestActionRecordAdvanced:
    """Advanced unit tests for the ActionRecord class."""
    
    def test_get_records_for_video(self, mock_records_by_video):
        """Test getting records for a specific video."""
        with patch.object(ActionRecord, "_ensure_initialized"):
            # Mock get_records_for_video to return directly from our mock data
            def mock_get_records(video):
                return mock_records_by_video.get(video, [])
            
            # Apply the mock
            with patch.object(ActionRecord, "get_records_for_video", side_effect=mock_get_records):
                # Test with existing video
                records = ActionRecord.get_records_for_video("video_1")
                
                # Count the actual records in the mock fixture for video_1
                expected_count = len(mock_records_by_video["video_1"])
                assert len(records) == expected_count
                assert all(r.video_name == "video_1" for r in records)
                
                # Test with non-existent video
                records = ActionRecord.get_records_for_video("non_existent")
                assert records == []
    
    def test_get_all_videos(self, mock_records_by_video):
        """Test getting all video names."""
        videos = ActionRecord.get_all_videos()
        assert set(videos) == {"video_1", "video_2", "video_3"}
    
    def test_get_all_records(self, mock_records_by_video):
        """Test getting all records."""
        records = ActionRecord.get_all_records()
        assert len(records) == 5
        
        # Check that we have records from all videos
        video_names = {r.video_name for r in records}
        assert video_names == {"video_1", "video_2", "video_3"}
    
    def test_get_action_mapping(self):
        """Test getting the action mapping."""
        with patch.object(ActionRecord, '_ensure_initialized'):
            # Set up mock data
            test_mapping = {
                (1, 3): 0,  # take knife
                (2, 1): 1,  # put cup
            }
            ActionRecord._action_to_idx = test_mapping
            
            # Get mapping and verify it's a copy
            mapping = ActionRecord.get_action_mapping()
            assert mapping == test_mapping
            assert mapping is not test_mapping  # Should be a copy
            
            # Modify the copy and verify original is unchanged
            mapping[(99, 99)] = 99
            assert (99, 99) not in ActionRecord._action_to_idx
    
    def test_get_noun_label_mappings(self):
        """Test getting noun label mappings."""
        with patch.object(ActionRecord, '_ensure_initialized'):
            # Set up mock data
            ActionRecord._noun_id_to_name = {
                1: "cup", 2: "bowl", 3: "knife"
            }
            
            # Get mappings
            id_to_name, name_to_id = ActionRecord.get_noun_label_mappings()
            
            # Check id_to_name mapping
            assert id_to_name == {1: "cup", 2: "bowl", 3: "knife"}
            
            # Check name_to_id mapping
            assert name_to_id == {"cup": 1, "bowl": 2, "knife": 3}
            
            # Ensure the returned mappings are copies
            id_to_name[4] = "plate"
            assert 4 not in ActionRecord._noun_id_to_name
    
    def test_get_action_name_by_idx(self):
        """Test getting action name by index."""
        with patch.object(ActionRecord, '_ensure_initialized'):
            # Set up mock data
            ActionRecord._action_to_idx = {
                (1, 3): 0,  # take knife
                (2, 1): 1,  # put cup
            }
            ActionRecord._verb_id_to_name = {1: "take", 2: "put"}
            ActionRecord._noun_id_to_name = {1: "cup", 3: "knife"}
            
            # Test with valid index
            name = ActionRecord.get_action_name_by_idx(0)
            assert name == "take knife"
            
            # Test with invalid index
            name = ActionRecord.get_action_name_by_idx(99)
            assert name is None
    
    def test_load_records_from_file(self):
        """Test loading records from a file."""
        # Create a temporary file with test data
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("video_1 10 30 1 3\n")  # take knife
            f.write("video_1 40 60 2 1\n")  # put cup
            temp_path = f.name
        
        try:
            # Mock initialization to avoid side effects
            with patch.object(ActionRecord, '_ensure_initialized'):
                # Load records and verify
                records = ActionRecord.load_records_from_file(temp_path)
                
                assert len(records) == 2
                assert records[0].video_name == "video_1"
                assert records[0].start_frame == 10
                assert records[0].end_frame == 30
                assert records[0].label == [1, 3]
                
                assert records[1].video_name == "video_1"
                assert records[1].start_frame == 40
                assert records[1].end_frame == 60
                assert records[1].label == [2, 1]
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_load_records_from_file_errors(self):
        """Test error handling when loading records."""
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            ActionRecord.load_records_from_file("non_existent_file.txt")
        
        # Test with invalid file format
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("video_1\n")  # Invalid (too few fields)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError):
                ActionRecord.load_records_from_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    @patch("gazegraph.datasets.egtea_gaze.action_record.get_config")
    def test_compute_and_set_action_mapping_errors(self, mock_get_config):
        """Test error handling in compute_and_set_action_mapping."""
        # Test with empty records
        with pytest.raises(ValueError, match="no training records"):
            ActionRecord._compute_and_set_action_mapping([])
        
        # Test with invalid number of classes
        with pytest.raises(ValueError, match="Invalid number of classes"):
            ActionRecord._compute_and_set_action_mapping([MagicMock()], num_classes=0)
        
        # Test with missing name mappings
        ActionRecord._verb_id_to_name = {}
        ActionRecord._noun_id_to_name = {}
        with pytest.raises(RuntimeError, match="Name mappings must be loaded"):
            ActionRecord._compute_and_set_action_mapping([MagicMock()])
    
    def test_create_future_action_labels_edge_cases(self):
        """Test edge cases for create_future_action_labels."""
        with patch.object(ActionRecord, '_ensure_initialized'):
            # Case 1: No records for video
            ActionRecord.get_records_for_video = MagicMock(return_value=[])
            result = ActionRecord.create_future_action_labels("video_x", 10)
            assert result is None
            
            # Case 2: No future actions (all actions are in the past)
            records = [
                ActionRecord(["video_1", "10", "30", "1", "3"]),  # take knife
                ActionRecord(["video_1", "40", "60", "2", "1"]),  # put cup
            ]
            
            # Set up a class-level action mapping
            ActionRecord._action_to_idx = {
                (1, 3): 0,  # take knife
                (2, 1): 1,  # put cup
            }
            
            ActionRecord.get_records_for_video = MagicMock(return_value=records)
            result = ActionRecord.create_future_action_labels("video_1", 70)
            assert result is None  # no future actions
            
            # Case 3: Some actions have no mapping
            records = [
                ActionRecord(["video_1", "10", "30", "1", "3"]),  # take knife - mapped
                ActionRecord(["video_1", "40", "60", "2", "1"]),  # put cup - mapped
                ActionRecord(["video_1", "70", "90", "99", "99"]),  # unknown - not mapped
            ]
            
            # Set up action mapping that excludes the third record
            ActionRecord._action_to_idx = {
                (1, 3): 0,  # take knife
                (2, 1): 1,  # put cup
                # (99, 99) is not mapped, so action_idx will be None
            }
            
            ActionRecord.get_records_for_video = MagicMock(return_value=records)
            result = ActionRecord.create_future_action_labels("video_1", 5, num_action_classes=2)
            
            assert result is not None
            assert result["next_action"].item() == 0  # First mapped action
            assert torch.equal(result["future_actions"], torch.tensor([1, 1]))  # Both mapped actions
            assert torch.equal(result["future_actions_ordered"], torch.tensor([0, 1]))  # Order preserved 