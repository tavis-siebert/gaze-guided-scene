"""
Unit tests for error handling and edge cases in the ActionRecord class.
"""

import pytest
from unittest.mock import patch, MagicMock
import os

from gazegraph.datasets.egtea_gaze.action_record import ActionRecord

@pytest.mark.unit
class TestActionRecordErrors:
    """Tests for error handling in the ActionRecord class."""
    
    def test_load_name_mappings_missing_files(self):
        """Test error handling when name mapping files are missing."""
        # Create a mock config
        config = MagicMock()
        config.dataset.egtea.action_annotations = "/nonexistent/path"
        
        with patch("gazegraph.datasets.egtea_gaze.action_record.get_config", return_value=config):
            # Test missing verb file
            with patch("os.path.exists") as mock_exists:
                # First call for verb file returns False (file doesn't exist)
                mock_exists.return_value = False
                
                with pytest.raises(FileNotFoundError, match="Verb index file not found"):
                    ActionRecord._load_name_mappings()
            
            # Test missing noun file
            with patch("os.path.exists") as mock_exists:
                # First call for verb file returns True, second call for noun file returns False
                mock_exists.side_effect = lambda path: "verb_idx.txt" in path
                
                # Mock the open function for verb file
                mock_open = MagicMock()
                mock_open.return_value.__enter__.return_value.readlines.return_value = []
                
                with patch("builtins.open", mock_open):
                    with pytest.raises(FileNotFoundError, match="Noun index file not found"):
                        ActionRecord._load_name_mappings()
    
    def test_load_all_records_file_not_found(self):
        """Test error handling when split files are not found."""
        # Create a mock config
        config = MagicMock()
        config.dataset.ego_topo.splits.train = "/nonexistent/train.txt"
        config.dataset.ego_topo.splits.val = "/nonexistent/val.txt"
        
        with patch("gazegraph.datasets.egtea_gaze.action_record.get_config", return_value=config):
            # Mock os.path.exists to return True so it attempts to open the file
            with patch("os.path.exists", return_value=True):
                # Mock open to raise FileNotFoundError
                with patch("builtins.open", side_effect=FileNotFoundError("No such file")):
                    with pytest.raises(FileNotFoundError):
                        ActionRecord._load_all_records()
    
    def test_load_all_records_invalid_format(self):
        """Test error handling with invalid record format in split files."""
        # Create a mock config
        config = MagicMock()
        config.dataset.ego_topo.splits.train = "train.txt"
        config.dataset.ego_topo.splits.val = "val.txt"
        
        # Mock the file handle that will be returned by the 'open' call
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__iter__.return_value = ["invalid line"]
        
        # Set up mocks - importantly we're letting the actual ActionRecord constructor run
        # but with invalid data from our mock file
        with patch("gazegraph.datasets.egtea_gaze.action_record.get_config", return_value=config), \
             patch("os.path.exists", return_value=True), \
             patch("builtins.open", return_value=mock_file):
            
            # The error will occur when trying to parse the invalid line
            with pytest.raises(Exception):
                ActionRecord._load_all_records()
    
    def test_action_name_with_none_values(self):
        """Test action_name property when verb or noun name is None."""
        with patch.object(ActionRecord, '_ensure_initialized'):
            # Case 1: Both verb and noun are None
            record = ActionRecord(["video_1", "10", "30", "99", "99"])
            # Set empty name mappings
            ActionRecord._verb_id_to_name = {}
            ActionRecord._noun_id_to_name = {}
            
            assert record.verb_name is None
            assert record.noun_name is None
            assert record.action_name is None
            
            # Case 2: Only verb has a name
            ActionRecord._verb_id_to_name = {99: "unknown_verb"}
            assert record.verb_name == "unknown_verb"
            assert record.noun_name is None
            assert record.action_name is None
            
            # Case 3: Only noun has a name
            ActionRecord._verb_id_to_name = {}
            ActionRecord._noun_id_to_name = {99: "unknown_noun"}
            assert record.verb_name is None
            assert record.noun_name == "unknown_noun"
            assert record.action_name is None
    
    def test_ensure_initialized_idempotent(self):
        """Test that _ensure_initialized is idempotent and only runs initialization once."""
        with patch.object(ActionRecord, '_load_name_mappings') as mock_load_names, \
             patch.object(ActionRecord, '_load_all_records') as mock_load_records, \
             patch.object(ActionRecord, '_compute_and_set_action_mapping') as mock_compute_mapping:
            
            # Setup for successful initialization
            mock_load_records.return_value = ({}, [])
            
            # Reset initialization state
            ActionRecord._is_initialized = False
            
            # First call should trigger initialization
            ActionRecord._ensure_initialized()
            assert ActionRecord._is_initialized is True
            mock_load_names.assert_called_once()
            mock_load_records.assert_called_once()
            mock_compute_mapping.assert_called_once()
            
            # Reset call counts
            mock_load_names.reset_mock()
            mock_load_records.reset_mock()
            mock_compute_mapping.reset_mock()
            
            # Second call should not trigger initialization again
            ActionRecord._ensure_initialized()
            mock_load_names.assert_not_called()
            mock_load_records.assert_not_called()
            mock_compute_mapping.assert_not_called()
            
    def test_str_representation_fallback(self):
        """Test string representation fallback when names are unavailable."""
        with patch.object(ActionRecord, '_ensure_initialized'):
            # Set empty name mappings
            ActionRecord._verb_id_to_name = {}
            ActionRecord._noun_id_to_name = {}
            
            record = ActionRecord(["video_1", "10", "30", "1", "3"])
            
            # String should contain tuple representation
            str_repr = str(record)
            assert "(1, 3)" in str_repr
            assert "video_1" in str_repr
            assert "start=10" in str_repr
            assert "end=30" in str_repr 