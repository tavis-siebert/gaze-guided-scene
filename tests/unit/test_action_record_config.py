"""
Unit tests for ActionRecord's integration with the config system.
"""

import pytest
from unittest.mock import patch, MagicMock
import os
import tempfile

from gazegraph.datasets.egtea_gaze.action_record import ActionRecord

@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = MagicMock()
    
    # Set up egtea paths
    config.dataset.egtea.action_annotations = "test_annotations"
    
    # Set up ego_topo split paths
    config.dataset.ego_topo.splits = MagicMock()
    config.dataset.ego_topo.splits.train = "test_train_split.txt"
    config.dataset.ego_topo.splits.val = "test_val_split.txt"
    
    return config

@pytest.mark.unit
class TestActionRecordConfig:
    """Tests for ActionRecord integration with the config system."""
    
    def test_get_config(self, mock_config):
        """Test that get_config returns the config instance."""
        with patch("gazegraph.datasets.egtea_gaze.action_record.get_config", return_value=mock_config):
            # Reset the class config
            ActionRecord._config = None
            
            # Get config and check it's initialized
            config = ActionRecord.get_config()
            assert config is mock_config
            
            # Check caching behavior
            with patch("gazegraph.datasets.egtea_gaze.action_record.get_config") as mock_get_config:
                # Second call should use cached version
                config2 = ActionRecord.get_config()
                assert config2 is mock_config
                # get_config should not be called again
                mock_get_config.assert_not_called()
    
    def test_initialization_uses_config_paths(self, mock_config):
        """Test that initialization uses paths from config."""
        with patch("gazegraph.datasets.egtea_gaze.action_record.get_config", return_value=mock_config), \
             patch.object(ActionRecord, '_load_name_mappings') as mock_load_names, \
             patch.object(ActionRecord, '_load_all_records') as mock_load_records, \
             patch.object(ActionRecord, '_compute_and_set_action_mapping') as mock_compute_mapping:
            
            # Setup for successful initialization
            mock_load_records.return_value = ({}, [])
            
            # Reset initialization state
            ActionRecord._is_initialized = False
            
            # Initialize
            ActionRecord._ensure_initialized()
            
            # Verify config was used
            mock_load_names.assert_called_once()
            mock_load_records.assert_called_once()
            mock_compute_mapping.assert_called_once()
            
            # Check that _load_name_mappings was called without explicit base_dir parameter
            # When no base_dir is specified, it will use the config path by default
            mock_load_names.assert_called_once_with()
    
    def test_load_name_mappings_with_custom_path(self):
        """Test loading name mappings from a custom path."""
        # Create temporary files for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create verb index file
            verb_file = os.path.join(temp_dir, "verb_idx.txt")
            with open(verb_file, 'w') as f:
                f.write("custom_verb 1\n")
            
            # Create noun index file
            noun_file = os.path.join(temp_dir, "noun_idx.txt")
            with open(noun_file, 'w') as f:
                f.write("custom_noun 1\n")
            
            # Reset name mappings
            ActionRecord._verb_id_to_name = {}
            ActionRecord._noun_id_to_name = {}
            
            # Call the method with custom path
            ActionRecord._load_name_mappings(base_dir=temp_dir)
            
            # Verify the mappings were loaded from the custom path
            assert ActionRecord._verb_id_to_name == {1: "custom_verb"}
            assert ActionRecord._noun_id_to_name == {1: "custom_noun"}
    
    def test_config_path_precedence(self, mock_config):
        """Test that config paths take precedence when no base_dir is provided."""
        # Mock the load_name_mappings method to extract the base_dir
        original_load_name_mappings = ActionRecord._load_name_mappings
        
        captured_base_dir = None
        
        def mock_load_name_mappings(base_dir=None):
            nonlocal captured_base_dir
            captured_base_dir = base_dir
            # Do nothing else to avoid side effects
            
        with patch("gazegraph.datasets.egtea_gaze.action_record.get_config", return_value=mock_config), \
             patch.object(ActionRecord, '_load_name_mappings', side_effect=mock_load_name_mappings), \
             patch.object(ActionRecord, '_load_all_records', return_value=({}, [])), \
             patch.object(ActionRecord, '_compute_and_set_action_mapping'):
            
            # Reset initialization state
            ActionRecord._is_initialized = False
            
            # Initialize
            ActionRecord._ensure_initialized()
            
            # Verify config path was used (captured_base_dir should be None, meaning it used config)
            assert captured_base_dir is None
            
    def test_missing_config_raises_error(self):
        """Test that missing config raises an appropriate error."""
        with patch("gazegraph.datasets.egtea_gaze.action_record.get_config", side_effect=ImportError("Config not found")):
            # Reset the class config
            ActionRecord._config = None
            
            # Attempt to get config should propagate the error
            with pytest.raises(ImportError, match="Config not found"):
                ActionRecord.get_config() 