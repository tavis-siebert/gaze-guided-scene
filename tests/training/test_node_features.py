import pytest
import torch
from unittest.mock import MagicMock, patch

from gazegraph.graph.checkpoint_manager import GraphCheckpoint
from gazegraph.training.dataset.node_features import (
    OneHotNodeFeatureExtractor,
    ROIEmbeddingNodeFeatureExtractor,
    ObjectLabelEmbeddingNodeFeatureExtractor,
    get_node_feature_extractor
)


class TestNodeFeatureExtractors:
    @pytest.fixture
    def mock_checkpoint(self):
        # Create a mock checkpoint
        checkpoint = MagicMock(spec=GraphCheckpoint)
        checkpoint.nodes = {
            0: {
                "object_label": "cup",
                "visits": [(10, 20), (30, 40)]
            },
            1: {
                "object_label": "bowl",
                "visits": [(15, 25)]
            }
        }
        checkpoint.object_labels_to_id = {"cup": 0, "bowl": 1}
        checkpoint.num_object_classes = 2
        checkpoint.non_black_frame_count = 100
        checkpoint.frame_number = 50
        checkpoint.video_length = 200
        checkpoint.video_name = "OP01-R01-PastaSalad"
        
        return checkpoint

    def test_one_hot_extractor(self, mock_checkpoint):
        """Test the OneHotNodeFeatureExtractor"""
        extractor = OneHotNodeFeatureExtractor()
        features = extractor.extract_features(mock_checkpoint)
        
        # Check shape: 2 nodes, 5 temporal features + 2 one-hot classes
        assert features.shape == (2, 7)
        
        # Check one-hot encoding for first node (cup, class index 0)
        assert features[0, 5] == 1  # One-hot for class 0
        assert features[0, 6] == 0  # One-hot for class 1
        
        # Check one-hot encoding for second node (bowl, class index 1)
        assert features[1, 5] == 0  # One-hot for class 0
        assert features[1, 6] == 1  # One-hot for class 1

    @patch('gazegraph.training.dataset.node_features.NodeEmbeddings')
    def test_roi_embedding_extractor(self, mock_node_embeddings_class, mock_checkpoint):
        """Test the ROIEmbeddingNodeFeatureExtractor"""
        # Setup mock for NodeEmbeddings
        mock_node_embeddings = MagicMock()
        mock_node_embeddings_class.return_value = mock_node_embeddings
        
        # Mock the get_object_node_embedding_roi method to return a fixed embedding
        embedding_dim = 16
        mock_embedding = torch.ones(embedding_dim)
        mock_node_embeddings.get_object_node_embedding_roi.return_value = mock_embedding
        
        # Create the extractor
        extractor = ROIEmbeddingNodeFeatureExtractor(device="cpu", embedding_dim=embedding_dim)
        
        # Mock the tracer and video
        mock_tracer = MagicMock()
        mock_video = MagicMock()
        extractor.set_context(mock_tracer, mock_video)
        
        # Extract features
        features = extractor.extract_features(mock_checkpoint)
        
        # Check shape: 2 nodes, 5 temporal features + embedding_dim
        assert features.shape == (2, 5 + embedding_dim)
        
        # Check that the embedding part is filled with ones
        assert torch.all(features[0, 5:] == 1)
        assert torch.all(features[1, 5:] == 1)
        
        # Check that the method was called with correct parameters
        mock_node_embeddings.get_object_node_embedding_roi.assert_any_call(
            mock_checkpoint, mock_tracer, mock_video, 0
        )
        mock_node_embeddings.get_object_node_embedding_roi.assert_any_call(
            mock_checkpoint, mock_tracer, mock_video, 1
        )

    @patch('gazegraph.training.dataset.node_features.NodeEmbeddings')
    def test_object_label_embedding_extractor(self, mock_node_embeddings_class, mock_checkpoint):
        """Test the ObjectLabelEmbeddingNodeFeatureExtractor"""
        # Setup mock for NodeEmbeddings
        mock_node_embeddings = MagicMock()
        mock_node_embeddings_class.return_value = mock_node_embeddings
        
        # Mock the get_object_node_embedding_label method to return a fixed embedding
        embedding_dim = 16
        mock_embedding = torch.ones(embedding_dim)
        mock_node_embeddings.get_object_node_embedding_label.return_value = mock_embedding
        
        # Create the extractor
        extractor = ObjectLabelEmbeddingNodeFeatureExtractor(device="cpu", embedding_dim=embedding_dim)
        
        # Extract features
        features = extractor.extract_features(mock_checkpoint)
        
        # Check shape: 2 nodes, 5 temporal features + embedding_dim
        assert features.shape == (2, 5 + embedding_dim)
        
        # Check that the embedding part is filled with ones
        assert torch.all(features[0, 5:] == 1)
        assert torch.all(features[1, 5:] == 1)
        
        # Check that the method was called with correct parameters
        mock_node_embeddings.get_object_node_embedding_label.assert_any_call(
            mock_checkpoint, 0
        )
        mock_node_embeddings.get_object_node_embedding_label.assert_any_call(
            mock_checkpoint, 1
        )

    def test_get_node_feature_extractor(self):
        """Test the factory function for node feature extractors"""
        # Test one-hot extractor
        extractor = get_node_feature_extractor("one-hot")
        assert isinstance(extractor, OneHotNodeFeatureExtractor)
        
        # Test ROI embedding extractor
        extractor = get_node_feature_extractor("roi-embeddings", device="cpu", embedding_dim=32)
        assert isinstance(extractor, ROIEmbeddingNodeFeatureExtractor)
        assert extractor.embedding_dim == 32
        
        # Test object label embedding extractor
        extractor = get_node_feature_extractor("object-label-embeddings", device="cpu", embedding_dim=64)
        assert isinstance(extractor, ObjectLabelEmbeddingNodeFeatureExtractor)
        assert extractor.embedding_dim == 64
        
        # Test invalid type
        with pytest.raises(ValueError):
            get_node_feature_extractor("invalid-type")

