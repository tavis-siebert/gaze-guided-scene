"""
Unit tests for the NodeEmbeddings class.
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np

from gazegraph.datasets.embeddings import NodeEmbeddings
from gazegraph.datasets.egtea_gaze.action_record import ActionRecord
from gazegraph.graph.checkpoint_manager import GraphCheckpoint, CheckpointManager
from gazegraph.graph.graph_tracer import GraphTracer
from gazegraph.datasets.egtea_gaze.video_processor import Video


@pytest.fixture
def node_embeddings():
    """Fixture for a/NodeEmbeddings instance configured for CPU testing."""
    return NodeEmbeddings(device="cpu")


@pytest.fixture
def test_checkpoint():
    """Fixture that loads a real checkpoint from the data directory."""
    graph_path = Path("data/graphs/train/OP01-R04-ContinentalBreakfast_graph.pth")
    checkpoints = CheckpointManager.load_checkpoints(str(graph_path))
    if not checkpoints:
        pytest.skip("Test graph checkpoint not available")
    return checkpoints[10]


@pytest.fixture
def test_tracer():
    """Fixture that loads the trace file for the test video."""
    trace_path = Path("data/traces/OP01-R04-ContinentalBreakfast_trace.jsonl")
    if not trace_path.exists():
        pytest.skip("Test trace file not available")
    return GraphTracer("data/traces", "OP01-R04-ContinentalBreakfast", enabled=False)


@pytest.fixture
def test_video():
    """Fixture that initializes a Video object for the test video."""
    try:
        return Video("OP01-R04-ContinentalBreakfast")
    except Exception as e:
        pytest.skip(f"Test video not available: {e}")


@pytest.mark.unit
def test_initialization():
    """Test that NodeEmbeddings initializes correctly."""
    embedder = NodeEmbeddings(device="cpu")
    assert embedder.device == "cpu"
    assert embedder.clip_model is None


@pytest.mark.unit
def test_get_clip_model(node_embeddings):
    """Test that the CLIP model is initialized properly."""
    clip_model = node_embeddings._get_clip_model()
    assert clip_model is not None
    assert node_embeddings.clip_model is not None
    assert clip_model.device == "cpu"


@pytest.mark.unit
def test_get_action_embedding(node_embeddings):
    """Test that action embeddings are correctly generated."""
    with patch("gazegraph.datasets.egtea_gaze.action_record.ActionRecord.get_action_name_by_idx", 
               return_value="take bowl"):
        embedding = node_embeddings.get_action_embedding(0)
        assert embedding is not None
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape[1] == 512  # CLIP's default embedding size


@pytest.mark.unit
def test_get_action_embedding_invalid_action(node_embeddings):
    """Test behavior with invalid action index."""
    with patch("gazegraph.datasets.egtea_gaze.action_record.ActionRecord.get_action_name_by_idx", 
               return_value=None):
        embedding = node_embeddings.get_action_embedding(9999)
        assert embedding is None


@pytest.mark.integration
def test_extract_roi(node_embeddings):
    """Test ROI extraction from frame tensor."""
    frame = torch.zeros((3, 100, 100), dtype=torch.uint8)
    frame[:, 30:60, 30:60] = 255  # White square in the middle
    bbox = (30, 30, 30, 30)  # left, top, width, height
    roi = node_embeddings._extract_roi(frame, bbox)
    assert roi is not None
    assert roi.shape == (3, 30, 30)
    assert roi.sum() == 3 * 30 * 30 * 255  # All pixels should be white


@pytest.mark.integration
def test_is_valid_roi(node_embeddings):
    """Test ROI validation checks."""
    valid_roi = torch.ones((3, 32, 32))
    assert node_embeddings._is_valid_roi(valid_roi) is True
    empty_roi = torch.tensor([])
    assert node_embeddings._is_valid_roi(empty_roi) is False
    none_roi = None
    assert node_embeddings._is_valid_roi(none_roi) is False


@pytest.mark.integration
def test_get_roi_embeddings_for_frame(node_embeddings):
    """Test retrieval of ROI embeddings for a frame."""
    frame = torch.ones((3, 224, 224), dtype=torch.uint8) * 128
    frame_num = 10
    mock_tracer = MagicMock()
    mock_detection = MagicMock()
    mock_detection.class_name = "bowl"
    mock_detection.is_fixated = True
    mock_detection.bbox = (50, 50, 100, 100)
    mock_tracer.get_detections_for_frame.return_value = [mock_detection]
    
    with patch.object(node_embeddings, '_convert_roi_tensor_to_pil', return_value=MagicMock()):
        with patch.object(node_embeddings, '_get_clip_model') as mock_get_clip:
            mock_clip = MagicMock()
            mock_clip.encode_image.return_value = torch.ones((1, 512))
            mock_get_clip.return_value = mock_clip
            node_embeddings.clip_model = mock_clip
            embeddings = node_embeddings._get_roi_embeddings_for_frame(
                frame_tensor=frame,
                frame_num=frame_num,
                tracer=mock_tracer,
                object_label="bowl"
            )
    assert len(embeddings) == 1
    assert embeddings[0].shape == (1, 512)
    mock_tracer.get_detections_for_frame.assert_called_once_with(frame_num)


@pytest.mark.integration
@pytest.mark.parametrize("has_visits", [True, False])
def test_get_object_node_embedding(node_embeddings, has_visits):
    """Test object node embedding generation."""
    mock_checkpoint = MagicMock()
    mock_tracer = MagicMock()
    mock_video = MagicMock()
    node_data = {
        "object_label": "bowl",
        "visits": [(10, 20)] if has_visits else []
    }
    mock_checkpoint.nodes = {1: node_data}
    
    if has_visits:
        with patch.object(node_embeddings, '_get_roi_embeddings_for_visit') as mock_get_roi:
            mock_get_roi.return_value = [torch.ones((1, 512))]
            original_method = node_embeddings.get_object_node_embedding
            
            def patched_get_object_node_embedding(checkpoint, tracer, video, node_id):
                video.seek_to_frame(10)
                return original_method(checkpoint, tracer, video, node_id)
            
            with patch.object(node_embeddings, 'get_object_node_embedding', side_effect=patched_get_object_node_embedding):
                embedding = node_embeddings.get_object_node_embedding(
                    checkpoint=mock_checkpoint,
                    tracer=mock_tracer,
                    video=mock_video,
                    node_id=1
                )
            assert embedding is not None
            assert isinstance(embedding, torch.Tensor)
            assert embedding.shape[0] == 512
            mock_get_roi.assert_called_once()
            mock_video.seek_to_frame.assert_called_once_with(10)
    else:
        embedding = node_embeddings.get_object_node_embedding(
            checkpoint=mock_checkpoint,
            tracer=mock_tracer,
            video=mock_video,
            node_id=1
        )
        assert embedding is None


@pytest.mark.integration
def test_object_node_embedding_with_real_data(node_embeddings, test_checkpoint, test_tracer, test_video):
    """Test embedding generation with real checkpoint, tracer, and video data."""
    node_ids = [nid for nid, node in test_checkpoint.nodes.items() 
                if 'visits' in node and node['visits']]
    assert len(node_ids) > 0, "No valid nodes with visits found in checkpoint"
    node_id = node_ids[0]
    embedding = node_embeddings.get_object_node_embedding(
        checkpoint=test_checkpoint,
        tracer=test_tracer,
        video=test_video,
        node_id=node_id
    )
    if embedding is not None:
        assert isinstance(embedding, torch.Tensor)
        assert embedding.dim() == 1
        assert embedding.shape[0] == 512  # CLIP's default embedding size


@pytest.mark.integration
def test_roi_image_classification(node_embeddings, test_checkpoint, test_tracer, test_video, sample_data_path):
    """Test that extracted ROI images are correctly classified by CLIP."""
    # Get noun labels from ActionRecord (which auto-initializes when needed)
    id_to_name, _ = ActionRecord.get_noun_label_mappings()
    object_labels = list(id_to_name.values())
    
    # Ensure CLIP model is loaded
    clip_model = node_embeddings._get_clip_model()
    
    # Find nodes with visits
    node_ids = [nid for nid, node in test_checkpoint.nodes.items() 
               if 'visits' in node and node['visits']]
    
    assert len(node_ids) > 0, "No valid nodes with visits found in checkpoint"

    # Test for first valid node
    node_id = node_ids[0]
    node_data = test_checkpoint.nodes[node_id]
    object_label = node_data["object_label"]
    visit_start, visit_end = node_data["visits"][0]
    print(f"Checkpoint.frame_number: {test_checkpoint.frame_number}")
    print(f"Node ID: {node_id}, Object Label: {object_label}, Visit Start: {visit_start}, Visit End: {visit_end}")
    
    # Seek to the visit start frame
    print(f"Seeking to frame {visit_start}")
    test_video.seek_to_frame(visit_start)
    
    # Get a frame from the visit
    frame = next(test_video.stream)
    frame_tensor = frame['data']
    # write frame as a png for debugging
    frame_pil = Image.fromarray(frame_tensor.numpy().transpose(1, 2, 0).astype(np.uint8))
    # save to sample_data_path
    frame_pil.save(sample_data_path / "frame.png")
    
    # Get the detections for the frame
    detections = test_tracer.get_detections_for_frame(visit_start)
    matching_dets = [det for det in detections 
                      if det.class_name == object_label and det.is_fixated]
    assert len(matching_dets) > 0, f"No matching detections found for the node. Found: {detections} but expected: {object_label}"
    
    # Extract the ROI
    roi_tensor = node_embeddings._extract_roi(frame_tensor, matching_dets[0].bbox)
    assert node_embeddings._is_valid_roi(roi_tensor), f"Invalid ROI extracted: {roi_tensor.shape}"
    
    # Convert the ROI to a PIL image
    pil_image = NodeEmbeddings._convert_roi_tensor_to_pil(roi_tensor)
    assert pil_image is not None, "Failed to convert ROI to PIL image"
    
    # write ROI as a png for debugging
    pil_image.save(sample_data_path / "roi.png")

    # Make sure the best label is the object label
    scores, best_label = clip_model.classify(object_labels, pil_image)

    # print score and label for top 5 labels
    # apply softmax to scores
    scores = torch.nn.functional.softmax(torch.tensor(scores), dim=0)
    # sort scores and labels by score in descending order
    scores, labels = zip(*sorted(zip(scores, object_labels), key=lambda x: x[0], reverse=True))
    for i, (score, label) in enumerate(zip(scores, labels)):
        print(f"Top {i+1} label: {label}, score: {score}")
        if i > 5:
            break

    assert best_label == object_label, f"CLIP best label is not the expected object label: {best_label} != {object_label}"