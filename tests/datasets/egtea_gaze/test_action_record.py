"""
Consolidated unit tests for ActionRecord.
"""

import os
import tempfile
import pytest
from unittest.mock import patch

from gazegraph.datasets.egtea_gaze.action_record import ActionRecord


@pytest.fixture
def mock_action_records(monkeypatch):
    """Fixture providing mock action records for testing."""
    # Disable initialization to avoid loading files
    monkeypatch.setattr(
        ActionRecord, "_ensure_initialized", lambda *args, **kwargs: None
    )
    return [
        ActionRecord(["video_1", "10", "20", "1", "3"]),
        ActionRecord(["video_1", "30", "40", "2", "1"]),
        ActionRecord(["video_2", "5", "15", "3", "4"]),
    ]


@pytest.fixture
def mock_records_by_video(mock_action_records):
    """Fixture providing mock records organized by video."""
    records_by_video = {}
    for record in mock_action_records:
        if record.video_name not in records_by_video:
            records_by_video[record.video_name] = []
        records_by_video[record.video_name].append(record)
    return records_by_video


# Basic initialization and properties
def test_init_and_properties(monkeypatch):
    monkeypatch.setattr(
        ActionRecord, "_ensure_initialized", lambda *args, **kwargs: None
    )
    record = ActionRecord(["video_1", "10", "30", "1", "3"])
    assert record.video_name == "video_1"
    assert record.start_frame == 10
    assert record.end_frame == 30
    assert record.label == [1, 3]
    assert record.verb_id == 1
    assert record.noun_id == 3
    assert record.num_frames == 21
    assert record.action_tuple == (1, 3)
    ActionRecord._verb_id_to_name = {1: "take"}
    ActionRecord._noun_id_to_name = {3: "knife"}
    assert record.verb_name == "take"
    assert record.noun_name == "knife"
    assert record.action_name == "take knife"
    ActionRecord._verb_id_to_name = {}
    ActionRecord._noun_id_to_name = {}
    assert ActionRecord(["v", "0", "1", "1", "2"]).action_name is None


# action_idx property
@pytest.mark.parametrize(
    "verb,noun,expected",
    [
        (1, 3, 0),
        (4, 4, 3),
        (99, 99, None),
    ],
)
def test_action_idx(monkeypatch, verb, noun, expected):
    monkeypatch.setattr(
        ActionRecord, "_ensure_initialized", lambda *args, **kwargs: None
    )
    ActionRecord._action_to_idx = {
        (1, 3): 0,
        (2, 1): 1,
        (3, 4): 2,
        (4, 4): 3,
        (5, 2): 4,
    }
    rec = ActionRecord(["v", "0", "1", str(verb), str(noun)])
    assert rec.action_idx == expected


# String representation
def test_str_representation(monkeypatch):
    monkeypatch.setattr(
        ActionRecord, "_ensure_initialized", lambda *args, **kwargs: None
    )
    ActionRecord._verb_id_to_name = {1: "take"}
    ActionRecord._noun_id_to_name = {3: "knife"}
    rec = ActionRecord(["v", "5", "15", "1", "3"])
    s = str(rec)
    assert "take knife" in s and "start=5" in s and "end=15" in s
    ActionRecord._verb_id_to_name = {}
    ActionRecord._noun_id_to_name = {}
    assert "(1, 3)" in str(ActionRecord(["v", "5", "15", "1", "3"]))


# Name mappings
def test_load_name_mappings_success(setup_mock_files, mock_config):
    ActionRecord._verb_id_to_name = {}
    ActionRecord._noun_id_to_name = {}
    with patch(
        "gazegraph.datasets.egtea_gaze.action_record.get_config",
        return_value=mock_config,
    ):
        ActionRecord._load_name_mappings()
    assert ActionRecord._verb_id_to_name == {
        1: "take",
        2: "put",
        3: "open",
        4: "close",
        5: "wash",
    }
    assert ActionRecord._noun_id_to_name == {
        1: "cup",
        2: "bowl",
        3: "knife",
        4: "microwave",
        5: "fridge",
    }


def test_load_name_mappings_missing(setup_mock_files, mock_config):
    with (
        patch(
            "gazegraph.datasets.egtea_gaze.action_record.get_config",
            return_value=mock_config,
        ),
        patch("os.path.exists", return_value=False),
    ):
        with pytest.raises(FileNotFoundError):
            ActionRecord._load_name_mappings()


# Compute mapping
def test_compute_action_mapping(monkeypatch):
    monkeypatch.setattr(
        ActionRecord, "_ensure_initialized", lambda *args, **kwargs: None
    )
    records = [ActionRecord([f"v{i}", "0", "1", "1", "3"]) for i in range(2)]
    records.append(ActionRecord(["v", "0", "1", "2", "1"]))
    with patch("collections.Counter") as mock_ctr:
        inst = mock_ctr.return_value
        inst.items.return_value = [((1, 3), 2), ((2, 1), 1)]
        mapping = ActionRecord._compute_action_mapping(records, num_classes=2)
    assert mapping == {(1, 3): 0, (2, 1): 1}


# Load all records
def test_load_all_records_success(setup_mock_files, mock_config):
    with patch(
        "gazegraph.datasets.egtea_gaze.action_record.get_config",
        return_value=mock_config,
    ):
        videos, train = ActionRecord._load_all_records()
    assert "video_1" in videos and "video_4" in videos
    assert len(train) == 5


def test_load_all_records_file_not_found(mock_config):
    with (
        patch(
            "gazegraph.datasets.egtea_gaze.action_record.get_config",
            return_value=mock_config,
        ),
        patch("os.path.exists", return_value=True),
        patch("builtins.open", side_effect=FileNotFoundError),
    ):
        with pytest.raises(FileNotFoundError):
            ActionRecord._load_all_records()


# load_records_from_file
def test_load_records_from_file(tmp_path, monkeypatch):
    p = tmp_path / "f.txt"
    p.write_text("video 10 20 1 2\n")
    monkeypatch.setattr(
        ActionRecord, "_ensure_initialized", lambda *args, **kwargs: None
    )
    recs = ActionRecord.load_records_from_file(str(p))
    assert len(recs) == 1 and recs[0].start_frame == 10


def test_load_records_from_file_errors(monkeypatch):
    monkeypatch.setattr(
        ActionRecord, "_ensure_initialized", lambda *args, **kwargs: None
    )
    with pytest.raises(FileNotFoundError):
        ActionRecord.load_records_from_file("no.txt")
    f = tempfile.NamedTemporaryFile(mode="w", delete=False)
    f.write("bad\n")
    f.close()
    with pytest.raises(ValueError):
        ActionRecord.load_records_from_file(f.name)
    os.unlink(f.name)


# API methods
def test_api_methods(mock_action_records, mock_records_by_video):
    ActionRecord._ensure_initialized = classmethod(lambda *args, **kwargs: None)

    # Patch the internal state to use our mock data
    with patch.object(ActionRecord, "_records_by_video", mock_records_by_video):
        with patch.object(ActionRecord, "_action_to_idx", {(1, 3): 0, (2, 1): 1}):
            # Test get_records_for_video
            assert (
                ActionRecord.get_records_for_video("video_1")
                == mock_records_by_video["video_1"]
            )
            assert ActionRecord.get_records_for_video("none") == []

            # Test get_all_videos
            assert set(ActionRecord.get_all_videos()) == set(mock_records_by_video)

            # Test get_all_records
            assert len(ActionRecord.get_all_records()) == sum(
                len(v) for v in mock_records_by_video.values()
            )

            # Test get_action_mapping
            mp = ActionRecord.get_action_mapping()
            assert (
                mp == {(1, 3): 0, (2, 1): 1} and mp is not ActionRecord._action_to_idx
            )

            # Test get_noun_label_mapping
            with patch.object(
                ActionRecord, "_noun_id_to_name", {1: "cup", 2: "bowl", 3: "knife"}
            ):
                n2i = ActionRecord.get_noun_label_mapping()
                assert n2i == {"cup": 1, "bowl": 2, "knife": 3}

                # Test get_action_name_by_idx
                with patch.object(
                    ActionRecord, "_verb_id_to_name", {1: "take", 2: "put"}
                ):
                    assert ActionRecord.get_action_name_by_idx(0) == "take knife"
                    assert ActionRecord.get_action_name_by_idx(1) == "put cup"

                    # Test get_action_names
                    names = ActionRecord.get_action_names()
                    assert isinstance(names, dict) and names


# Future action labels
def test_create_future_action_labels(mock_records_by_video):
    ActionRecord._ensure_initialized = classmethod(lambda *args, **kwargs: None)
    ActionRecord._action_to_idx = {(1, 3): 0, (2, 1): 1}
    ActionRecord.get_records_for_video = lambda v: mock_records_by_video.get(v, [])
    assert ActionRecord.create_future_action_labels("none", 0) is None
    res = ActionRecord.create_future_action_labels("video_1", 5, num_action_classes=2)
    assert res and res["next_action"].item() == 0


# Config tests
def test_get_config_and_caching(mock_config):
    with patch(
        "gazegraph.datasets.egtea_gaze.action_record.get_config",
        return_value=mock_config,
    ):
        ActionRecord._config = None
        assert ActionRecord.get_config() is mock_config
        with patch("gazegraph.datasets.egtea_gaze.action_record.get_config") as m:
            assert ActionRecord.get_config() is mock_config
            m.assert_not_called()


def test_missing_config_raises():
    with patch(
        "gazegraph.datasets.egtea_gaze.action_record.get_config",
        side_effect=ImportError,
    ):
        ActionRecord._config = None
        with pytest.raises(ImportError):
            ActionRecord.get_config()


# Initialization idempotence
def test_ensure_initialized_idempotent(mock_config):
    calls = []
    with (
        patch(
            "gazegraph.datasets.egtea_gaze.action_record.get_config",
            return_value=mock_config,
        ),
        patch.object(
            ActionRecord, "_load_name_mappings", side_effect=lambda: calls.append("nm")
        ),
        patch.object(ActionRecord, "_load_all_records", side_effect=lambda: ({}, [])),
        patch.object(
            ActionRecord,
            "_compute_and_set_action_mapping",
            side_effect=lambda *args, **kwargs: calls.append("cm"),
        ),
    ):
        ActionRecord._is_initialized = False
        ActionRecord._ensure_initialized()
        assert "nm" in calls and "cm" in calls
        calls.clear()
        ActionRecord._ensure_initialized()
        assert not calls
