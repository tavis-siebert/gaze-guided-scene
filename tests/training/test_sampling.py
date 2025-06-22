import pytest

from gazegraph.training.dataset import sampling


class DummyCheckpoint:
    def __init__(self, frame_number, label=None):
        self.frame_number = frame_number
        self.label = label

    def get_future_action_labels(self, frame, metadata):
        # For test purposes, return label if frame is valid
        if hasattr(metadata, "valid_frames") and frame not in metadata.valid_frames:
            return None
        return {"label": self.label if self.label is not None else frame}


class DummyMetadata:
    def __init__(self, valid_frames=None, start=0, end=9):
        self.valid_frames = (
            valid_frames if valid_frames is not None else list(range(start, end + 1))
        )
        self.start = start
        self.end = end

    def get_action_frame_range(self, video_name):
        return self.start, self.end


class DummyCheckpoint:
    def __init__(self, frame_number, label=None):
        self.frame_number = frame_number
        self.label = label

    def get_future_action_labels(self, frame, metadata):
        if hasattr(metadata, "valid_frames") and frame not in metadata.valid_frames:
            return None
        return {"label": self.label if self.label is not None else frame}


class DummyMetadata:
    def __init__(self, valid_frames=None, start=0, end=9):
        self.valid_frames = (
            valid_frames if valid_frames is not None else list(range(start, end + 1))
        )
        self.start = start
        self.end = end

    def get_action_frame_range(self, video_name):
        return self.start, self.end


@pytest.mark.parametrize("strategy", ["all", "uniform", "random"])
@pytest.mark.parametrize("oversampling", [False, True])
@pytest.mark.parametrize("allow_duplicates", [False, True])
@pytest.mark.parametrize("samples_per_video", [0, 3, 10, 15])
def test_sampling_matrix(strategy, oversampling, allow_duplicates, samples_per_video):
    # Use 5 checkpoints, 10 frames for oversampling
    checkpoints = [DummyCheckpoint(i) for i in range(0, 10, 2)]
    metadata = DummyMetadata(start=0, end=9)
    samples = sampling.get_samples(
        checkpoints,
        "video",
        strategy,
        samples_per_video,
        allow_duplicates,
        oversampling,
        metadata,
    )
    # Check output types
    assert all(isinstance(x[0], DummyCheckpoint) for x in samples)
    assert all(isinstance(x[1], dict) for x in samples)
    # Determine potential size
    if oversampling:
        potential = []
        for frame in range(0, 10):
            suitable = [cp for cp in checkpoints if cp.frame_number <= frame]
            if not suitable:
                continue
            cp = suitable[-1]
            labels = cp.get_future_action_labels(frame, metadata)
            if labels is not None:
                potential.append((cp, labels))
    else:
        potential = [
            (cp, cp.get_future_action_labels(cp.frame_number, metadata))
            for cp in checkpoints
        ]
    potential = [x for x in potential if x[1] is not None]
    # Check sample counts and duplication
    if strategy == "all" or samples_per_video == 0:
        assert len(samples) == len(potential)
    elif samples_per_video >= len(potential):
        if allow_duplicates:
            assert len(samples) == samples_per_video
        else:
            assert len(samples) == len(potential)
    else:
        assert len(samples) == samples_per_video
        if not allow_duplicates:
            # No duplicates
            assert (
                len({(x[0].frame_number, x[1]["label"]) for x in samples})
                == samples_per_video
            )


@pytest.mark.parametrize("oversampling", [False, True])
def test_empty_checkpoints(oversampling):
    samples = sampling.get_samples(
        [], "video", "all", 5, False, oversampling, DummyMetadata()
    )
    assert samples == []


def test_invalid_strategy():
    checkpoints = [DummyCheckpoint(i) for i in range(5)]
    metadata = DummyMetadata()
    with pytest.raises(ValueError):
        sampling.get_samples(
            checkpoints, "video", "not_a_strategy", 5, False, False, metadata
        )
    with pytest.raises(ValueError):
        sampling.get_samples(
            checkpoints, "video", "not_a_strategy", 5, False, True, metadata
        )


def test_oversampling_fallback_on_metadata():
    checkpoints = [DummyCheckpoint(i) for i in range(5)]

    class BadMetadata(DummyMetadata):
        def get_action_frame_range(self, video_name):
            raise ValueError()

    metadata = BadMetadata()
    samples = sampling.get_samples(
        checkpoints, "video", "all", 0, False, True, metadata
    )
    assert len(samples) == 5
