"""
Frame index arithmetic, kept separate from decoding so it can be tested.

The bug this guards against is the one that doesn't raise: `np.linspace(..., dtype=int)`
floors every index, pulling the whole sample toward the start of the clip.
"""

import numpy as np
import pytest

from transformers.video_utils import VideoMetadata

from data.videos import frame_indices, window_frame_indices


def test_returns_requested_count():
    assert len(frame_indices(1438, 32)) == 32


def test_spans_first_and_last_frame():
    idx = frame_indices(1438, 32)
    assert idx[0] == 0
    assert idx[-1] == 1437


def test_indices_are_sorted_and_unique():
    idx = frame_indices(1438, 58)
    assert list(idx) == sorted(idx)
    assert len(set(idx)) == len(idx)


def test_rounds_instead_of_truncating():
    """The regression test: truncation biases every interior index downward."""
    total, n = 100, 7
    rounded = frame_indices(total, n)
    truncated = np.linspace(0, total - 1, n, dtype=int)
    assert list(rounded) == [0, 16, 33, 50, 66, 82, 99]
    assert list(truncated) == [0, 16, 33, 49, 66, 82, 99]  # index 3 lands a frame early
    assert rounded.sum() > truncated.sum()


def test_more_frames_requested_than_exist_returns_all():
    """Clamp rather than pad: padding hides a decode failure behind valid-looking input."""
    idx = frame_indices(8, 32)
    assert list(idx) == list(range(8))


def test_single_frame_requested():
    assert list(frame_indices(1438, 1)) == [0]


def test_single_frame_video():
    assert list(frame_indices(1, 32)) == [0]


def test_dtype_is_integer():
    assert np.issubdtype(frame_indices(1438, 32).dtype, np.integer)


@pytest.mark.parametrize("total, num", [(0, 32), (-1, 32), (1438, 0), (1438, -5)])
def test_rejects_nonpositive_inputs(total, num):
    with pytest.raises(ValueError):
        frame_indices(total, num)


# --- window_frame_indices: seconds -> frame range -------------------------------------
# Uses the real VideoMetadata rather than a stand-in, so a field rename upstream fails
# here instead of at decode time.


def meta(total_num_frames=9000, fps=30.0):
    return VideoMetadata(total_num_frames=total_num_frames, fps=fps)


def test_window_of_none_is_the_whole_video():
    assert list(window_frame_indices(meta(), 32)) == list(frame_indices(9000, 32))


def test_window_maps_seconds_onto_the_right_frames():
    """1s-2s at 30fps is frames 30..59, inclusive at both ends."""
    idx = window_frame_indices(meta(), 8, start=1.0, end=2.0)
    assert idx[0] == 30 and idx[-1] == 59
    assert len(idx) == 8


def test_window_end_past_the_last_frame_clamps():
    """A final chunk runs to ffprobe's duration, which can round past the frame count."""
    idx = window_frame_indices(meta(total_num_frames=300), 16, start=0.0, end=100.0)
    assert idx[-1] == 299


def test_window_starting_at_zero():
    assert window_frame_indices(meta(), 4, start=0.0, end=1.0)[0] == 0


def test_window_shorter_than_request_returns_what_exists():
    idx = window_frame_indices(meta(), 32, start=0.0, end=0.1)  # 3 frames at 30fps
    assert len(idx) == 3


def test_window_rejects_start_without_end():
    for kwargs in ({"start": 1.0}, {"end": 2.0}):
        with pytest.raises(ValueError):
            window_frame_indices(meta(), 8, **kwargs)


def test_window_rejects_a_range_past_the_video():
    with pytest.raises(ValueError):
        window_frame_indices(meta(total_num_frames=300), 8, start=100.0, end=110.0)


@pytest.mark.parametrize("total", [0, None])
def test_window_rejects_a_bogus_frame_count(total):
    """pyav reads this from container metadata, which has been wrong on this dataset."""
    with pytest.raises(RuntimeError):
        window_frame_indices(meta(total_num_frames=total), 8)


def test_window_rejects_missing_fps_only_when_it_needs_it():
    assert len(window_frame_indices(meta(fps=None), 8)) == 8  # whole video needs no fps
    with pytest.raises(RuntimeError):
        window_frame_indices(meta(fps=None), 8, start=1.0, end=2.0)
