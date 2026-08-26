"""
Frame index arithmetic, kept separate from decoding so it can be tested.

The bug this guards against is the one that doesn't raise: `np.linspace(..., dtype=int)`
floors every index, pulling the whole sample toward the start of the clip.
"""

import numpy as np
import pytest

from data.utils import frame_indices


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
