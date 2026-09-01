import pytest

from data.oracles import compute_segments_around_oracle

@pytest.mark.parametrize(
    "oracle,duration,buffer,expected_slices,expected_oracle_idx",
    [
        ((127.11, 187.11), 200, 10, [[0,67.11], [67.11, 127.11], [127.11, 187.11], [187.11, 200]], 2),
        ((0.0, 60.0), 136.649433, 10, [[0.0, 60.0], [60.0, 120.0], [120.0, 136.649433]], 0),
        ((5.0, 65.0), 90, 10, [[5.0, 65.0], [65.0, 90.0]], 0),  # NOTE: here the first chunk is the oracle, so buffer should not apply!
        ((15.0, 75.0), 84, 10, [[0.0, 15.0], [15.0, 75.0],], 1),  # NOTE: here the last chunk is the oracle, so buffer should not apply!
        # ...but when a real chunk follows the oracle, the tail sliver *is* absorbed into it.
        ((0.0, 60.0), 245, 10, [[0.0, 60.0], [60.0, 120.0], [120.0, 180.0], [180.0, 245]], 0),
        # Duration an exact multiple of chunk_size: remainder is 0, so neither end may emit a
        # zero-length chunk (the tail branch still runs, it just has nothing to absorb).
        ((60.0, 120.0), 240, 10, [[0, 60.0], [60.0, 120.0], [120.0, 180.0], [180.0, 240]], 1),
        # Oracle flush against the end of the video: no trailing chunk at all.
        ((60.0, 120.0), 120, 10, [[0, 60.0], [60.0, 120.0]], 1),
        # Video is barely longer than the oracle -- everything else is sliver, so the oracle
        # is the only chunk and the video contributes no distractors.
        ((0.0, 60.0), 65, 10, [[0.0, 60.0]], 0),
        # Head shorter than chunk_size but longer than the buffer: kept as a short chunk.
        ((100.0, 160.0), 280, 10, [[0, 40.0], [40.0, 100.0], [100.0, 160.0], [160.0, 220.0], [220.0, 280]], 2),
    ]
)
def test_compute_segments_around_oracle(
    oracle: tuple[float, float],
    duration: float,
    buffer: int,
    expected_slices: list[tuple[float, float]],
    expected_oracle_idx: int,
):
    slices, oracle_idx = compute_segments_around_oracle(oracle, duration, buffer_size=buffer)
    assert slices == expected_slices
    assert oracle_idx == expected_oracle_idx


# Every (oracle_start, duration) pair the real data can produce: oracles are always 60s and
# always fall inside the video, so duration >= oracle_start + 60 is the only constraint.
GRID = [
    (start, duration)
    for start in (0.0, 3.0, 15.0, 59.98, 60.0, 127.11, 300.0)
    for duration in (60.0, 65.0, 90.0, 120.0, 187.5, 245.0, 600.0, 1000.0)
    if duration >= start + 60
]


@pytest.mark.parametrize("buffer", [0, 10, 30])
@pytest.mark.parametrize("start,duration", GRID)
def test_segments_hold_structural_invariants(start: float, duration: float, buffer: int):
    """
    The properties the point cases above can't cover exhaustively. Contiguity and an
    untouched oracle are what the downstream dataset actually depends on: chunk boundaries
    become ffmpeg cut points, and `oracle_idx` is the retrieval label, so a resized or
    misindexed oracle mislabels every question on that video without ever raising.
    """
    oracle = (start, start + 60)
    chunks, oracle_idx = compute_segments_around_oracle(oracle, duration, buffer_size=buffer)

    assert chunks, "a valid oracle must always yield at least the oracle chunk"
    assert chunks[oracle_idx] == list(oracle), "oracle bounds must never move"
    assert chunks.count(list(oracle)) == 1, "oracle must appear exactly once"

    for lo, hi in chunks:
        assert lo < hi, f"non-positive-length chunk [{lo}, {hi}]"
    for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:]):
        assert prev_end == next_start, "chunks must tile without gaps or overlap"

    assert 0 <= chunks[0][0] and chunks[-1][1] <= duration, "chunks must stay inside the video"
    # Time is only ever discarded at the two edges, only as a sliver, and only when the
    # chunk that would have absorbed it is the oracle.
    assert chunks[0][0] <= buffer
    assert duration - chunks[-1][1] <= buffer
    # No non-oracle chunk is short enough to be a sliver itself.
    for i, (lo, hi) in enumerate(chunks):
        if i != oracle_idx:
            assert hi - lo > buffer, f"chunk [{lo}, {hi}] is shorter than the buffer"

