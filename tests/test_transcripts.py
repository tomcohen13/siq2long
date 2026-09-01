"""
Cue-to-chunk bucketing, tested against synthetic VTTs written to tmp_path rather than
files on the T7, so the cases stay legible and the awkward transcripts (no captions at
all, captions that stop before the video does) can be written down instead of hunted for.
"""

import itertools

import pytest

from data.transcripts import split_transcript_by_ranges, to_text

THIRDS = [[0, 60], [60, 120], [120, 180]]


def _ts(seconds: float) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


@pytest.fixture
def write_vtt(tmp_path):
    """Write a minimal WEBVTT file from (start_sec, end_sec, payload) triples."""
    counter = itertools.count()

    def _write(*cues: tuple[float, float, str]):
        body = "\n\n".join(f"{_ts(start)} --> {_ts(end)}\n{text}" for start, end, text in cues)
        path = tmp_path / f"cues{next(counter)}.vtt"
        path.write_text(f"WEBVTT\n\n{body}\n")
        return path

    return _write


def test_buckets_each_cue_into_its_range(write_vtt):
    doc = write_vtt((5, 7, "alpha"), (65, 67, "beta"), (125, 127, "gamma"))
    assert split_transcript_by_ranges(doc, THIRDS) == ["alpha", "beta", "gamma"]


def test_silent_range_yields_empty_string_not_a_missing_entry(write_vtt):
    doc = write_vtt((5, 7, "alpha"), (125, 127, "gamma"))
    assert split_transcript_by_ranges(doc, THIRDS) == ["alpha", "", "gamma"]


def test_transcript_with_no_cues_yields_one_empty_string_per_range(write_vtt):
    """Caption-less VTTs exist in the dataset; indexing cues[0] here used to raise."""
    assert split_transcript_by_ranges(write_vtt(), THIRDS) == ["", "", ""]


def test_cues_before_the_first_range_are_skipped(write_vtt):
    """A chunk list can start past 0 when the head sliver is dropped as the oracle's."""
    doc = write_vtt((5, 7, "early"), (105, 107, "inside"))
    assert split_transcript_by_ranges(doc, [[100, 160], [160, 220]]) == ["inside", ""]


def test_cues_after_the_last_range_are_ignored(write_vtt):
    """Transcripts routinely run past the last chunk when a tail sliver was dropped."""
    doc = write_vtt((5, 7, "inside"), (500, 502, "late"))
    assert split_transcript_by_ranges(doc, [[0, 60]]) == ["inside"]


def test_output_is_ascending_by_start_even_when_ranges_are_not(write_vtt):
    doc = write_vtt((5, 7, "alpha"), (65, 67, "beta"))
    assert split_transcript_by_ranges(doc, [[60, 120], [0, 60]]) == ["alpha", "beta"]


def test_rolling_window_repeats_collapse_within_a_range(write_vtt):
    doc = write_vtt(
        (0, 2, "line one"),
        (2, 4, "line one\nline two"),
        (4, 6, "line two\nline three"),
    )
    assert split_transcript_by_ranges(doc, [[0, 60]]) == ["line one line two line three"]


def test_karaoke_tags_are_stripped_from_the_payload(write_vtt):
    doc = write_vtt((10, 14, "faculty<00:00:13.559><c> reception</c>"))
    assert split_transcript_by_ranges(doc, [[0, 60]]) == ["faculty reception"]


def test_entities_and_nbsp_are_normalized(write_vtt):
    doc = write_vtt((5, 7, "it&#39;s&nbsp;&nbsp;fine"))
    assert split_transcript_by_ranges(doc, [[0, 60]]) == ["it's fine"]


def test_straddling_cue_counts_toward_the_earlier_range_only(write_vtt):
    doc = write_vtt((58, 62, "straddle"))
    assert split_transcript_by_ranges(doc, [[0, 60], [60, 120]]) == ["straddle", ""]


def test_one_long_cue_fills_only_the_range_it_first_overlaps(write_vtt):
    """The documented cost of single assignment -- hand-authored VTTs with long cues."""
    doc = write_vtt((0, 300, "one very long cue"))
    assert split_transcript_by_ranges(doc, THIRDS) == ["one very long cue", "", ""]


def test_boundary_line_is_counted_in_both_chunks(write_vtt):
    """
    Per-stack dedup means the rolling window's restatement of a chunk's closing line
    survives into the next chunk. Pinned because it inflates chunk word counts a few
    percent against the whole transcript, and because the alternative -- one cursor for
    all ranges -- would empty a chunk whose only line repeats the previous chunk's last.
    """
    doc = write_vtt((58, 59, "tail line"), (61, 63, "tail line\nnext line"))
    assert split_transcript_by_ranges(doc, [[0, 60], [60, 120]]) == [
        "tail line",
        "tail line next line",
    ]


@pytest.mark.parametrize(
    "ranges",
    [
        [[0, 240]],
        [[0, 120], [120, 240]],
        [[0, 60], [60, 120], [120, 180], [180, 240]],
    ],
)
def test_chunking_never_drops_a_word(write_vtt, ranges: list[list[float]]):
    """Boundaries may duplicate a line, but no chunking of a covered span may lose one."""
    doc = write_vtt(
        (5, 7, "first thing said"),
        (65, 67, "second thing said"),
        (125, 127, "third thing said"),
        (185, 187, "fourth thing said"),
    )
    chunked = " ".join(split_transcript_by_ranges(doc, ranges)).split()
    for word in to_text(doc).split():
        assert word in chunked


def test_matches_to_text_when_one_range_covers_everything(write_vtt):
    doc = write_vtt((0, 2, "line one"), (2, 4, "line one\nline two"), (4, 6, "line three"))
    assert split_transcript_by_ranges(doc, [[0, 600]]) == [to_text(doc)]
