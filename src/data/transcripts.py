
# load & process transcripts 
"""Deduplicate YouTube-style WEBVTT transcripts.

Algorithm: walk cues in order, normalize each line, dedup.

NOTE: genuine immediate repetition in speech ("hi hi" on two lines) collapses
to one. That is the accepted cost of exact-match dedup.
"""

import html
import logging
from collections.abc import Sequence
from pathlib import Path

from config import Columns
import webvtt
from webvtt import WebVTT

logger = logging.getLogger(__name__)


def _normalize(line: str) -> str:
    """Unescape entities and collapse all whitespace (incl. &nbsp;) to single spaces."""
    return " ".join(html.unescape(line).split())


def dedup(path: Path) -> list[tuple[float, str]]:
    """Return [(start_sec, line)] with cue tags and rolling-window repeats removed."""
    out: list[tuple[float, str]] = []
    last = None
    for cue in webvtt.read(str(path)):
        for line in cue.text.splitlines():  # .text -> karaoke tags already stripped
            line = _normalize(line)
            if line and line != last:
                out.append((cue.start_in_seconds, line))
                last = line
    return out


def to_text(path: Path) -> str:
    """Model-ready plain text: deduped lines joined into a single string."""
    return " ".join(line for _, line in dedup(path))


def load_transcripts(records: list[dict]) -> dict[str, str]:
	"""
    Load transcripts for existing records.
	
	Args:
		records: a list of dictionary records, holding at least the following keys:
			- vid_name: video id
			- path_to_transcript: path to local .vtt file
    
    Returns: A mapping of video id to transcript text
	"""
	transcripts = {}
	missing = []
	for r in records:
		# `vid` is bound before the try: a KeyError on the id itself would otherwise
		# raise NameError inside the handler.
		vid = r.get(Columns.VIDEO_ID, "<no vid_name>")
		try:
			path = Path(r[Columns.TRANSCRIPT_PATH])
			if not path.is_file():
				missing.append(vid)  # absent file is a miss, not a silent skip
				continue
			transcripts[vid] = to_text(path)
		except Exception:
			logger.exception("could not read transcript for %s", vid)
			missing.append(vid)
	if missing:
		logger.warning("%d transcripts missing or unreadable, e.g. %s", len(missing), missing[:5])
	return transcripts


def _overlapping(r1: tuple|list, r2: tuple|list):
	"""
	Return whether two ranges overlap.

	Examples:
		>> r1 = [0     5]
		>> r2 =     [4      8]

		>> r1 = 		[6  8]
		>> r2 =     [4     7]
		
		>> r1 = 		[6]
		>> r2 =     [4     7]

		>> r1 = 	[4			10]
		>> r2 =     [4     7]
	"""
	assert len(r1) == len(r2) == 2, "Invalid input(s). Both should have length exactly 2."
	return r1[0] <= r2[1] and r2[0] <= r1[1]

def split_transcript_by_ranges(
	path: Path,
	ranges: Sequence[Sequence[float]],
) -> list[str]:
	"""
	Bucket a transcript's cues into time ranges -- the transcript-side companion to
	`compute_segments_around_oracle`. Two sorted pointers walk ranges and cues together
	in O(n + m). Two behaviors worth knowing:

	- A cue goes to the *first* range it overlaps, so one straddling a boundary counts
	  toward the earlier chunk only.
	- Dedup compares against the current range's stack, not a cursor spanning all of
	  them, so the rolling window's restatement of a chunk's last line lands in both
	  chunks (~5% word overlap). A global cursor would instead empty a chunk whose only
	  line repeats the previous chunk's last -- "[Music]" over a long instrumental.

	Args:
		path: path to a .vtt file
		ranges: [start, end] second pairs, typically the chunks of a single video

	Returns:
		One string per range, ascending by start, so `texts[oracle_idx]` is the oracle's
		speech. Silent ranges yield "" rather than dropping out.
	"""
	ranges = sorted((tuple(r) for r in ranges), key=lambda r: r[0])  # sort by start just in case
	stacks: dict[tuple[float, float], list[str]] = {r: [] for r in ranges}
	vtt: WebVTT = webvtt.read(str(path))
	cues = [(cue.start_in_seconds, cue.end_in_seconds, cue.text) for cue in vtt]

	p_range = p_cue = 0
	while p_range < len(ranges) and p_cue < len(cues):
		current = ranges[p_range]
		cue_start, cue_end, text = cues[p_cue]
		if _overlapping(current, (cue_start, cue_end)):
			stack = stacks[current]
			for line in text.splitlines():  # .text -> karaoke tags already stripped
				line = _normalize(line)
				if line and (not stack or stack[-1] != line):
					stack.append(line)
			p_cue += 1
		elif cue_end < current[0]:
			p_cue += 1
		else:
			p_range += 1

	# concatenate all lines per stack
	return [" ".join(stacks[r]) for r in ranges]
