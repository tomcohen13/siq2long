
# load & process transcripts 
"""Deduplicate YouTube-style rolling WEBVTT transcripts.

Auto-generated VTTs use a rolling two-line window: each ~2s cue shows
[previous line, current line], and each 10ms "transition" cue repeats the
current line followed by a blank. So every spoken line appears 3-4 times,
but crucially all repeats are *consecutive* in cue order.

Newer yt-dlp downloads also carry word-level karaoke tags inside the cue
payload (`faculty<00:00:13.559><c> recep</c>...`). These must be stripped
*before* the duplicate comparison: a tagged line and its untagged repeat are
not string-equal, so leaving the tags in defeats the dedup and every line
survives twice. `webvtt.Caption.text` returns the payload with tags already
removed (`.raw_text` and `.lines` keep them) -- hence the parser.

Algorithm: walk cues in order, normalize each payload line, emit it only if
it differs from the last emitted line. O(n), no fuzzy matching. Each emitted
line keeps the start time of the cue it first appeared in, which survives
into (start, text) pairs for chunk alignment.

Note: genuine immediate repetition in speech ("hi hi" on two lines) collapses
to one. That is the accepted cost of exact-match dedup.
"""

import html
import io
from pathlib import Path

from config import Columns
import webvtt


def _captions(vtt: str | Path):
    """Cues from a .vtt file (pass a Path) or from raw VTT text (pass a str)."""
    if isinstance(vtt, Path):
        return webvtt.read(str(vtt))
    return webvtt.read_buffer(io.StringIO(vtt))


def _normalize(line: str) -> str:
    """Unescape entities and collapse all whitespace (incl. &nbsp;) to single spaces."""
    return " ".join(html.unescape(line).split())


def dedup(vtt: str | Path) -> list[tuple[float, str]]:
    """Return [(start_sec, line)] with cue tags and rolling-window repeats removed."""
    out: list[tuple[float, str]] = []
    last = None
    for cue in _captions(vtt):
        for line in cue.text.splitlines():  # .text -> karaoke tags already stripped
            line = _normalize(line)
            if line and line != last:
                out.append((cue.start_in_seconds, line))
                last = line
    return out


def to_text(vtt: str | Path) -> str:
    """Model-ready plain text: deduped lines joined into a single string."""
    return " ".join(line for _, line in dedup(vtt))


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
		try:
			vid: str = r[Columns.VIDEO_ID]
			path = Path(r[Columns.TRANSCRIPT_PATH])
			if path.is_file():
				transcripts[vid] = to_text(path.read_text(errors="strict"))
		except:
			missing.append(vid)
	print(f"Encountered {len(missing)} missing or erroring files while processing \n : {missing}")
	return transcripts
