"""
Build the SIQ2-Long chunk manifest: one entry per video, describing how it is cut.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from config import Columns
from data.load import find_downloaded_files
from data.oracles import compute_segments_around_oracle, load_oracles
from data.videos import get_duration

logger = logging.getLogger(__name__)

#: Videos cut into more chunks than this are dropped. Retrieval over a 97-chunk pool is a
#: different (and hopeless) problem from retrieval over ten, and a handful of very long
#: videos would otherwise dominate any mean that weights videos equally.
MAX_CHUNKS = 20

#: A video whose oracle is its only chunk scores 100% for free and cannot be got wrong,
#: so it inflates every retrieval metric without testing anything.
MIN_CHUNKS = 2


@dataclass
class ManifestStats:
    """Why each video is in or out, so coverage is reported rather than discovered later."""

    downloaded: int = 0
    no_trim: list[str] = field(default_factory=list)
    probe_failed: list[str] = field(default_factory=list)
    too_short: list[str] = field(default_factory=list)  # oracle is the whole video
    too_many_chunks: list[str] = field(default_factory=list)
    kept: int = 0

    def log(self) -> None:
        logger.info("downloaded videos with both mp4 and vtt : %d", self.downloaded)
        for name, dropped in (
            ("absent from trims.json", self.no_trim),
            ("ffprobe failed", self.probe_failed),
            (f"fewer than {MIN_CHUNKS} chunks", self.too_short),
            (f"more than {MAX_CHUNKS} chunks", self.too_many_chunks),
        ):
            if dropped:
                logger.warning("dropped %4d (%s), e.g. %s", len(dropped), name, dropped[:3])
        logger.info("kept %d videos", self.kept)


def _durations(paths: dict[str, Path], workers: int = 16) -> dict[str, float]:
    """ffprobe every video. IO-bound subprocesses, so threads are the right pool."""

    def probe(item):
        vid, path = item
        try:
            return vid, get_duration(str(path))
        except Exception:
            logger.debug("ffprobe failed for %s", vid, exc_info=True)
            return vid, None

    with ThreadPoolExecutor(workers) as pool:
        return dict(pool.map(probe, paths.items()))


def build_manifest(
    path_to_media: str | Path,
    trims_path: str | Path,
    max_chunks: int = MAX_CHUNKS,
    min_chunks: int = MIN_CHUNKS,
    **chunk_kwargs,
) -> tuple[dict[str, dict], ManifestStats]:
    """
    Chunk every downloaded video around its oracle.

    Args:
        path_to_media: the dataset's media root, holding `video/` and `transcript/`
        trims_path: the original SocialIQ-2.0 `trims.json`
        max_chunks / min_chunks: retrieval-pool bounds, see the constants above
        chunk_kwargs: forwarded to `compute_segments_around_oracle` (chunk_size, buffer_size)

    Returns:
        `{vid: {"duration": float, "chunks": [[start, end], ...], "oracle_idx": int}}`
        and the stats describing who was dropped and why.
    """
    files = find_downloaded_files(path_to_media, to_dataframe=True)
    oracles = load_oracles(trims_path)
    stats = ManifestStats(downloaded=len(files))

    paths = dict(zip(files[Columns.VIDEO_ID], files[Columns.VIDEO_PATH]))
    known = {vid: p for vid, p in paths.items() if vid in oracles}
    stats.no_trim = [vid for vid in paths if vid not in oracles]

    durations = _durations(known)

    manifest: dict[str, dict] = {}
    for vid, duration in durations.items():
        if duration is None:
            stats.probe_failed.append(vid)
            continue
        chunks, oracle_idx = compute_segments_around_oracle(
            oracles[vid]["oracle"], duration, **chunk_kwargs
        )
        if len(chunks) < min_chunks:
            stats.too_short.append(vid)
        elif len(chunks) > max_chunks:
            stats.too_many_chunks.append(vid)
        else:
            manifest[vid] = {
                "duration": duration,
                "chunks": chunks,
                "oracle_idx": oracle_idx,
            }

    stats.kept = len(manifest)
    return manifest, stats


def write_manifest(manifest: dict[str, dict], path: str | Path) -> None:
    """Write the manifest as JSON, keys sorted so diffs between rebuilds are readable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True))


def load_manifest(path: str | Path) -> dict[str, dict]:
    """Read a manifest written by `write_manifest`."""
    return json.loads(Path(path).read_text())
