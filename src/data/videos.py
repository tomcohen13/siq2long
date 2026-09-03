"""Video probing and frame sampling."""

import subprocess
from pathlib import Path

import numpy as np

def get_duration(filename: str) -> float:
    """Get duration of video file in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        filename,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err = (result.stderr or "").strip() or "ffprobe failed with no stderr"
        raise RuntimeError(f"ffprobe exited {result.returncode}: {err}")
    raw = (result.stdout or "").strip()
    if not raw:
        raise RuntimeError("ffprobe returned empty duration")
    return float(raw)


def frame_indices(total_frames: int, num_frames: int) -> np.ndarray:
    """
    `num_frames` indices spread evenly over `[0, total_frames - 1]`, inclusive.

    Rounded, not truncated. `np.linspace(..., dtype=int)` casts, which floors every
    index and biases the whole sample toward the start of the clip -- subtly wrong in
    a way that never raises. Asking for more frames than exist returns every frame
    rather than padding, because padding hides a decode problem behind plausible input.
    """
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")

    n = min(num_frames, total_frames)
    return np.linspace(0, total_frames - 1, n).round().astype(int)


def window_frame_indices(
    metadata, num_frames: int, start: float | None = None, end: float | None = None
) -> np.ndarray:
    """
    `num_frames` indices over the `[start, end]` second window, or the whole video.

    Converts seconds to a frame range using the video's own fps and then defers to
    `frame_indices`, so rounding behaves identically for a chunk and for a full video.
    `end` past the last frame clamps: a video's final chunk legitimately runs to the
    duration ffprobe reported, which can round a frame or two past the frame count.

    Kept apart from decoding so it can be tested without a video file, and because the
    two guards below are the ones that matter -- pyav takes `total_num_frames` from
    container metadata, which has been wrong on this dataset before. Failing here beats
    silently sampling a bogus range.
    """
    if (start is None) != (end is None):
        raise ValueError("pass both `start` and `end`, or neither")

    total = metadata.total_num_frames
    if not total or total <= 0:
        raise RuntimeError(f"video reports {total} frames; refusing to sample")
    if start is None:
        return frame_indices(total, num_frames)

    fps = metadata.fps
    if not fps:
        raise RuntimeError("video reports no frame rate; cannot convert seconds to frames")
    lo = max(0, round(start * fps))
    hi = min(total - 1, round(end * fps) - 1)
    if hi < lo:
        raise ValueError(f"empty frame range for window [{start}, {end}] at {fps} fps")
    return frame_indices(hi - lo + 1, num_frames) + lo


def sample_frames(
    video_path: str,
    num_frames: int,
    start: float | None = None,
    end: float | None = None,
    backend: str = "pyav",
) -> np.ndarray:
    """
    Decode `num_frames` evenly spaced frames as a [T, H, W, C] uint8 array.

    With `start`/`end` the frames come from that second-window only, and nothing is cut
    on disk -- the window becomes a set of frame indices and just those are decoded. That
    is what lets a chunked video be encoded without materializing clips, whose keyframe-
    snapped boundaries would move the oracle window off the SIQ2 trim.

    `transformers.video_utils.load_video` rather than decord, which publishes no arm64
    macOS wheel; load_video's pyav backend installs everywhere and accepts the
    `sample_indices_fn` this needs. The array form matters downstream too -- transformers
    skips its own video decoding when handed frames.
    """
    from transformers.video_utils import load_video  # lazy: heavy import

    def indices(metadata, **kwargs):
        return window_frame_indices(metadata, num_frames, start, end)

    frames, _ = load_video(str(video_path), sample_indices_fn=indices, backend=backend)
    if len(frames) == 0:
        raise RuntimeError(f"decoded 0 frames from {video_path}")
    return frames


def sample_windows(
    video_path: str,
    windows: list[list[float]],
    frames_per_window: int,
    backend: str = "pyav",
) -> list[np.ndarray]:
    """
    Decode `frames_per_window` from each `[start, end]` window, in one pass over the file.

    PyAV reads forward from the start instead of seeking, so decoding windows one at a
    time re-reads the whole file each time -- about 4x slower here, for identical frames.

    Windows must be ascending and non-overlapping, as the chunker emits them. Overlapping
    ones come back short, since one pass returns frames in file order and drops repeats;
    the count check catches that rather than letting rows misalign.
    """
    from transformers.video_utils import load_video  # lazy: heavy import

    sizes: list[int] = []

    def indices(metadata, **kwargs):
        per_window = [
            window_frame_indices(metadata, frames_per_window, s, e)
            for s, e in windows
        ]
        sizes[:] = [len(i) for i in per_window]
        return np.concatenate(per_window)

    frames, _ = load_video(str(video_path), sample_indices_fn=indices, backend=backend)
    if len(frames) != sum(sizes):
        raise RuntimeError(
            f"decoded {len(frames)} frames for {sum(sizes)} requested from {video_path}"
        )

    out, start_row = [], 0
    for size in sizes:
        out.append(frames[start_row : start_row + size])
        start_row += size
    return out


def slice_video(video_path: str, start: float, end: float, dest: str | Path, crf: int = 18) -> Path:
    """
    Write `[start, end]` to its own file, re-encoded. Returns `dest`, reusing it if present.

    Only the VLM path needs this. The encoders read frame indices straight out of the
    source, but the VLM backends take a video *path* and decode it whole, so a chunk has
    to exist as a file before one can answer from it.

    Re-encoded rather than stream-copied because `-c copy` snaps the cut to the nearest
    keyframe -- measured up to 4.4s of drift on this dataset, which would move the oracle
    window off the SIQ2 trim it is defined by. At crf 18 the second generation is visually
    near-lossless and costs about 0.7s per chunk. Audio is dropped: none of the backends
    listen to it.
    """
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    tmp = dest.with_suffix(".partial.mp4")  # never leave a truncated chunk behind a cache hit
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-y",
        "-ss", f"{start:.3f}", "-i", str(video_path), "-t", f"{end - start:.3f}",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg exited {result.returncode} cutting {video_path}: {result.stderr.strip()}")
    tmp.rename(dest)
    return dest
