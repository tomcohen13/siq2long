

import subprocess

import numpy as np

from config import Columns


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


def sample_frames(video_path: str, num_frames: int) -> np.ndarray:
    """
    Decode `num_frames` evenly spaced frames as a [T, H, W, C] uint8 array.

    decord rather than PyAV or torchcodec: container-reported frame counts have been
    unreliable on this dataset, decord's index is derived from the stream itself, and
    torchcodec needs its shared libraries to match the installed CUDA (see DEPLOY.md).
    The array form also matters downstream -- transformers skips its own video decoding
    entirely when handed frames, so backends built on this are immune to that.
    """
    from decord import VideoReader  # lazy: importable without decord installed

    reader = VideoReader(str(video_path))
    total = len(reader)
    if total == 0:
        raise RuntimeError(f"decoded 0 frames from {video_path}")
    return reader.get_batch(frame_indices(total, num_frames)).asnumpy()


def to_messages(record: dict) -> list[dict]:
	"""
	Convert dict record to LLM-compatible messages list of the format:
		{{
			"role": "user",
			"content": [
				{"type": "video", "video": ...}
				{"type": "tesx", "text": [see template]},
			]
		}}

	Args:
		record: a dictionary, assumed to have the following SIQ2.0 fields:
			- vid_name: video id
			- q: the question
			- [a0, a1, a2, a3]: candidate answers
			- path_to_video
			- transcript
	"""

	TEMPLATE = """
	Context:
	{transcript}

	Question:
	{question}
	
	Answer options:
	{answers}

	Respond only with the index of the most likely correct answer to the question,
	nothing before or after:
	"""

	try:
		answers = "\n".join(
			[
				f"{col_name}: {record[col_name]}\n"
				for col_name in ["a0", "a1", "a2", "a3"]
			]
		)
		text = TEMPLATE.format(
			transcript=record["transcript"],
			question=record[Columns.QUESTION],
			answers=answers
		)
		return [
			{
				"role": "user",
				"content": [
					{
						"type": "video",
						"video": f"file://{record['path_to_video']}",
						"max_pixels": 360 * 420,  # TODO: remove from function, this is model dependent
						"fps": 1.0,  # TODO: remove from function, this is model dependent
					},
					{"type": "text", "text": text},
				]
			}
		]
	except Exception as e:
		print(e)
	return []