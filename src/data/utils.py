

import subprocess

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