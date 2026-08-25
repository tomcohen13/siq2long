
import pandas as pd

from pathlib import Path
from config import DATASET_TO_DIR, Columns

def load_qa(split: str, dataset: str) -> pd.DataFrame:
	path = DATASET_TO_DIR.get(dataset) / "qa" / f"qa_{split}.json"
	print("loading from %s...", path)
	df = pd.read_json(path, lines=True)
	return df


def find_downloaded_files(path_to_dataset: str | Path, to_dataframe: bool = False):
	"""
	Find and return all video ids that have both a video and a transcript in the dataset directory.

	Returns:
		pd.DataFrame with columns:
			- vid_name
			- path_to_video
			- path_to_transcript
	"""
	if isinstance(path_to_dataset, str):
		path_to_dataset = Path(path_to_dataset)

	videos_dir = path_to_dataset / "video"
	transcripts_dir = path_to_dataset / "transcript"

	videos_df = pd.DataFrame(
		list(
			{Columns.VIDEO_ID: v.stem, Columns.VIDEO_PATH: v}
			for v in videos_dir.iterdir()
			if not v.stem.startswith("._")
		)
	)

	transcripts_df = pd.DataFrame(
		list(
			{Columns.VIDEO_ID: v.stem, Columns.VIDEO_PATH: v}
			for v in transcripts_dir.iterdir()
			if not v.stem.startswith("._")
			and v.name.endswith(".vtt")
		)
	)

	both = videos_df.merge(transcripts_df, on=Columns.VIDEO_ID, how="inner")
	if to_dataframe:
		return both
	return both.to_dict(orient="index")
