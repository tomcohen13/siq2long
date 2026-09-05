
import logging

import pandas as pd

from pathlib import Path
from config import ROOT_DIR, Columns, Datasets, Dirs

logger = logging.getLogger(__name__)

def load_qa(split: str, dataset: str = Datasets.SIQ2LONG) -> pd.DataFrame:
	"""
	The QA rows for one split: "train", "val" or "test".

	Read from the repo rather than from wherever the media lives, because these are a few MB
	and no result can be read without them.

	`dataset` chooses whose copy. SIQ2 and SIQ2-Long ask the same questions about the same
	videos and differ only in how much of each video the model is shown -- the 60s oracle
	trim or the full original -- so each keeps its own copy and either can be run unchanged.
	"""
	if dataset not in set(Datasets):
		raise ValueError(f"unknown dataset {dataset!r}, options are: {[*Datasets]}")
	path = ROOT_DIR / dataset / Dirs.QA / f"qa_{split}.json"
	if not path.exists():
		raise FileNotFoundError(f"no {dataset} QA for split {split!r} at {path}")
	logger.info("loading %s", path)
	return pd.read_json(path, lines=True)


def find_downloaded_files(path_to_media: str | Path, to_dataframe: bool = False):
	"""
	Find and return all video ids that have both a video and a transcript.

	A video id with only one of the two is dropped: the pipeline encodes both modalities per
	chunk, so half a video is not usable.

	Args:
		path_to_media: the dataset's media root, holding `video/` and `transcript/`. This is
			the relocatable half of the dataset, so it is passed in rather than looked up --
			see DATASET_TO_MEDIA_DIR.
		to_dataframe: return the DataFrame itself rather than a dict keyed by row number.

	Returns:
		pd.DataFrame with columns:
			- vid_name
			- path_to_video
			- path_to_transcript
	"""
	if isinstance(path_to_media, str):
		path_to_media = Path(path_to_media)

	videos_dir = path_to_media / Dirs.VIDEO
	transcripts_dir = path_to_media / Dirs.TRANSCRIPT

	videos_df = pd.DataFrame(
		list(
			{Columns.VIDEO_ID: v.stem, Columns.VIDEO_PATH: v}
			for v in videos_dir.iterdir()
			if not v.stem.startswith("._")
		)
	)

	transcripts_df = pd.DataFrame(
		list(
			{Columns.VIDEO_ID: v.stem, Columns.TRANSCRIPT_PATH: v}
			for v in transcripts_dir.iterdir()
			if not v.stem.startswith("._")
			and v.name.endswith(".vtt")
		)
	)

	both = videos_df.merge(transcripts_df, on=Columns.VIDEO_ID, how="inner")
	if to_dataframe:
		return both
	return both.to_dict(orient="index")
