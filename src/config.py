"""Project-wide configurations"""
import os
from enum import StrEnum
from pathlib import Path
from dotenv import load_dotenv

# Anchored to this file, not the cwd, so imports behave the same from a notebook,
# a script, or pytest.
ROOT_DIR = Path(__file__).resolve().parents[1]

load_dotenv(ROOT_DIR / ".env")


class Datasets(StrEnum):
    SIQ2 = "siq2"
    SIQ2LONG = "siq2long"

class Dirs(StrEnum):
    """Subdirectory names under a dataset root, so a path is never spelled twice."""
    QA = "qa"
    SPLITS = "splits"
    VIDEO = "video"
    TRANSCRIPT = "transcript"
    EMBEDDINGS = "embeddings"

class Files(StrEnum):
    """File names at a dataset root. Where the root is depends on the caller, the name does not."""
    MANIFEST = "video_chunks.json"
    TRIMS = "trims.json"

class Columns(StrEnum):
    # original siq2 columns
    QID = "qid"
    VIDEO_ID = "vid_name"
    QUESTION = "q"
    ANSWER_0 = "a0"
    ANSWER_1 = "a1"
    ANSWER_2 = "a2"
    ANSWER_3 = "a3"
    ANSWER_IDX = "answer_idx"  # absent in the test split
    ANSWER_TEXT = "ans_corr"   # the correct answer's text; absent in the test split

    VIDEO_PATH = "path_to_video"
    TRANSCRIPT_PATH = "path_to_transcript"

    # siq2-long specific
    ORACLE = "oracle_idx"

# Where dataset media lives -- video, transcripts, embeddings.
# Amounts to tens of GBs, so not assumed to be the same directory as the dataset repo.
DATASET_TO_MEDIA_DIR = {
	Datasets.SIQ2: Path(os.getenv("PATH_TO_SIQ2", ROOT_DIR / Datasets.SIQ2)),
	Datasets.SIQ2LONG: Path(os.getenv("PATH_TO_SIQ2LONG", ROOT_DIR / Datasets.SIQ2LONG)),
}

ANSWER_KEYS = [Columns.ANSWER_0, Columns.ANSWER_1, Columns.ANSWER_2, Columns.ANSWER_3]

