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

DATASET_TO_DIR = {
	Datasets.SIQ2: Path(os.getenv("PATH_TO_SIQ2", ROOT_DIR / "datasets/socialiq2/siq2/")),
	Datasets.SIQ2LONG: Path(os.getenv("PATH_TO_SIQ2LONG", ROOT_DIR / "datasets/siq2long/")),
}

ANSWER_KEYS = [Columns.ANSWER_0, Columns.ANSWER_1, Columns.ANSWER_2, Columns.ANSWER_3]

