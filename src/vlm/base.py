

from abc import ABC, abstractmethod
from typing import Any

from config import ANSWER_KEYS, Columns


class VLM(ABC):
    """
    Abstraction of video-language model, 
    to be inherited and implemented by individual models.
    """

    max_batch_size: int = 1

    @abstractmethod
    def prepare_clip(self, video_path: str, transcript: str) -> Any:
        """Whatever this backend needs to represent one clip. Opaque to the pipeline."""
        raise NotImplementedError("Must implement method within inheriting class")

    @abstractmethod
    def answer(self, clip: Any, rows: list[dict], use_transcript: bool) -> list[str]:
        """Returns one raw completion per row."""
        raise NotImplementedError("Must implement method within inheriting class")

    def render_question(record):
        PROMPT = (
            "{question}\n{options}\n"
            "Answer with the number of the correct option, and nothing else."
        ).format()

        options = "\n".join(f"{i}. {record[k]}" for i, k in enumerate(ANSWER_KEYS))
        return PROMPT.format(question=record[Columns.QUESTION], options=options)
