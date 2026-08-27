

from abc import ABC, abstractmethod
from typing import Any

from config import ANSWER_KEYS, Columns


class VLM(ABC):
    """
    Abstraction of video-language model, 
    to be inherited and implemented by individual models.
    """

    max_batch_size: int = 1

    #: False -> [video, transcript, question]; True -> [question, transcript, video].
    #: Placeholder position measurably matters for some model families -- LLaVA's model
    #: card puts all text before the video placeholder, Qwen's examples lead with the
    #: video. A flag rather than a per-backend hardcode, so a run can A/B it and so
    #: existing results stay reproducible against whichever order produced them.
    question_first: bool = False

    @abstractmethod
    def prepare_clip(self, video_path: str, transcript: str) -> Any:
        """Whatever this backend needs to represent one clip. Opaque to the pipeline."""
        raise NotImplementedError("Must implement method within inheriting class")

    @abstractmethod
    def answer(self, clip: Any, rows: list[dict], use_transcript: bool) -> list[str]:
        """Returns one raw completion per row."""
        raise NotImplementedError("Must implement method within inheriting class")

    PROMPT = (
        "{question}\n{options}\n"
        "Answer with the number of the correct option, and nothing else."
    )

    @classmethod
    def render_question(cls, record: dict) -> str:
        options = "\n".join(f"{i}. {record[k]}" for i, k in enumerate(ANSWER_KEYS))
        return cls.PROMPT.format(question=record[Columns.QUESTION], options=options)

    def content_parts(self, record: dict, transcript: str = "") -> list[tuple[str, str]]:
        """
        The prompt's parts in order, as ("video", "") and ("text", str) pairs.

        Backends map these onto whatever content dict their processor expects, so the
        ordering lives here once instead of in four places. An empty `transcript`
        contributes no part, which is what the no-transcript condition wants.
        """
        text_parts = []
        if transcript:
            text_parts.append(("text", f"Transcript:\n{transcript}"))
        question = ("text", self.render_question(record))

        if self.question_first:
            return [question, *text_parts, ("video", "")]
        return [("video", ""), *text_parts, question]
