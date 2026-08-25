"""A VLM stand-in, so the runner can be tested without a GPU or a real video."""

from pathlib import Path

from config import Columns
from vlm.base import VLM


class FakeVLM(VLM):
    """
    Records how it was called and replies from a qid -> completion table.

    Attributes populated during a run:
        prepared: video ids passed to `prepare_clip`, in order.
        batch_sizes: size of every batch handed to `answer`, in order.
        asked: qids seen by `answer`, in order.
        transcripts_seen: transcript text per `prepare_clip` call.
    """

    def __init__(
        self,
        replies: dict[str, str] | None = None,
        default_reply: str = "0",
        max_batch_size: int = 1,
        undecodable: tuple[str, ...] = (),
    ):
        self.replies = replies or {}
        self.default_reply = default_reply
        self.max_batch_size = max_batch_size
        self.undecodable = set(undecodable)

        self.prepared: list[str] = []
        self.batch_sizes: list[int] = []
        self.asked: list[str] = []
        self.transcripts_seen: list[str] = []
        self.use_transcript_seen: list[bool] = []

    def prepare_clip(self, video_path, transcript):
        vid = Path(video_path).stem
        if vid in self.undecodable:
            raise RuntimeError(f"cannot decode {vid}")
        self.prepared.append(vid)
        self.transcripts_seen.append(transcript)
        return {"vid": vid, "transcript": transcript}

    def answer(self, clip, rows, use_transcript=True):
        assert len(rows) <= self.max_batch_size, f"{len(rows)} rows > cap {self.max_batch_size}"
        self.batch_sizes.append(len(rows))
        self.use_transcript_seen.append(use_transcript)
        out = []
        for r in rows:
            self.asked.append(r[Columns.QID])
            out.append(self.replies.get(r[Columns.QID], self.default_reply))
        return out


def make_row(qid: str, vid: str, answer_idx: int | None = 0, **extra) -> dict:
    """One SIQ2-shaped QA record. `answer_idx=None` mimics the unlabeled test split."""
    row = {
        Columns.QID: qid,
        Columns.VIDEO_ID: vid,
        Columns.QUESTION: f"question for {qid}?",
        Columns.ANSWER_0: "zero",
        Columns.ANSWER_1: "one",
        Columns.ANSWER_2: "two",
        Columns.ANSWER_3: "three",
        Columns.VIDEO_PATH: f"/videos/{vid}.mp4",
    }
    if answer_idx is not None:
        row[Columns.ANSWER_IDX] = answer_idx
    row.update(extra)
    return row
