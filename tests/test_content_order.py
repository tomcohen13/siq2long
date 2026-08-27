"""
Prompt part ordering, shared by every backend.

Placeholder position measurably changes accuracy for some model families, so the
order is a flag and these tests pin both settings — including that the default is
video-first, which is what already-collected results were produced with.
"""

import pytest

from vlm.base import VLM
from vlm.intern import InternVL3_8B
from vlm.llava_next_video import LlavaNextVideo
from vlm.qwen import Qwen2_5VL, Qwen3VL
from vlm.videollama import VideoLlama3
from fakes import make_row


class Bare(VLM):
    """Concrete VLM with nothing but the shared behaviour under test."""

    def prepare_clip(self, video_path, transcript):
        raise NotImplementedError

    def answer(self, clip, rows, use_transcript=True):
        raise NotImplementedError


def kinds(parts):
    return [kind for kind, _ in parts]


def test_default_is_video_first():
    """Every result collected so far used this order; don't let it drift silently."""
    assert VLM.question_first is False
    for cls in (Qwen2_5VL, Qwen3VL, InternVL3_8B, LlavaNextVideo, VideoLlama3):
        assert cls.question_first is False


def test_video_first_order():
    parts = Bare().content_parts(make_row("q1", "vidA"), "a transcript")
    assert kinds(parts) == ["video", "text", "text"]
    assert parts[1][1] == "Transcript:\na transcript"
    assert "question for q1?" in parts[2][1]


def test_question_first_order():
    model = Bare()
    model.question_first = True
    parts = model.content_parts(make_row("q1", "vidA"), "a transcript")
    assert kinds(parts) == ["text", "text", "video"]
    assert "question for q1?" in parts[0][1]
    assert parts[1][1] == "Transcript:\na transcript"


@pytest.mark.parametrize("question_first", [False, True])
def test_empty_transcript_contributes_no_part(question_first):
    """The no-transcript condition must not smuggle in an empty Transcript: header."""
    model = Bare()
    model.question_first = question_first
    parts = model.content_parts(make_row("q1", "vidA"), "")
    assert kinds(parts) == (["text", "video"] if question_first else ["video", "text"])
    assert not any("Transcript:" in value for _, value in parts)


@pytest.mark.parametrize("question_first", [False, True])
def test_question_text_is_identical_across_orders(question_first):
    """Only the position changes; the rendered question must not."""
    row = make_row("q1", "vidA")
    model = Bare()
    model.question_first = question_first
    texts = [value for kind, value in model.content_parts(row, "t") if kind == "text"]
    assert VLM.render_question(row) in texts


def test_exactly_one_video_part():
    for question_first in (False, True):
        model = Bare()
        model.question_first = question_first
        for transcript in ("", "t"):
            parts = model.content_parts(make_row("q1", "vidA"), transcript)
            assert kinds(parts).count("video") == 1
