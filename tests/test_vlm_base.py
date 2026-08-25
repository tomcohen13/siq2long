"""Regression tests for the prompt the models actually send."""

from vlm.base import VLM
from vlm.qwen import Qwen2_5VL, Qwen3VL
from vlm.videollama import VideoLlama3
from fakes import make_row


def test_render_question_is_callable_on_class_and_instance():
    """It used to take `record` as its only arg, so `self.render_question(r)` raised."""
    row = make_row("q1", "vidA")
    assert VLM.render_question(row) == VideoLlama3.render_question(row)


def test_render_question_numbers_options_from_zero():
    prompt = VLM.render_question(make_row("q1", "vidA"))
    assert prompt.splitlines()[0] == "question for q1?"
    assert "0. zero" in prompt
    assert "1. one" in prompt
    assert "2. two" in prompt
    assert "3. three" in prompt


def test_render_question_asks_for_a_bare_number():
    prompt = VLM.render_question(make_row("q1", "vidA"))
    assert prompt.rstrip().endswith("Answer with the number of the correct option, and nothing else.")


def test_render_question_uses_the_row_question():
    row = make_row("q1", "vidA")
    row["q"] = "How does she feel?"
    assert "How does she feel?" in VLM.render_question(row)


def test_batch_size_caps_are_declared():
    """The runner batches off these, so they must be set on the classes, not just the ABC."""
    assert VideoLlama3.max_batch_size == 1
    for cls in (Qwen2_5VL, Qwen3VL):
        assert cls.max_batch_size >= 1
