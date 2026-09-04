"""
Query rendering and the row ordering behind it.

The query tensors in a cache are positional -- row `i` belongs to `meta["qids"][i]`. Every
later consumer that rebuilds query text has to reproduce that order exactly, or it scores
the right chunks against the wrong questions and nothing raises. That alignment, and the
three query forms, are what these pin.
"""

import pandas as pd
import pytest

from config import ANSWER_KEYS, Columns
from retrieval import render_queries_in_order, render_query, rows_in_query_order


def make_artifact(qids):
    """A cache carrying only the qid ordering that query rendering depends on."""
    return {"checkpoint": "test", "num_frames": 0, "tensors": {}, "meta": {"qids": list(qids)}}


def qa_frame(qids):
    """
    One QA row per qid. Every string names its own qid, so an ordering mistake shows up as
    the wrong qid in the output rather than as an off-by-one nobody can read.
    """
    return pd.DataFrame(
        [
            {
                Columns.QID: qid,
                Columns.QUESTION: f"question {qid}",
                Columns.ANSWER_IDX: 0,
                Columns.ANSWER_TEXT: f"answer {qid}",
                **{k: f"opt{j}-{qid}" for j, k in enumerate(ANSWER_KEYS)},
            }
            for qid in qids
        ]
    )


@pytest.fixture
def artifact():
    return make_artifact(["q0", "q1"])


# --- render_query: the three forms -------------------------------------------

def test_question_form_is_the_bare_question():
    row = qa_frame(["q0"]).iloc[0].to_dict()
    assert render_query(row, "question") == "question q0"


def test_options_form_appends_all_four():
    row = qa_frame(["q0"]).iloc[0].to_dict()
    rendered = render_query(row, "question+options")
    assert rendered.startswith("question q0 ")
    assert all(f"opt{j}-q0" in rendered for j in range(4))


def test_answer_form_is_the_gold_text_alone():
    """The diagnostic form: a declarative statement, with no question attached."""
    row = qa_frame(["q0"]).iloc[0].to_dict()
    assert render_query(row, "answer") == "answer q0"


def test_answer_form_rejects_a_row_with_no_gold():
    """The test split has none; returning the question there would compare a form to itself."""
    row = qa_frame(["q0"]).iloc[0].to_dict() | {Columns.ANSWER_TEXT: None}
    with pytest.raises(ValueError, match="no gold answer"):
        render_query(row, "answer")


def test_unknown_form_is_rejected():
    row = qa_frame(["q0"]).iloc[0].to_dict()
    with pytest.raises(ValueError, match="unknown query form"):
        render_query(row, "vibes")


# --- rows_in_query_order -----------------------------------------------------

def test_rows_follow_the_cache_not_the_frame(artifact):
    """qa in any order; the output must line up with meta['qids'] or scores misalign."""
    rows = rows_in_query_order(artifact, qa_frame(["q1", "q0"]))
    assert [row[Columns.QID] for row in rows] == ["q0", "q1"]


def test_rows_tolerate_duplicate_qids(artifact):
    """Duplicated qids made .loc return a DataFrame and broke a 500-question run once."""
    qa = qa_frame(["q0", "q1"])
    doubled = pd.concat([qa, qa], ignore_index=True)
    assert rows_in_query_order(artifact, doubled) == rows_in_query_order(artifact, qa)


def test_rows_reject_a_missing_qid(artifact):
    with pytest.raises(KeyError, match="not in qa"):
        rows_in_query_order(artifact, qa_frame(["q0"]))


def test_one_row_per_cached_question(artifact):
    assert len(rows_in_query_order(artifact, qa_frame(["q0", "q1"]))) == 2


# --- render_queries_in_order -------------------------------------------------

def test_rendered_queries_follow_cache_order(artifact):
    """The frame is reversed; the output must still be q0 then q1."""
    rendered = render_queries_in_order(artifact, qa_frame(["q1", "q0"]), form="question")
    assert rendered == ["question q0", "question q1"]


def test_rendered_queries_default_to_question_plus_options(artifact):
    rendered = render_queries_in_order(artifact, qa_frame(["q0", "q1"]))
    assert rendered[0].startswith("question q0 ") and "opt0-q0" in rendered[0]


def test_rendered_queries_pass_the_form_through(artifact):
    rendered = render_queries_in_order(artifact, qa_frame(["q0", "q1"]), form="answer")
    assert rendered == ["answer q0", "answer q1"]
