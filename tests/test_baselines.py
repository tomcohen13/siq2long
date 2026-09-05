"""
Text-only retrieval baselines.

BM25 is scored through the same `question_pools` / `rank_of` path as the encoders, so what
is pinned here is the wiring: that pool rows index the corpus correctly, and that the
oracle is found within the pool rather than in the global table. BGE differs only in how
scores are produced, so it is not exercised here -- that would download a checkpoint to
test arithmetic already covered. Query rendering lives in `test_retrieval_queries.py`.
"""

import numpy as np
import pytest

from encoders.baselines import bm25_ranks, tokenize


def make_artifact(chunk_texts: dict[str, list[str]], oracles: dict[str, int], query_vid: list[str]):
    """A cache carrying only what the text baselines read."""
    chunk_ids = [(vid, i) for vid, texts in chunk_texts.items() for i in range(len(texts))]
    return {
        "checkpoint": "test",
        "num_frames": 0,
        "tensors": {},
        "meta": {
            "chunk_ids": chunk_ids,
            "chunk_texts": [t for texts in chunk_texts.values() for t in texts],
            "oracle_idx": oracles,
            "query_vid": query_vid,
            "qids": [f"q{i}" for i in range(len(query_vid))],
        },
    }


# Chunk 1 of each video is the oracle and is the only one mentioning its query's terms.
TEXTS = {
    "A": ["weather report clouds", "she apologised for shouting", "traffic on the bridge"],
    "B": ["cooking pasta tonight", "he refused the handshake firmly", "a train timetable"],
}
ORACLES = {"A": 1, "B": 1}
QUERY_VID = ["A", "B"]
QUERIES = ["why did she apologise for shouting", "why did he refuse the handshake"]


@pytest.fixture
def artifact():
    return make_artifact(TEXTS, ORACLES, QUERY_VID)


# --- tokenize ----------------------------------------------------------------

def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Are the TWO respectful?") == ["are", "the", "two", "respectful"]


def test_tokenize_keeps_contractions_whole():
    """Splitting isn't into is/nt would silently change every BM25 score."""
    assert tokenize("isn't") == ["isn't"]


def test_tokenize_keeps_digits():
    assert tokenize("chunk 42") == ["chunk", "42"]


# --- bm25_ranks --------------------------------------------------------------

def test_bm25_finds_a_lexically_obvious_oracle(artifact):
    ranks, _ = bm25_ranks(artifact, QUERIES)
    assert list(ranks) == [0, 0]


def test_bm25_reports_pool_sizes(artifact):
    _, pools = bm25_ranks(artifact, QUERIES)
    assert list(pools) == [3, 3]


def test_bm25_ranks_within_the_pool_not_the_global_table(artifact):
    """Video B's oracle is global row 4; its rank must still be 0 of 3, not 4."""
    ranks, pools = bm25_ranks(artifact, QUERIES)
    assert ranks[1] < pools[1]


def test_bm25_scores_only_the_questions_own_video(artifact):
    """Swapping the queries must not let A's question retrieve B's oracle."""
    ranks, _ = bm25_ranks(artifact, list(reversed(QUERIES)))
    assert ranks[0] > 0 and ranks[1] > 0


def test_bm25_puts_the_oracle_last_when_nothing_matches(artifact):
    """All-zero scores are a tie, and a tie is not a hit."""
    ranks, pools = bm25_ranks(artifact, ["zzz qqq", "zzz qqq"])
    assert list(ranks) == [p - 1 for p in pools]


def test_bm25_cross_pool_keeps_the_pool_size(artifact):
    _, within = bm25_ranks(artifact, QUERIES, pool_type="within")
    _, cross = bm25_ranks(artifact, QUERIES, pool_type="cross")
    assert list(within) == list(cross)


def test_bm25_returns_arrays(artifact):
    ranks, pools = bm25_ranks(artifact, QUERIES)
    assert isinstance(ranks, np.ndarray) and isinstance(pools, np.ndarray)


