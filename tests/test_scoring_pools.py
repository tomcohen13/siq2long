"""
Pool construction and rank arithmetic for oracle find.

`question_pools` decides *what the oracle competes against*, which is the difference
between the task (`within`: the other minutes of the same video) and its control
(`cross`: the same number of chunks drawn from other videos). Getting the pool wrong
changes every retrieval number in the paper without raising anything, so it is pinned
here rather than eyeballed.
"""

import numpy as np
import pytest
import torch

from scoring import POOL_TYPES, question_pools, rank_of


def make_meta(sizes: dict[str, int], oracles: dict[str, int], query_vid: list[str]) -> dict:
    """A minimal artifact `meta`: `sizes` chunks per video, laid out in one flat table."""
    chunk_ids = [(vid, i) for vid, n in sizes.items() for i in range(n)]
    return {
        "chunk_ids": chunk_ids,
        "oracle_idx": oracles,
        "query_vid": query_vid,
        "qids": [f"q{i}" for i in range(len(query_vid))],
    }


#  A: rows 0,1,2  oracle -> row 1
#  B: rows 3,4    oracle -> row 3
#  C: rows 5,6,7,8  oracle -> row 8
SIZES = {"A": 3, "B": 2, "C": 4}
ORACLES = {"A": 1, "B": 0, "C": 3}
QUERY_VID = ["A", "C", "B", "A"]


@pytest.fixture
def meta():
    return make_meta(SIZES, ORACLES, QUERY_VID)


@pytest.fixture
def wide_meta():
    """20 videos x 5 chunks -- enough distractors that two seeds should disagree."""
    sizes = {f"v{i}": 5 for i in range(20)}
    return make_meta(sizes, {v: 2 for v in sizes}, [f"v{i}" for i in range(20)])


# --- within: the task --------------------------------------------------------

def test_within_pool_is_the_videos_own_chunks_in_order(meta):
    assert question_pools(meta, "within") == [
        ([0, 1, 2], 1),
        ([5, 6, 7, 8], 8),
        ([3, 4], 3),
        ([0, 1, 2], 1),
    ]


def test_within_oracle_is_a_member_not_an_addition(meta):
    """The oracle is one of the video's chunks; appending it would double-count."""
    for rows, oracle in question_pools(meta, "within"):
        assert rows.count(oracle) == 1


def test_within_pool_size_is_the_chunk_count(meta):
    sizes = [len(rows) for rows, _ in question_pools(meta, "within")]
    assert sizes == [SIZES[v] for v in QUERY_VID]


# --- cross: the control ------------------------------------------------------

def test_cross_matches_within_on_pool_size(meta):
    """Matched k keeps the random floor at 1/k in both conditions, so only provenance differs."""
    within = [len(rows) for rows, _ in question_pools(meta, "within")]
    cross = [len(rows) for rows, _ in question_pools(meta, "cross")]
    assert within == cross


def test_cross_contains_the_oracle_exactly_once(meta):
    for rows, oracle in question_pools(meta, "cross"):
        assert rows.count(oracle) == 1


def test_cross_draws_no_distractor_from_the_same_video(meta):
    """The whole video is excluded, not just the oracle -- a mixed pool blunts the contrast."""
    own_rows = {"A": {0, 1, 2}, "B": {3, 4}, "C": {5, 6, 7, 8}}
    for (rows, oracle), vid in zip(question_pools(meta, "cross"), QUERY_VID):
        distractors = set(rows) - {oracle}
        assert not (distractors & own_rows[vid])


def test_cross_samples_without_replacement(meta):
    for rows, _ in question_pools(meta, "cross"):
        assert len(set(rows)) == len(rows)


def test_cross_stays_inside_the_chunk_table(meta):
    n = len(meta["chunk_ids"])
    for rows, _ in question_pools(meta, "cross"):
        assert all(0 <= r < n for r in rows)


def test_cross_oracle_is_the_same_row_within_reports(meta):
    """Callers must not need to know which branch built the pool."""
    within = question_pools(meta, "within")
    cross = question_pools(meta, "cross")
    assert [o for _, o in within] == [o for _, o in cross]


# --- reproducibility ---------------------------------------------------------

def test_same_seed_reproduces_the_draw(wide_meta):
    assert question_pools(wide_meta, "cross", seed=7) == question_pools(wide_meta, "cross", seed=7)


def test_different_seeds_draw_differently(wide_meta):
    assert question_pools(wide_meta, "cross", seed=0) != question_pools(wide_meta, "cross", seed=1)


def test_within_ignores_the_seed(meta):
    assert question_pools(meta, "within", seed=0) == question_pools(meta, "within", seed=99)


# --- alignment and validation ------------------------------------------------

def test_one_pool_per_question(meta):
    assert len(question_pools(meta, "within")) == len(meta["qids"])
    assert len(question_pools(meta, "cross")) == len(meta["qids"])


def test_repeated_video_gets_the_same_within_pool(meta):
    """Questions 0 and 3 are both about A; the task pool cannot depend on the question."""
    pools = question_pools(meta, "within")
    assert pools[0] == pools[3]


@pytest.mark.parametrize("pool_type", POOL_TYPES)
def test_declared_pool_types_are_accepted(meta, pool_type):
    assert question_pools(meta, pool_type)


def test_unknown_pool_type_is_rejected(meta):
    with pytest.raises(ValueError, match="unknown pool_type"):
        question_pools(meta, "elsewhere")


# --- rank_of -----------------------------------------------------------------

def test_best_score_ranks_zero():
    assert rank_of(torch.tensor([0.9, 0.1, 0.2]), 0.9) == 0


def test_worst_score_ranks_last():
    assert rank_of(torch.tensor([0.9, 0.1, 0.2]), 0.1) == 2


def test_ties_are_broken_pessimistically():
    """argmax would call a three-way tie a hit; it is not one."""
    assert rank_of(torch.tensor([0.5, 0.5, 0.5]), 0.5) == 2


def test_one_tie_costs_one_rank():
    assert rank_of(torch.tensor([0.5, 0.5, 0.1]), 0.5) == 1


def test_single_chunk_pool_is_always_a_hit():
    assert rank_of(torch.tensor([0.42]), 0.42) == 0


def test_numpy_and_torch_agree():
    scores = [0.3, 0.7, 0.7, 0.1]
    assert rank_of(torch.tensor(scores), 0.7) == rank_of(np.array(scores), 0.7)


def test_rank_is_a_plain_int():
    """It indexes and gets compared downstream; a 0-d tensor would propagate silently."""
    assert type(rank_of(torch.tensor([0.9, 0.1]), 0.9)) is int
