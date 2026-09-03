"""
Score an embedding cache: where does the oracle chunk rank among its video's chunks?

Reads only what `retrieval.encode` wrote, so every variant here is a tensor op over
cached vectors -- no video is touched. Each question is ranked against the chunks of its
own video, which is the task: find the oracle inside one long video, not across a corpus.
"""

import logging
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from stats import wilson_interval

logger = logging.getLogger(__name__)

#: Chunk-side representations. `fused` is a plain renormalized mean of the two towers --
#: the untrained stand-in for the fusion adapters, not the adapters themselves.
REPRESENTATIONS = ("video", "transcript", "fused")

#: Query-side forms. The options are available at retrieval time in a multiple-choice
#: setting, but they carry content the bare question does not, so both are reported.
QUERIES = {"question": "query_q", "question+options": "query_qa"}


def chunk_matrix(tensors: dict[str, torch.Tensor], representation: str) -> torch.Tensor:
    if representation == "video":
        return tensors["chunk_video"]
    if representation == "transcript":
        return tensors["chunk_text"]
    if representation == "fused":
        return F.normalize(tensors["chunk_video"] + tensors["chunk_text"], dim=-1)
    raise ValueError(f"unknown representation {representation!r}")


def rows_by_video(chunk_ids: list) -> dict[str, list[int]]:
    """Row indices into the chunk tensors, per video, in chunk order."""
    rows = defaultdict(list)
    for row, (vid, _) in enumerate(chunk_ids):
        rows[vid].append(row)
    return rows


def oracle_ranks(bundle: dict, representation: str, query: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Where the oracle lands for each question, and how many chunks it competed against.

    Returns `(rank, pool)` per question; rank is 0-based, so 0 is a top-1 hit. Ties are
    resolved pessimistically -- the oracle is credited only with the rank it would get if
    every chunk scoring at least as high came first -- because a tie is not a hit and
    `argmax` would quietly award it one.
    """
    tensors, meta = bundle["tensors"], bundle["meta"]
    chunks = chunk_matrix(tensors, representation)
    queries = tensors[QUERIES[query]]
    rows = rows_by_video(meta["chunk_ids"])

    ranks, pools = [], []
    for qi, vid in enumerate(meta["query_vid"]):
        idx = rows[vid]
        scores = queries[qi] @ chunks[idx].T
        oracle = scores[meta["oracle_idx"][vid]]
        ranks.append(int((scores > oracle).sum() + (scores == oracle).sum() - 1))
        pools.append(len(idx))
    return np.array(ranks), np.array(pools)


def select_chunks(
    bundle: dict,
    condition: str,
    representation: str = "fused",
    query: str = "question+options",
    seed: int = 0,
) -> list[tuple[str, str, int]]:
    """
    Which chunk each question should be answered from, as `(qid, vid, chunk_idx)`.

    The qid rides along rather than leaving the caller to zip against `meta["qids"]`:
    downstream this becomes one QA row per entry, and a silent off-by-one there would
    answer every question from someone else's video without anything looking wrong.

    The conditions are the rows of the end-to-end table. `gold` is the upper bound a
    perfect retriever would reach; `top1` is what the retriever actually returns; `random`
    and `prior` are the floors the retrieval numbers are measured against, carried through
    to QA so the same reference frame holds on both halves of the paper.

    `prior` predicts the modal oracle position, clamped to each video's own chunk count --
    the trivial heuristic that beats every encoder configuration on retrieval.
    """
    meta, tensors = bundle["meta"], bundle["tensors"]
    rows = rows_by_video(meta["chunk_ids"])
    rng = np.random.default_rng(seed)

    if condition == "top1":
        chunks = chunk_matrix(tensors, representation)
        queries = tensors[QUERIES[query]]

    modal = Counter(meta["oracle_idx"][v] for v in meta["query_vid"]).most_common(1)[0][0]

    out = []
    for qi, (qid, vid) in enumerate(zip(meta["qids"], meta["query_vid"], strict=True)):
        idx = rows[vid]
        if condition == "gold":
            pick = meta["oracle_idx"][vid]
        elif condition == "top1":
            pick = int((queries[qi] @ chunks[idx].T).argmax())
        elif condition == "random":
            pick = int(rng.integers(len(idx)))
        elif condition == "prior":
            pick = min(modal, len(idx) - 1)
        else:
            raise ValueError(f"unknown condition {condition!r}")
        out.append((qid, vid, pick))
    return out


def metrics(ranks: np.ndarray, pools: np.ndarray) -> dict[str, float]:
    """
    Top-1, top-3, MRR, and the random floor these pools imply.

    The random floor is `mean(1 / pool)` rather than a constant: pools run from 2 to 20
    chunks, so a fixed 1/N would misstate the baseline for every video.
    """
    return {
        "top1": float((ranks == 0).mean()),
        "top3": float((ranks < 3).mean()),
        "mrr": float((1 / (ranks + 1)).mean()),
        "random": float((1 / pools).mean()),
        "n": int(len(ranks)),
    }


def by_pool_size(
    ranks: np.ndarray, pools: np.ndarray, bins: list[tuple[int, int]]
) -> list[dict]:
    """
    Top-1 within each chunk-count band, with a Wilson interval and the random floor.

    Aggregate accuracy conflates retrieval skill with how many chunks a video has -- a
    2-chunk video is a coin flip and a 20-chunk one is not -- so the strata are where the
    degradation claim actually lives.
    """
    out = []
    for lo, hi in bins:
        mask = (pools >= lo) & (pools <= hi)
        n = int(mask.sum())
        hits = int((ranks[mask] == 0).sum())
        low, high = wilson_interval(hits, n) if n else (float("nan"), float("nan"))
        out.append(
            {
                "band": f"{lo}-{hi}" if lo != hi else str(lo),
                "n": n,
                "top1": hits / n if n else float("nan"),
                "lo": low,
                "hi": high,
                "random": float((1 / pools[mask]).mean()) if n else float("nan"),
            }
        )
    return out
