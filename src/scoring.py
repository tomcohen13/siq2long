"""
Score an embedding cache: where does the oracle chunk rank among its video's chunks?

Reads only what `retrieval.encode` wrote, so every variant here is a tensor op over
cached vectors -- no video is touched. Each question is ranked against the chunks of its
own video, which is the task: find the oracle inside one long video, not across a corpus.
"""

import logging
from collections import defaultdict
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from encoders.base import DualEncoderArtifact
from stats import wilson_interval

logger = logging.getLogger(__name__)

#: Chunk-side representations. `fused` is a plain renormalized mean of the two towers --
#: the untrained stand-in for the fusion adapters, not the adapters themselves.
REPRESENTATIONS = ("video", "transcript", "fused")

#: Query-side forms. The options are available at retrieval time in a multiple-choice
#: setting, but they carry content the bare question does not, so both are reported.
QUERIES = {"question": "query_q", "question+options": "query_qa"}


def select_chunk_embeddings(
    tensors: dict[str, torch.Tensor], representation: Literal[*REPRESENTATIONS]
) -> torch.Tensor:
    """Pick the per-chunk embedding matrix for a given representation.

    Args:
        tensors: A `DualEncoderArtifact`'s tensors, holding ``video_embeddings`` and
            ``text_embeddings``, each shaped ``(num_chunks, dim)``.
        representation: Which embedding to return:
            - ``"video"``: the video-encoder embeddings, unchanged.
            - ``"transcript"``: the text-encoder embeddings, unchanged.
            - ``"fused"``: the elementwise sum of the two, L2-normalized
              along the last dimension. Note that ``"video"`` and
              ``"transcript"`` are returned as-is, so only ``"fused"`` is
              guaranteed to be unit-norm.

    Returns:
        A ``(num_chunks, dim)`` tensor of chunk embeddings.

    Raises:
        ValueError: If ``representation`` is not one of the three options.
    """
    if representation == "video":
        return tensors["video_embeddings"]
    if representation == "transcript":
        return tensors["text_embeddings"]
    if representation == "fused":
        return F.normalize(tensors["video_embeddings"] + tensors["text_embeddings"], dim=-1)
    raise ValueError(f"unknown representation {representation!r}")


def rows_by_video(chunk_ids: list[tuple[str, int]]) -> dict[str, list[int]]:
    """Row indices into the chunk tensors, per video, in chunk order."""
    rows = defaultdict(list)
    for row, (vid, _) in enumerate(chunk_ids):
        rows[vid].append(row)
    return rows


def oracle_ranks(
    enc_artifact: DualEncoderArtifact,
    representation: Literal[*REPRESENTATIONS],
    query: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Where the oracle lands for each question, and how many chunks it competed against.

    Returns `(rank, pool)` per question; rank is 0-based, so 0 is a top-1 hit. Ties are
    resolved pessimistically -- the oracle is credited only with the rank it would get if
    every chunk scoring at least as high came first -- because a tie is not a hit and
    `argmax` would quietly award it one.
    """
    tensors, meta = enc_artifact["tensors"], enc_artifact["meta"]
    chunks = select_chunk_embeddings(tensors, representation)
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


def select_chunks_for_questions(
    enc_artifact: DualEncoderArtifact,
    selection_type: Literal["oracle", "top1", "random"],
    representation: Literal[*REPRESENTATIONS] = "fused",
    query: str = "question+options",
    seed: int = 0,
) -> list[tuple[str, str, int]]:
    """
    Select which video chunk each question in the dataset should be answered from.

    In the SIQ2Long pipeline, full videos are ingested and split into 1-minute chunks.
    Then, for each question, one chunk is selected from its corresponding video,
    and that chunk is passed to the VLM for answering. 
    
    A few selection options:
    1. "oracle": always choose the original SIQ2.0 video trim.
    2. "top1": choose the chunk whose embeddings are most similar to the question's,
                given a specified representation (video, transcript, or fused).
    3. "random": choose a chunk at random from the video. 
                This serves as a baseline to compare against the other selection methods.

    A positional prior is deliberately absent. The obvious form -- predict the modal
    oracle index, clamped to videos with fewer chunks -- degenerates into "the last
    chunk" exactly on the short videos where guessing is easiest, so it flatters itself.
    A principled version would fall back down the empirical distribution (mode, then
    next most common that exists) rather than clamping; until then the positional
    baseline stays in the retrieval table, where it is computed honestly.

    Args:
        enc_artifact: The artifact returned by `retrieval.encode`, containing the
            embeddings and metadata for all videos and questions.
        selection_type: One of "oracle", "top1", or "random", determining how to select
            the chunk for each question.
        representation: Which chunk representation to use when selecting the top1 chunk.
            Must be one of "video", "transcript", or "fused".
        query: Which query representation to use when selecting the top1 chunk. Must be
            one of "question" or "question+options".
        seed: Random seed for reproducibility when using random selection.
    """
    meta, tensors = enc_artifact["meta"], enc_artifact["tensors"]
    questions = list(zip(meta["qids"], meta["query_vid"], strict=True))

    if selection_type == "oracle":
        return [(qid, vid, meta["oracle_idx"][vid]) for qid, vid in questions]

    rows_per_video = rows_by_video(meta["chunk_ids"])

    if selection_type == "top1":
        chunks = select_chunk_embeddings(tensors, representation)
        queries = tensors[QUERIES[query]]
        return [
            (qid, vid, int((queries[qi] @ chunks[rows_per_video[vid]].T).argmax()))
            for qi, (qid, vid) in enumerate(questions)
        ]

    if selection_type == "random":
        rng = np.random.default_rng(seed)
        return [
            (qid, vid, int(rng.integers(len(rows_per_video[vid])))) for qid, vid in questions
        ]

    raise ValueError(f"unknown selection_type {selection_type!r}")


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
