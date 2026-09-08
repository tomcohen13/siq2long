"""
Text-only retrieval baselines over chunk transcripts: BM25 and BGE.
"""

import logging
import re

import numpy as np
import torch
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer

from encoders.base import DualEncoderArtifact, best_device
from scoring import POOL_TYPES, question_pools, rank_of

logger = logging.getLogger(__name__)

#: BGE v1.5 asks for an instruction on the query side only; passages are embedded bare.
BGE_CHECKPOINT = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_WORD = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Deliberately plain -- BM25 is the floor, not a tuned system."""
    return _WORD.findall(text.lower())


def bm25_ranks(
    artifact: DualEncoderArtifact,
    queries: list[str],
    pool_type: str = "within",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Where the oracle lands under BM25 over chunk transcripts.

    IDF is fitted on **every** chunk in the cache, not per video: term statistics over the
    six documents of one video are meaningless, and a corpus-level fit is what BM25
    normally gets. Only the pool's chunks are then ranked.

    A chunk whose transcript is empty scores 0. When every chunk in a pool scores 0 the
    pessimistic tie-break puts the oracle last, which is the honest reading -- BM25 made
    no choice there.

    Returns:
        `(ranks, pool_sizes)`, same shape and meaning as `scoring.oracle_ranks`.
    """
    meta = artifact["meta"]
    bm25 = BM25Okapi([tokenize(t) for t in meta["chunk_texts"]])
    logger.info("BM25 over %d chunk transcripts", len(meta["chunk_texts"]))

    ranks, pool_sizes = [], []
    for qi, (rows, oracle) in enumerate(question_pools(meta, pool_type, seed)):
        scores = bm25.get_scores(tokenize(queries[qi]))[rows]
        ranks.append(rank_of(scores, scores[rows.index(oracle)]))
        pool_sizes.append(len(rows))
    return np.array(ranks), np.array(pool_sizes)


@torch.inference_mode()
def bge_embed(texts: list[str], tokenizer, model, batch_size: int = 64) -> torch.Tensor:
    """CLS-pooled, L2-normalized BGE embeddings -- the pooling its checkpoint was trained for."""
    out = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[i : i + batch_size],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(model.device)
        cls = model(**batch).last_hidden_state[:, 0]
        out.append(F.normalize(cls, dim=-1).cpu())
    return torch.cat(out)


def bge_ranks(
    artifact: DualEncoderArtifact,
    queries: list[str],
    pool_type: str = "within",
    seed: int = 0,
    checkpoint: str = BGE_CHECKPOINT,
    device: str | None = None,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Where the oracle lands under BGE dense-text retrieval over chunk transcripts.

    The rung above BM25: a text retriever that is actually good at retrieval, given the
    same transcripts. It reads 512 tokens against X-CLIP's 77, so if the transcript tower
    were context-limited this is where that would show -- §1b already argues it is not.

    Returns:
        `(ranks, pool_sizes)`, same shape and meaning as `scoring.oracle_ranks`.
    """
    meta = artifact["meta"]
    device = device or best_device()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint).to(device).eval()
    logger.info("BGE %s on %s over %d chunks", checkpoint, device, len(meta["chunk_texts"]))

    chunk_vecs = bge_embed(meta["chunk_texts"], tokenizer, model, batch_size)
    query_vecs = bge_embed([BGE_QUERY_PREFIX + q for q in queries], tokenizer, model, batch_size)

    ranks, pool_sizes = [], []
    for qi, (rows, oracle) in enumerate(question_pools(meta, pool_type, seed)):
        scores = query_vecs[qi] @ chunk_vecs[rows].T
        ranks.append(rank_of(scores, scores[rows.index(oracle)]))
        pool_sizes.append(len(rows))
    return np.array(ranks), np.array(pool_sizes)


#: Name -> ranking function, for scripts that sweep the text baselines.
TEXT_BASELINES = {"bm25": bm25_ranks, "bge": bge_ranks}

__all__ = [
    "BGE_CHECKPOINT",
    "POOL_TYPES",
    "TEXT_BASELINES",
    "bge_ranks",
    "bm25_ranks",
    "tokenize",
]
