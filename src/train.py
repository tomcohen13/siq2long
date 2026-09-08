"""
Train an adapter on oracle find, over a cached artifact.

A training example is a query and a pool of candidate chunks, exactly one of which is
the oracle.

The loss is InfoNCE with the negatives chosen as follows:

- **hard negatives**, other chunks from the same video as the oracle.
- **easy negatives**, chunks from other videos.
"""

import logging
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from adapters import DualEncoderAdapter, adapt_artifact
from encoders.base import DualEncoderArtifact
from scoring import QUERY_TENSORS, metrics, oracle_ranks, question_pools

logger = logging.getLogger(__name__)


def sample_pools(meta: dict, n_easy: int = 256, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a "pool" of candidate video chunks per question:
     - its oracle (positive)
     - all other chunks from its own video (hard negatives)
     - `n_easy` chunks sampled other videos (easy negatives)

    Args:
        meta: an artifact's `meta`.
        n_easy: how many chunks to draw from other videos, per question. Never from a video
            in `meta["dev_vids"]` -- these are training pools, so held-out videos stay out
            entirely, not even as negatives.
        seed: which draw to take. Vary it per epoch for fresh outside chunks.

    Returns:
        `(rows, mask)`. `rows` indexes the chunk tensors, with the oracle in column 0. `mask`
        marks which entries are real: videos differ in length, so shorter ones leave padding.
        Padding beats repeating a short video's few chunks, which would weight them more
        heavily than the rest.
    """
    if n_easy < 0:
        raise ValueError(f"n_easy must be non-negative, got {n_easy}")

    within = question_pools(meta, "within")
    dev_vids = set(meta.get("dev_vids", ()))
    cross = (
        question_pools(meta, "cross", seed=seed, pool_size=n_easy + 1, exclude_vids=dev_vids)
        if n_easy
        else None
    )

    width = max(len(own_rows) for own_rows, _ in within) + n_easy
    rows = np.zeros((len(within), width), dtype=np.int64)
    mask = np.zeros((len(within), width), dtype=bool)

    for i, (own_rows, oracle_row) in enumerate(within):
        hard = [row for row in own_rows if row != oracle_row]
        # question_pools puts the oracle first in a cross pool; the rest are the easy ones.
        easy = cross[i][0][1:] if cross else []

        pool = [oracle_row, *hard, *easy]
        rows[i, : len(pool)] = pool
        mask[i, : len(pool)] = True

    filled = mask.sum(axis=1)
    logger.info(
        "pools up to %d wide (1 oracle + up to %d hard + %d easy); %d..%d filled, median %d",
        width, width - n_easy - 1, n_easy, filled.min(), filled.max(), int(np.median(filled)),
    )
    return torch.from_numpy(rows), torch.from_numpy(mask)


def pool_logits(
    query_embeddings: torch.Tensor,
    candidates_embeddings: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Compute similarity of each query to its own pool of video chunks, scaled by temperature.

    Args:
        query_embeddings: `(batch, dim)` of queries.
        candidates_embeddings: `(batch, width, dim)` of candidate chunks.
        mask: `(batch, width)` of booleans, true where `rows` holds a real candidate.
            Pools differ in size, so the rest is padding and scores `-inf` to drop out of
            the softmax.
        temperature: scale the logits by this before softmax. Lower is sharper, higher is
            flatter.

    Returns:
        `(batch, width)` scores, ready for cross-entropy against a target of zeros.
    """
    logits = torch.einsum("bd,bwd->bw", query_embeddings, candidates_embeddings)
    return (logits / temperature).masked_fill(~mask, float("-inf"))


def pool_loss(
    query_logits: torch.Tensor,
    answer_logits: torch.Tensor | None = None,
    answer_weight: float = 0.0,
) -> torch.Tensor:
    """
    Cross-entropy over the pool: pick the oracle out of its candidates.

    The oracle is column 0 by construction, so the target is a column of zeros. Passing
    `answer_logits` adds a second term in which the gold answer searches the same pool, which
    pulls the oracle chunk toward the sentence describing it. The wrong options never enter
    the loss.

    Args:
        query_logits: `(batch, width)` from `pool_logits`, the question searching its pool.
        answer_logits: the same for the gold answer. None drops the term.
        answer_weight: how much that second term counts against the first.

    Returns:
        A scalar loss.
    """
    target = torch.zeros(len(query_logits), dtype=torch.long, device=query_logits.device)
    loss = F.cross_entropy(query_logits, target)
    if answer_logits is not None and answer_weight:
        loss = loss + answer_weight * F.cross_entropy(answer_logits, target)
    return loss


def evaluate(
    artifact: DualEncoderArtifact,
    query_form: str,
    vids: set[str] | None = None,
    pool_type: str = "within",
) -> dict[str, float]:
    """
    Score a cache on oracle find, over the questions of some videos.

    Takes a cache, not a model, so a frozen cache and one pushed through `adapt_artifact`
    are scored by the same call -- which is what makes a before-and-after comparison mean
    something. Ranking goes through `oracle_ranks` rather than being reimplemented here, so
    a dev number watched during training matches a number reported afterwards.

    Args:
        artifact: the cache to score.
        query_form: what to search with, a key of `scoring.QUERY_TENSORS`.
        vids: score only questions about these videos. None scores all of them. A `within`
            pool is the question's own video, so narrowing never mixes chunks across a split.
        pool_type: what the oracle competes against, see `scoring.POOL_TYPES`.

    Returns:
        Top-1, top-3, MRR, the random floor these pools imply, and the question count.
    """
    ranks, pools = oracle_ranks(artifact, "fused", query_form, pool_type)
    if vids is not None:
        keep = np.array([vid in vids for vid in artifact["meta"]["query_vid"]])
        ranks, pools = ranks[keep], pools[keep]
    return metrics(ranks, pools)


def train(
    adapter: DualEncoderAdapter,
    artifact: DualEncoderArtifact,
    query_form: str = "question",
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    n_easy: int = 256,
    answer_weight: float = 0.0,
    patience: int = 5,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[DualEncoderAdapter, list[dict]]:
    """
    Fit an adapter, early-stopping on the held-out dev videos.

    Dev videos come from `meta["dev_vids"]`, which the split wrote into the artifact. Fitting
    uses every other video. Dev is never trained on, and the test split is a different file
    entirely, so nothing here has seen it.

    Args:
        adapter: the model to fit, modified in place and returned on the CPU.
        artifact: the cache to fit on. Must carry `dev_vids` in its meta. Its tensors are
            moved to `device` in place and left there.
        query_form: what to search with, a key of `scoring.QUERY_TENSORS`.
        epochs: how many passes at most; early stopping usually ends it sooner.
        batch_size: questions per step.
        lr, weight_decay: for AdamW, over the trainable parameters only.
        n_easy: chunks drawn from other videos per pool, redrawn each epoch.
        answer_weight: how much the gold-answer term counts. 0 drops it.
        patience: stop after this many epochs with no new best dev top-1.
        seed: fixes the batch order and the per-epoch draw of outside chunks.
        device: where to put the tensors and the model.

    Returns:
        The adapter with its best-scoring weights restored, and one record per epoch holding
        the loss and that epoch's dev scores.
    """
    meta = artifact["meta"]
    if "dev_vids" not in meta:
        raise KeyError("artifact has no dev_vids; the split has not been written into it")

    if answer_weight and QUERY_TENSORS["answer"] not in artifact["tensors"]:
        raise KeyError("answer_weight is set but the cache holds no gold-answer queries")

    dev_vids = set(meta["dev_vids"])
    is_fit = np.array([vid not in dev_vids for vid in meta["query_vid"]])
    fit_index = torch.from_numpy(np.flatnonzero(is_fit))
    if not len(fit_index):
        raise ValueError("every question belongs to a dev video; nothing left to fit on")

    # Move the cache to the training device once, in place. Scoring reads the artifact and
    # the loop reads `tensors`, so they cannot end up disagreeing about where things live.
    artifact["tensors"] = {name: t.to(device) for name, t in artifact["tensors"].items()}
    tensors = artifact["tensors"]

    adapter = adapter.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in adapter.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)

    baseline = evaluate(adapt_artifact(adapter, artifact), query_form, dev_vids)
    logger.info(
        "%s on %d fit / %d dev questions, %d trainable parameters; dev top-1 starts at %.1f%% "
        "(random %.1f%%)",
        type(adapter).__name__, int(is_fit.sum()), len(is_fit) - int(is_fit.sum()),
        adapter.trainable_parameters(), 100 * baseline["top1"], 100 * baseline["random"],
    )

    history = [{"epoch": 0, "loss": float("nan"), **baseline}]
    best = {"top1": baseline["top1"], "epoch": 0, "state": deepcopy(adapter.state_dict())}

    for epoch in range(1, epochs + 1):
        # Redraw each epoch: only the outside chunks change.
        rows, mask = sample_pools(meta, n_easy, seed=seed + epoch)
        rows, mask = rows.to(device), mask.to(device)

        adapter.train()
        order = fit_index[torch.randperm(len(fit_index), generator=generator)]
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch_ids = order[start : start + batch_size].to(device)

            # Encode each chunk this batch needs exactly once -- the same chunk sits in many
            # of the pools. Re-encoding every step is unavoidable: the weights just moved,
            # and the chunk side only gets a gradient from inside this step's graph.
            used, positions = torch.unique(rows[batch_ids], return_inverse=True)
            chunks = adapter.encode_chunks(
                tensors["video_embeddings"][used], tensors["text_embeddings"][used]
            )
            batch_candidates = chunks[positions]
            batch_mask = mask[batch_ids]
            queries = adapter.encode_queries(tensors[QUERY_TENSORS[query_form]][batch_ids])
            # Read inside the loop: `temperature` is exp(log_temperature), a fresh graph node
            # each step. Hoisting it would reuse a node whose graph the last backward freed.
            queries_logits = pool_logits(
                queries, batch_candidates, batch_mask, adapter.temperature
            )

            answer_logits = None
            if answer_weight:
                answers = adapter.encode_queries(tensors[QUERY_TENSORS["answer"]][batch_ids])
                answer_logits = pool_logits(
                    answers, batch_candidates, batch_mask, adapter.temperature
                )

            loss = pool_loss(queries_logits, answer_logits, answer_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch_ids)

        scored = evaluate(adapt_artifact(adapter, artifact), query_form, dev_vids)
        record = {"epoch": epoch, "loss": total / len(order), **scored}
        history.append(record)
        logger.info(
            "epoch %2d  loss %.4f  dev top-1 %.1f%%  top-3 %.1f%%  mrr %.3f",
            epoch, record["loss"], 100 * scored["top1"], 100 * scored["top3"], scored["mrr"],
        )

        if scored["top1"] > best["top1"]:
            best = {"top1": scored["top1"], "epoch": epoch, "state": deepcopy(adapter.state_dict())}
        elif epoch - best["epoch"] >= patience:
            logger.info("no dev improvement in %d epochs; stopping", patience)
            break

    adapter.load_state_dict(best["state"])
    logger.info(
        "best dev top-1 %.1f%% at epoch %d, against %.1f%% frozen and a %.1f%% random floor",
        100 * best["top1"], best["epoch"], 100 * baseline["top1"], 100 * baseline["random"],
    )
    return adapter.cpu(), history
