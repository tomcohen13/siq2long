#!/usr/bin/env python
"""
Score one or more embedding caches on oracle find, print the tables, draw the figures.

The task: rank a set of encoded 60-second chunks by their similarity to an encoded
question, and record where the oracle chunk -- the one the question was written about --
ranked. Both sides are already in the cache/artifacts `encode_chunks.py` wrote,
so this is a few tensor operations rather than a re-encode, and a full run takes seconds.

    python scripts/eval_retrieval.py --cache embeddings/xclip_val.pt
    python scripts/eval_retrieval.py --cache embeddings/*.pt --pool-type both

With `--pool-type both`, each question is ranked twice: once against a "pool" of chunks from its own video, and once against chunks sampled from other videos.
The pool size (number of chunks compared against) is equal in both cases.
An encoder that does well on the second but badly on the first can likely pick out the right video
but cannot tell one minute of it from another.

Requires `src` on the import path: `uv pip install -e .`, or PYTHONPATH=src.
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Locate src/ relative to this file, so this works from any cwd.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from encoders.base import DualEncoder  # noqa: E402
from logs import banner, setup_logging  # noqa: E402
from plots import (  # noqa: E402
    plot_pool_by_condition,
    plot_pool_contrast,
    plot_pool_contrast_overlay,
    plot_top1_by_pool_size,
)
from scoring import (  # noqa: E402
    POOL_TYPES,
    QUERY_TENSORS,
    REPRESENTATIONS,
    by_pool_size,
    metrics,
    oracle_ranks,
)
from stats import mcnemar_test  # noqa: E402

#: Chunk-count bands for the per-band figure. Sized so every band has enough val questions
#: to mean something (255/125/132/155/111), which is why they aren't round numbers.
BANDS = [(2, 3), (4, 5), (6, 8), (9, 12), (13, 20)]

FIGURE_TITLE = "Finding the oracle among a video's own chunks vs chunks from other videos"

#: What each pool_type means, spelled out for table headers.
POOL_LABELS = {"within": "chunks from the same video", "cross": "chunks from other videos"}

log = logging.getLogger("eval_retrieval")


@dataclass(frozen=True)
class Setup:
    """One combination we score. Four choices, and we sweep all of them."""

    encoder: str          # which cache the embeddings came from: "xclip", "pe-video"
    pool_type: str        # which chunks it ranked against, see POOL_LABELS
    representation: str   # which chunk embedding: video, transcript, or the two fused
    query_form: str       # what the question was encoded as, see scoring.QUERY_TENSORS


@dataclass(frozen=True)
class Score:
    """How one Setup did."""

    top1: float
    top3: float
    mrr: float
    random_floor: float   # what guessing would get, given these pool sizes
    ranks: np.ndarray     # per question, 0-based; 0 means the oracle was ranked first
    pool_sizes: np.ndarray

    @property
    def hit(self) -> np.ndarray:
        """Per question: did we rank the oracle first? Needed for the paired tests."""
        return self.ranks == 0


def encoder_name(cache: Path) -> str:
    """Caches are named `<encoder>_<split>.pt`, and the encoder half labels the results."""
    return cache.stem.rsplit("_", 1)[0]


def score_cache(cache: Path, pool_types: tuple[str, ...], seed: int) -> dict[Setup, Score]:
    """Score every representation and query form in one cache, for each pool type."""
    artifact = DualEncoder.load(cache)
    meta = artifact["meta"]
    encoder = encoder_name(cache)

    banner(log, f"eval_retrieval: {encoder}", {
        "cache": cache,
        "checkpoint": artifact["checkpoint"],
        "split": meta.get("split", "?"),
        "questions": len(meta["qids"]),
        "chunks": len(meta["chunk_ids"]),
        "videos": len(set(meta["query_vid"])),
        "pool": "/".join(pool_types),
    })

    scores = {}
    for pool_type in pool_types:
        for representation in REPRESENTATIONS:
            for query_form in QUERY_TENSORS:
                ranks, pool_sizes = oracle_ranks(
                    artifact, representation, query_form, pool_type, seed
                )
                measured = metrics(ranks, pool_sizes)
                setup = Setup(encoder, pool_type, representation, query_form)
                scores[setup] = Score(
                    top1=measured["top1"],
                    top3=measured["top3"],
                    mrr=measured["mrr"],
                    random_floor=measured["random"],
                    ranks=ranks,
                    pool_sizes=pool_sizes,
                )
    return scores


def log_scores(scores: dict[Setup, Score]) -> None:
    """Print one table per encoder and pool type."""
    for encoder in unique(s.encoder for s in scores):
        for pool_type in unique(s.pool_type for s in scores if s.encoder == encoder):
            log.info("--- %s, ranked against %s ---", encoder, POOL_LABELS[pool_type])
            log.info("%-12s %-16s %7s %7s %7s %8s",
                     "chunks", "query", "top1", "top3", "mrr", "random")
            for setup, score in scores.items():
                if setup.encoder != encoder or setup.pool_type != pool_type:
                    continue
                log.info("%-12s %-16s %6.1f%% %6.1f%% %7.3f %7.1f%%",
                         setup.representation, setup.query_form,
                         100 * score.top1, 100 * score.top3, score.mrr,
                         100 * score.random_floor)


def log_within_vs_cross(scores: dict[Setup, Score]) -> None:
    """
    Print how much better `cross` did than `within`, and whether the difference is real.

    Both rank the same questions, so the test is paired.
    """
    log.info("--- cross minus within (exact McNemar, same questions) ---")
    log.info("%-10s %-12s %-16s %8s %10s %10s",
             "encoder", "chunks", "query", "delta", "discordant", "p")

    for setup, score in scores.items():
        if setup.pool_type != "cross":
            continue
        within = scores.get(Setup(setup.encoder, "within", setup.representation, setup.query_form))
        if within is None:
            continue
        test = mcnemar_test(score.hit, within.hit)
        log.info("%-10s %-12s %-16s %+7.1fpp %5d/%-4d %10.3g",
                 setup.encoder, setup.representation, setup.query_form,
                 100 * test["delta"], test["n10"], test["n01"], test["p"])


def write_band_figures(scores: dict[Setup, Score], out_dir: Path) -> None:
    """One figure per encoder and pool type: top-1 split by how many chunks were competing."""
    for encoder in unique(s.encoder for s in scores):
        for pool_type in unique(s.pool_type for s in scores if s.encoder == encoder):
            bands = {
                (setup.representation, setup.query_form):
                    by_pool_size(score.ranks, score.pool_sizes, BANDS)
                for setup, score in scores.items()
                if setup.encoder == encoder and setup.pool_type == pool_type
            }
            name = f"{encoder}_{pool_type}"
            path = plot_top1_by_pool_size(bands, out_dir / f"{name}_top1_by_pool.png", title=name)
            log.info("wrote %s", path)


def write_contrast_figures(scores: dict[Setup, Score], out_dir: Path) -> None:
    """Draw `within` against `cross`. Skipped for any encoder that was only scored on one."""
    encoders = [
        encoder for encoder in unique(s.encoder for s in scores)
        if {s.pool_type for s in scores if s.encoder == encoder} == set(POOL_TYPES)
    ]
    if not encoders:
        return

    def top1_per_setup(pool_type: str) -> dict[str, dict[tuple[str, str], float]]:
        return {
            encoder: {
                (s.representation, s.query_form): score.top1
                for s, score in scores.items()
                if s.encoder == encoder and s.pool_type == pool_type
            }
            for encoder in encoders
        }

    within, cross = top1_per_setup("within"), top1_per_setup("cross")
    floors = {
        encoder: next(sc.random_floor for s, sc in scores.items() if s.encoder == encoder)
        for encoder in encoders
    }

    path = plot_pool_contrast(within, cross, floors,
                              out_dir / "pool_contrast.png", title=FIGURE_TITLE)
    log.info("wrote %s", path)

    # The other two figures put every encoder on one shared axis, which needs a single
    # random floor. Pool sizes are matched, so the floors should already be identical --
    # if they aren't, the caches cover different videos and comparing them would mislead.
    if len({round(floor, 4) for floor in floors.values()}) > 1:
        log.warning("caches disagree on the random floor (%s), so they cover different "
                    "videos; skipping the shared-axis figures", floors)
        return

    shared_floor = next(iter(floors.values()))
    for plot, filename in ((plot_pool_contrast_overlay, "pool_contrast_overlay.png"),
                           (plot_pool_by_condition, "pool_by_condition.png")):
        path = plot(within, cross, shared_floor, out_dir / filename, title=FIGURE_TITLE)
        log.info("wrote %s", path)


def unique(values) -> list:
    """Distinct values, first-seen order, so tables come out in the order we scored."""
    return list(dict.fromkeys(values))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cache", type=Path, required=True, nargs="+",
                   help="one or more .pt written by encode_chunks")
    p.add_argument("--pool-type", default="within", choices=[*POOL_TYPES, "both"],
                   help="score against the video's own chunks (within), against chunks from "
                        "other videos (cross), or both so they can be compared")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for picking the cross-video distractors")
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    p.add_argument("--no-plot", action="store_true", help="print the tables, skip the figures")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(level=args.log_level)

    pool_types = POOL_TYPES if args.pool_type == "both" else (args.pool_type,)

    scores: dict[Setup, Score] = {}
    for cache in args.cache:
        scores |= score_cache(cache, pool_types, args.seed)

    log_scores(scores)
    if set(pool_types) == set(POOL_TYPES):
        log_within_vs_cross(scores)

    if not args.no_plot:
        write_band_figures(scores, args.out_dir)
        write_contrast_figures(scores, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
