#!/usr/bin/env python
"""
Score an embedding cache and plot top-1 against the number of chunks competed against.

Reads only the cache `encode_chunks.py` wrote, so this is seconds, not a re-encode. Each
question is ranked against its own video's chunks.

    python scripts/eval_retrieval.py --cache .../embeddings/xclip_val.pt
    python scripts/eval_retrieval.py --cache .../xclip_val.pt --no-plot

Requires `src` on the import path: `uv pip install -e .`, or PYTHONPATH=src.
"""

import argparse
import logging
import sys
from pathlib import Path

# Locate src/ relative to this file, so this works from any cwd.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from encoders.base import DualEncoder  # noqa: E402
from logs import banner, setup_logging  # noqa: E402
from plots import plot_top1_by_pool_size  # noqa: E402
from scoring import QUERIES, REPRESENTATIONS, by_pool_size, metrics, oracle_ranks  # noqa: E402

#: Chunk-count bands. Chosen for support on val (255/125/132/155/111 questions), not for
#: round numbers -- a band with twenty questions in it says nothing.
BANDS = [(2, 3), (4, 5), (6, 8), (9, 12), (13, 20)]

log = logging.getLogger("eval_retrieval")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cache", type=Path, required=True, help="a .pt written by encode_chunks")
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(level=args.log_level)

    bundle = DualEncoder.load(args.cache)
    meta = bundle["meta"]
    banner(
        log,
        "eval_retrieval",
        {
            "cache": args.cache,
            "checkpoint": bundle["checkpoint"],
            "split": meta.get("split", "?"),
            "questions": len(meta["qids"]),
            "videos scored": len(set(meta["query_vid"])),
        },
    )

    results = {}
    log.info("%-12s %-16s %7s %7s %7s %8s", "chunks", "query", "top1", "top3", "mrr", "random")
    for rep in REPRESENTATIONS:
        for query in QUERIES:
            ranks, pools = oracle_ranks(bundle, rep, query)
            m = metrics(ranks, pools)
            results[(rep, query)] = (ranks, pools)
            log.info(
                "%-12s %-16s %6.1f%% %6.1f%% %7.3f %7.1f%%",
                rep,
                query,
                100 * m["top1"],
                100 * m["top3"],
                m["mrr"],
                100 * m["random"],
            )

    if args.no_plot:
        return 0

    strata = {k: by_pool_size(r, p, BANDS) for k, (r, p) in results.items()}
    name = args.cache.stem
    path = plot_top1_by_pool_size(strata, args.out_dir / f"{name}_top1_by_pool.png", title=name)
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
