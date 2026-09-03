#!/usr/bin/env python
"""
Score the text-only retrieval baselines against an embedding cache.

Rungs 3 and 4 of the ladder in FINDINGS section 4: BM25 (lexical) and BGE (dense text),
both over the chunk transcripts already persisted in the cache. No video is decoded and
the cache's own encoder is never loaded -- only its `chunk_texts` are read -- so this is
seconds for BM25 and a couple of minutes for BGE on CPU.

    python scripts/eval_baselines.py --cache datasets/siq2long/embeddings/xclip_val.pt
    python scripts/eval_baselines.py --cache .../xclip_val.pt --baseline bm25 --pool-type both

Queries are rebuilt from the QA rows, because the cache stores their embeddings and not
their text. Pass the dataset the cache was built from.

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

from baselines import TEXT_BASELINES, render_queries  # noqa: E402
from config import Datasets  # noqa: E402
from data.load import load_qa  # noqa: E402
from encoders.base import DualEncoder  # noqa: E402
from logs import banner, setup_logging  # noqa: E402
from scoring import POOL_TYPES, QUERIES, metrics  # noqa: E402

log = logging.getLogger("eval_baselines")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cache", type=Path, required=True, help="a .pt written by encode_chunks")
    p.add_argument("--dataset", default=Datasets.SIQ2LONG, choices=[d.value for d in Datasets],
                   help="where to read the question text from")
    p.add_argument("--split", help="default: the split recorded in the cache")
    p.add_argument("--baseline", default="both", choices=[*TEXT_BASELINES, "both"])
    p.add_argument("--query", default="both", choices=[*QUERIES, "both"])
    p.add_argument("--pool-type", default="within", choices=[*POOL_TYPES, "both"])
    p.add_argument("--seed", type=int, default=0, help="cross-video sampling seed")
    p.add_argument("--device", help="BGE device; default is the best available")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(level=args.log_level)

    bundle = DualEncoder.load(args.cache)
    meta = bundle["meta"]
    split = args.split or meta.get("split")
    if not split:
        log.error("the cache records no split; pass --split")
        return 1

    banner(log, "eval_baselines", {
        "cache": args.cache,
        "dataset": f"{args.dataset} ({split})",
        "chunks": len(meta["chunk_texts"]),
        "questions": len(meta["qids"]),
        "baseline": args.baseline,
        "pool": args.pool_type,
    })

    qa = load_qa(split, args.dataset)
    names = TEXT_BASELINES if args.baseline == "both" else {args.baseline: TEXT_BASELINES[args.baseline]}
    query_forms = QUERIES if args.query == "both" else {args.query: QUERIES[args.query]}
    pool_types = POOL_TYPES if args.pool_type == "both" else (args.pool_type,)

    log.info("%-6s %-16s %-8s %7s %7s %7s %8s",
             "model", "query", "pool", "top1", "top3", "mrr", "random")
    for name, rank_fn in names.items():
        for query in query_forms:
            queries = render_queries(bundle, qa, with_options=query == "question+options")
            for pool_type in pool_types:
                kwargs = {"device": args.device} if name == "bge" else {}
                ranks, pools = rank_fn(bundle, queries, pool_type, args.seed, **kwargs)
                m = metrics(ranks, pools)
                log.info("%-6s %-16s %-8s %6.1f%% %6.1f%% %7.3f %7.1f%%",
                         name, query, pool_type,
                         100 * m["top1"], 100 * m["top3"], m["mrr"], 100 * m["random"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
