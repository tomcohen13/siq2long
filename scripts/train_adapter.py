#!/usr/bin/env python
"""
Train an adapter on oracle find and save it.

Reads a cache written by encode_chunks, fits a small head on top of the frozen embeddings,
and early-stops on the dev videos the split wrote into the cache. No video is decoded and
the backbone never runs, so this is minutes on a laptop and seconds on a GPU.

    python scripts/train_adapter.py --cache siq2long/embeddings/xclip_train.pt
    python scripts/train_adapter.py --cache siq2long/embeddings/xclip_train.pt \\
        --adapter big --query-form question+options --answer-weight 0.5

Saves the weights, the config that produced them, and the per-epoch history to one file, so
a run can be scored later without retraining.

Requires `src` on the import path: `uv pip install -e .`, or PYTHONPATH=src.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Locate src/ relative to this file, so this works from any cwd.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from adapters import ADAPTERS, BigAdapter, SmallAdapter  # noqa: E402
from encoders.base import DualEncoder, best_device  # noqa: E402
from logs import banner, setup_logging  # noqa: E402
from scoring import QUERY_TENSORS  # noqa: E402
from train import train  # noqa: E402

log = logging.getLogger("train_adapter")


def build_adapter(args, dim: int, checkpoint: str):
    """The adapter named on the command line, sized to the cache it will read."""
    shared = dict(dim=dim, freeze_chunks=args.freeze_chunks, checkpoint=checkpoint)
    if args.adapter == "small":
        return SmallAdapter(rank=args.rank, dropout=args.dropout, **shared)
    return BigAdapter(hidden=args.hidden, dropout=args.dropout, **shared)


def run_name(args) -> str:
    """A filename that says what the run was, since we will have several side by side."""
    parts = [args.adapter, args.query_form.replace("+", "-")]
    if args.freeze_chunks:
        parts.append("queryonly")
    if args.answer_weight:
        parts.append(f"ans{args.answer_weight:g}")
    return "_".join(parts)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cache", type=Path, required=True, help="the train cache to fit on")
    p.add_argument("--adapter", default="small", choices=sorted(ADAPTERS))
    p.add_argument("--query-form", default="question", choices=list(QUERY_TENSORS),
                   help="what the adapter learns to search with")

    p.add_argument("--rank", type=int, default=8, help="small adapter: bottleneck width")
    p.add_argument("--hidden", type=int, default=2048, help="big adapter: hidden width")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--freeze-chunks", action="store_true",
                   help="train the query side only, leaving chunks exactly as cached")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--n-easy", type=int, default=64,
                   help="chunks drawn from other videos per pool; 0 for same-video only. "
                        "Raise it on a GPU; 64 is what a laptop handles")
    p.add_argument("--answer-weight", type=float, default=0.0,
                   help="weight on the gold answer searching the same pool; 0 drops it")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], help="default: best available")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/adapters"))
    p.add_argument("--log-file", type=Path)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_file, args.log_level)

    device = args.device or best_device()
    out_path = args.out_dir / f"{run_name(args)}.pt"

    banner(log, f"train_adapter: {run_name(args)}", {
        "cache": args.cache,
        "adapter": args.adapter,
        "query": args.query_form,
        "chunks": "frozen" if args.freeze_chunks else "trained",
        "easy negatives": args.n_easy,
        "answer weight": args.answer_weight,
        "device": device,
        "out": out_path,
    })

    try:
        artifact = DualEncoder.load(args.cache)
        dim = artifact["tensors"]["video_embeddings"].shape[-1]
        adapter = build_adapter(args, dim, artifact["checkpoint"])

        adapter, history = train(
            adapter,
            artifact,
            query_form=args.query_form,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            n_easy=args.n_easy,
            answer_weight=args.answer_weight,
            patience=args.patience,
            seed=args.seed,
            device=device,
        )
    except KeyboardInterrupt:
        log.warning("interrupted; nothing written")
        return 130
    except Exception:
        log.exception("training failed")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": adapter.state_dict(), "config": vars(args) | {"device": device},
         "history": history},
        out_path,
    )
    log.info("wrote %s", out_path)

    best = max(history, key=lambda record: record["top1"])
    log.info(
        "dev top-1 %.1f%% at epoch %d, from %.1f%% frozen, against a %.1f%% random floor",
        100 * best["top1"], best["epoch"], 100 * history[0]["top1"], 100 * best["random"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
