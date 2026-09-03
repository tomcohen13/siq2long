#!/usr/bin/env python
"""
Encode a split's chunks and questions into one embedding cache.

Runs the expensive half of oracle find: decode every chunk's frames, encode them and the
chunk transcript, encode every question with and without its answer options, and write
the lot to a single file stamped with the checkpoint that produced it. Scoring reads only
that file, so this runs once per encoder and evaluation runs as often as you like.

    python scripts/encode_chunks.py --encoder xclip --split val
    python scripts/encode_chunks.py --encoder pe-video --split val --limit 20

Requires `src` on the import path: `uv pip install -e .`, or PYTHONPATH=src.
"""

import argparse
import logging
import sys
from pathlib import Path

# Locate src/ relative to this file, so `python scripts/encode_chunks.py` works from any
# cwd with no editable install and no PYTHONPATH.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import DATASET_TO_DIR, Columns, Datasets  # noqa: E402
from data.load import find_downloaded_files, load_qa  # noqa: E402
from data.manifest import load_manifest  # noqa: E402
from encoders import PEVideoEncoder, XCLIPEncoder  # noqa: E402
from encoders.base import best_device  # noqa: E402
from logs import banner, setup_logging  # noqa: E402
from retrieval import encode  # noqa: E402

ENCODERS = {"xclip": XCLIPEncoder, "pe-video": PEVideoEncoder}

log = logging.getLogger("encode_chunks")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--encoder", required=True, choices=sorted(ENCODERS))
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--dataset-dir", type=Path, default=DATASET_TO_DIR[Datasets.SIQ2LONG])
    p.add_argument("--manifest", type=Path, help="default: <dataset-dir>/video_chunks.json")
    p.add_argument("--out", type=Path, help="default: <dataset-dir>/embeddings/<encoder>_<split>.pt")
    p.add_argument("--limit", type=int, help="first N videos only, for smoke tests")
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], help="default: best available")
    p.add_argument("--batch-size", type=int, help="override the encoder's default")
    p.add_argument("--force", action="store_true", help="overwrite an existing cache")
    p.add_argument("--log-file", type=Path)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_file, args.log_level)

    manifest_path = args.manifest or args.dataset_dir / "video_chunks.json"
    out_path = args.out or args.dataset_dir / "embeddings" / f"{args.encoder}_{args.split}.pt"

    banner(
        log,
        "encode_chunks",
        {
            "encoder": args.encoder,
            "split": args.split,
            "manifest": manifest_path,
            "out": out_path,
            "limit": args.limit or "all",
        },
    )

    if out_path.exists() and not args.force:
        log.error("%s exists; pass --force to overwrite it", out_path)
        return 1

    try:
        qa = load_qa(split=args.split, dataset=Datasets.SIQ2LONG)
        # Chunks belong to videos, not to splits, so the manifest covers every video in
        # the dataset. Narrow it to this split here -- `encode` filters only the questions,
        # so an unnarrowed manifest encodes all 748 videos to answer one split's 778.
        split_vids = set(qa[Columns.VIDEO_ID])
        manifest = {v: c for v, c in load_manifest(manifest_path).items() if v in split_vids}
        files = find_downloaded_files(args.dataset_dir, to_dataframe=True)
        log.info("%s covers %d of the manifest's videos", args.split, len(manifest))

        encoder = ENCODERS[args.encoder]().to(args.device or best_device())
        encoder.eval()
        if args.batch_size:
            encoder.batch_size = args.batch_size
        log.info("loaded %s on %s", encoder.checkpoint, encoder.device)
        tensors, meta = encode(encoder, manifest, qa, files, limit=args.limit)
    except KeyboardInterrupt:
        log.warning("interrupted; nothing written")
        return 130
    except Exception:
        log.exception("encoding failed")
        return 1

    meta["split"] = args.split
    encoder.save(out_path, meta=meta, **tensors)

    log.info(
        "wrote %d chunks over %d videos and %d questions to %s",
        len(meta["chunk_ids"]),
        len(meta["oracle_idx"]),
        len(meta["qids"]),
        out_path,
    )
    if meta["failed"]:
        log.warning("%d videos failed to encode, e.g. %s", len(meta["failed"]), meta["failed"][:5])
    return 0


if __name__ == "__main__":
    sys.exit(main())