#!/usr/bin/env python
"""
Build the SIQ2-Long chunk manifest.

Chunks every downloaded video around its SIQ2 oracle window and writes one JSON entry per
video: its duration, its [start, end] chunk boundaries, and the index of the oracle chunk
(the retrieval label). No video is cut on disk -- the manifest *is* the dataset.

    python scripts/build_manifest.py
    python scripts/build_manifest.py --max-chunks 20 --out /tmp/chunks.json

ffprobe plus arithmetic, so it runs on a laptop in seconds and imports no torch.
Requires `src` on the import path: `uv pip install -e .`, or PYTHONPATH=src.
"""

import argparse
import logging
import sys
from pathlib import Path

# Locate src/ relative to this file, so `python scripts/build_manifest.py` works from any
# cwd with no editable install and no PYTHONPATH.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import DATASET_TO_DIR, Columns, Datasets  # noqa: E402
from data.load import load_qa  # noqa: E402
from data.manifest import MAX_CHUNKS, MIN_CHUNKS, build_manifest, write_manifest  # noqa: E402
from logs import banner, setup_logging  # noqa: E402

log = logging.getLogger("build_manifest")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset-dir", type=Path, default=DATASET_TO_DIR[Datasets.SIQ2LONG])
    p.add_argument("--trims", type=Path, help="default: <dataset-dir>/trims.json")
    p.add_argument("--out", type=Path, help="default: <dataset-dir>/video_chunks.json")
    p.add_argument("--max-chunks", type=int, default=MAX_CHUNKS)
    p.add_argument("--min-chunks", type=int, default=MIN_CHUNKS)
    p.add_argument("--chunk-size", type=int, default=60, help="seconds per chunk")
    p.add_argument("--buffer-size", type=int, default=10, help="edge sliver absorbed, seconds")
    p.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    p.add_argument("--log-file", type=Path)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def report_coverage(manifest: dict) -> None:
    """How many QA rows survive per split -- the shrinkage that makes tables disagree."""
    for split in ("train", "val", "test"):
        try:
            qa = load_qa(split, Datasets.SIQ2LONG)
        except Exception:
            log.warning("could not load the %s split; skipping its coverage", split)
            continue
        covered = qa[Columns.VIDEO_ID].isin(manifest).sum()
        log.info(
            "%-5s : %5d / %5d questions (%4.1f%%) over %d videos",
            split,
            covered,
            len(qa),
            100 * covered / max(len(qa), 1),
            qa.loc[qa[Columns.VIDEO_ID].isin(manifest), Columns.VIDEO_ID].nunique(),
        )


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_file, args.log_level)
    out_path = args.out or args.dataset_dir / "video_chunks.json"
    trims_path = args.trims or args.dataset_dir / "trims.json"

    banner(
        log,
        "build_manifest",
        {
            "dataset_dir": args.dataset_dir,
            "trims": trims_path,
            "out": out_path,
            "chunks": f"{args.min_chunks}..{args.max_chunks} of {args.chunk_size}s",
            "buffer": f"{args.buffer_size}s",
        },
    )

    if out_path.exists() and not args.force:
        log.error("%s exists; pass --force to overwrite it", out_path)
        return 1

    try:
        manifest, stats = build_manifest(
            dataset_dir=args.dataset_dir,
            trims_path=trims_path,
            max_chunks=args.max_chunks,
            min_chunks=args.min_chunks,
            chunk_size=args.chunk_size,
            buffer_size=args.buffer_size,
        )
    except KeyboardInterrupt:
        log.warning("interrupted; nothing written")
        return 130
    except Exception:
        log.exception("manifest build failed")
        return 1

    stats.log()
    if not manifest:
        log.error("manifest is empty; refusing to write")
        return 1

    report_coverage(manifest)
    write_manifest(manifest, out_path)
    log.info("wrote %d videos to %s", len(manifest), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
