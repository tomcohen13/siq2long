#!/usr/bin/env python
"""
Run one (model, dataset, split, condition) evaluation.

Every run writes two artifacts next to each other: a JSONL of per-question
results and a log of how it was produced, including the git commit. Re-running
the same command resumes from the JSONL, so a preempted instance costs only the
batch in flight.

    python scripts/run_eval.py --model qwen3-vl --split val --limit 12
    python scripts/run_eval.py --model internvl3 --split val --no-transcript

Requires `src` on the import path: `uv pip install -e .`, or PYTHONPATH=src.
"""

import argparse
import inspect
import logging
import subprocess
import sys
from pathlib import Path

from config import DATASET_TO_DIR, Columns, Datasets
from data.load import find_downloaded_files, load_qa
from data.transcripts import load_transcripts
from inference import run
from logs import banner, setup_logging
from vlm import InternVL3_8B, Qwen2_5VL, Qwen3VL, VideoLlama3

MODELS = {
    "qwen2.5-vl": Qwen2_5VL,
    "qwen3-vl": Qwen3VL,
    "videollama3": VideoLlama3,
    "internvl3": InternVL3_8B,
}

log = logging.getLogger("run_eval")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--dataset", default=Datasets.SIQ2, choices=[d.value for d in Datasets])
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--no-transcript", action="store_true", help="video only, no transcript in the prompt")
    p.add_argument("--limit", type=int, help="first N questions only, for smoke tests")

    p.add_argument("--out", type=Path, help="results JSONL (default: outputs/<run>.jsonl)")
    p.add_argument("--log-file", type=Path, help="log file (default: alongside --out)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--no-resume", action="store_true", help="ignore an existing results file")

    # Forwarded to the backend only if its __init__ accepts them; anything dropped
    # is logged, since a silently ignored --fps would confound the comparison.
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--max-frames", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--max-batch-size", type=int)
    return p.parse_args(argv)


def git_revision() -> str:
    """Short SHA, marked dirty when the tree has uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run_name(args) -> str:
    condition = "notx" if args.no_transcript else "tx"
    stem = f"{args.model}_{args.dataset}_{args.split}_{condition}"
    return f"{stem}_n{args.limit}" if args.limit else stem


def load_rows(dataset: str, split: str, limit: int | None):
    """QA rows joined to local media paths, plus the transcript per video."""
    qa = load_qa(split, dataset)
    files = find_downloaded_files(DATASET_TO_DIR[dataset], to_dataframe=True)

    rows = qa.merge(files, on=Columns.VIDEO_ID, how="inner")
    missing = sorted(set(qa[Columns.VIDEO_ID]) - set(files[Columns.VIDEO_ID]))
    log.info(
        "%d/%d questions have local media, across %d videos",
        len(rows), len(qa), rows[Columns.VIDEO_ID].nunique(),
    )
    if missing:
        log.warning("%d videos absent locally, e.g. %s", len(missing), missing[:5])
    if rows.empty:
        raise SystemExit(f"no questions left after joining {split} to media in {DATASET_TO_DIR[dataset]}")

    if limit:
        rows = rows.head(limit)
        log.info("limited to the first %d questions", len(rows))

    transcripts = load_transcripts(files.to_dict(orient="records"))
    log.info("loaded %d transcripts", len(transcripts))
    return rows, transcripts


def build_model(args):
    """Instantiate the backend, forwarding only the kwargs it declares."""
    cls = MODELS[args.model]
    wanted = {
        "fps": args.fps,
        "max_frames": args.max_frames,
        "max_new_tokens": args.max_new_tokens,
        "max_batch_size": args.max_batch_size,
    }
    wanted = {k: v for k, v in wanted.items() if v is not None}

    accepted = inspect.signature(cls.__init__).parameters
    kwargs = {k: v for k, v in wanted.items() if k in accepted}
    for k in wanted.keys() - kwargs.keys():
        log.warning("%s does not accept --%s; leaving it at the backend default", args.model, k.replace("_", "-"))

    log.info("loading %s (%s)", cls.__name__, cls.CHECKPOINT)
    model = cls(**kwargs)
    log.info("loaded with attn_implementation=%s, batch cap %d", model.attn, model.max_batch_size)
    return model


def main(argv=None) -> int:
    args = parse_args(argv)

    name = run_name(args)
    out_path = args.out or Path("outputs") / f"{name}.jsonl"
    log_file = args.log_file or out_path.with_suffix(".log")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    setup_logging(log_file=log_file, level=args.log_level)
    banner(log, f"siq2long eval: {name}", {
        "model": args.model,
        "dataset": f"{args.dataset} ({args.split})",
        "transcript": "no" if args.no_transcript else "yes",
        "fps / frames": f"{args.fps} / {args.max_frames}",
        "results": out_path,
        "log": log_file,
        "revision": git_revision(),
        "resume": "no" if args.no_resume else "yes",
    })

    try:
        rows, transcripts = load_rows(args.dataset, args.split, args.limit)
        model = build_model(args)
        results = run(
            model,
            rows,
            transcripts,
            use_transcript=not args.no_transcript,
            out_path=out_path,
            resume=not args.no_resume,
        )
    except KeyboardInterrupt:
        log.warning("interrupted; %s holds everything completed so far", out_path)
        return 130
    except Exception:
        log.exception("run failed")
        return 1

    log.info("wrote %d results to %s", len(results), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
