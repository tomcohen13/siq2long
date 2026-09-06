#!/usr/bin/env python
"""
Run one (model, dataset, split, condition) evaluation.

Every run writes two artifacts next to each other: a JSONL of per-question
results and a log of how it was produced, including the git commit. Re-running
the same command resumes from the JSONL, so a preempted instance costs only the
batch in flight.

    python scripts/run_eval.py --model qwen3-vl --dataset siq2long --split val --limit 12
    python scripts/run_eval.py --model internvl3 --dataset siq2 --split val --no-transcript

Pass --embeddings to answer each question from one retrieved chunk instead of the whole
video. --condition oracle is the ceiling a perfect retriever reaches, top1 is what the
retriever returns, and the gap between them is what retrieval failure costs:

    python scripts/run_eval.py --model qwen3-vl --dataset siq2long --split val \
        --embeddings outputs/embeddings/xclip_val.pt --condition top1

Requires `src` on the import path: `uv pip install -e .`, or PYTHONPATH=src.
"""

import argparse
import inspect
import logging
import os
import subprocess
import sys
from pathlib import Path

from tqdm.auto import tqdm

# Locate src/ relative to this file, so `python scripts/run_eval.py` works from any
# cwd with no editable install and no PYTHONPATH. Notebook environments lose `%env`
# on a runtime restart, and this script is the one thing that must always start.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import DATASET_TO_MEDIA_DIR, Columns, Datasets  # noqa: E402
from data.load import find_downloaded_files, load_qa
from data.transcripts import load_transcripts
from data.videos import slice_video
from encoders.base import DualEncoder
from inference import run
from logs import banner, setup_logging
from scoring import select_chunks_for_questions
from vlm import InternVL3_8B, LlavaNextVideo, Qwen2_5VL, Qwen3VL, VideoLlama3

MODELS = {
    "qwen2.5-vl": Qwen2_5VL,
    "qwen3-vl": Qwen3VL,
    "videollama3": VideoLlama3,
    "internvl3": InternVL3_8B,
    "llava-next-video": LlavaNextVideo,
}

log = logging.getLogger("run_eval")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--dataset", required=True, choices=[d.value for d in Datasets])
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--no-transcript", action="store_true", help="video only, no transcript in the prompt")
    p.add_argument("--limit", type=int, help="first N questions only, for smoke tests")

    # Chunk mode: answer each question from one retrieved chunk instead of the whole video.
    p.add_argument("--embeddings", type=Path, help="embedding cache from encode_chunks.py")
    p.add_argument("--condition", default="oracle", choices=["oracle", "top1", "random"],
                   help="which chunk answers each question (with --embeddings)")
    p.add_argument("--chunks-dir", type=Path, default=Path("outputs/chunks"),
                   help="where cut video chunks are cached")

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
    p.add_argument("--num-frames", type=int,
                   help="fixed frame budget, for backends trained on one (llava-next-video)")
    p.add_argument("--question-first", action="store_true",
                   help="put question and transcript before the video placeholder "
                        "(LLaVA's card ordering); default is video first")
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


def environment() -> str:
    """
    Versions of everything that can change a result, for the log.

    The video decoder belongs here: which backend loads decides which frames the
    model sees, and it has already differed between machines on this project.
    """
    import torch
    import transformers

    parts = [f"torch {torch.__version__}", f"transformers {transformers.__version__}"]
    for name in ("torchcodec", "decord", "av"):
        try:
            parts.append(f"{name} {__import__(name).__version__}")
        except (ImportError, AttributeError):
            parts.append(f"{name} -")
    reader = os.environ.get("FORCE_QWENVL_VIDEO_READER", "unset")
    return f"{', '.join(parts)}, qwen reader={reader}"


def run_name(args) -> str:
    condition = "notx" if args.no_transcript else "tx"
    order = "_qfirst" if args.question_first else ""
    # The retrieval condition is part of the identity: oracle and top1 runs differ only in
    # which chunk answered, and sharing an output file would silently resume across them.
    chunks = f"_{args.condition}" if args.embeddings else ""
    stem = f"{args.model}_{args.dataset}_{args.split}_{condition}{order}{chunks}"
    return f"{stem}_n{args.limit}" if args.limit else stem


def load_chunk_rows(args):
    """
    QA rows answered from a single chunk each, with that chunk cut to its own file.

    The end-to-end half of the paper: instead of handing a model the whole video, hand it
    one 60s window and see what the answer costs. `--condition oracle` is the ceiling a
    perfect retriever reaches, `top1` is what the retriever actually returns, and the gap
    between them is the price of retrieval failure.

    Each row's `vid_name` becomes `"{vid}#{chunk}"`. That is what lets the rest of the
    pipeline stay untouched: `inference.run` groups by `vid_name` to decode each chunk once
    and looks up `transcripts[vid_name]`, so keying on the chunk gives correct grouping,
    the chunk's own transcript, and a results file that records which chunk answered.

    Chunks have to exist on disk because the VLM backends take a path and decode it whole,
    unlike the encoders, which read frame indices straight out of the source.
    """
    artifact = DualEncoder.load(args.embeddings)
    meta = artifact["meta"]
    picks = select_chunks_for_questions(
        artifact,
        selection_type=args.condition,
        representation="fused",
        query_form="question+options",
    )
    if args.limit:
        picks = picks[: args.limit]

    # Deduplicated before indexing: a repeated qid makes `.loc` return a DataFrame rather
    # than a row, `.to_dict()` then nests every field under it, and the first symptom is
    # `int(gold)` failing on a dict several hundred questions into the run.
    qa = load_qa(args.split, args.dataset).drop_duplicates(Columns.QID).set_index(Columns.QID)
    files = find_downloaded_files(DATASET_TO_MEDIA_DIR[args.dataset], to_dataframe=True)
    files = files.set_index(Columns.VIDEO_ID)
    texts = {tuple(cid): t for cid, t in zip(meta["chunk_ids"], meta["chunk_texts"])}

    log.info(
        "%s: %d questions over %d distinct chunks (cache: %s)",
        args.condition, len(picks), len({(v, i) for _, v, i in picks}), artifact["checkpoint"],
    )

    rows, transcripts = [], {}
    for qid, vid, i in tqdm(picks, desc="chunks", unit="q"):
        start, end = meta["chunks"][vid][i]
        chunk_id = f"{vid}#{i}"
        # Absolute, because qwen-vl-utils hands the path to its decoder as `file://<path>`.
        # A relative path there parses as a URI *host* -- "outputs/chunks/x.mp4" becomes
        # host "outputs", path "/chunks/x.mp4" -- and every clip fails to open.
        dest = (args.chunks_dir / f"{vid}_{i}.mp4").resolve()
        slice_video(files.loc[vid, Columns.VIDEO_PATH], start, end, dest)
        rows.append(
            qa.loc[qid].to_dict()
            | {Columns.QID: qid, Columns.VIDEO_ID: chunk_id, Columns.VIDEO_PATH: dest}
        )
        transcripts[chunk_id] = texts[(vid, i)]
    return rows, transcripts


def load_rows(dataset: str, split: str, limit: int | None):
    """QA rows joined to local media paths, plus the transcript per video."""
    qa = load_qa(split, dataset)
    files = find_downloaded_files(DATASET_TO_MEDIA_DIR[dataset], to_dataframe=True)

    rows = qa.merge(files, on=Columns.VIDEO_ID, how="inner")
    missing = sorted(set(qa[Columns.VIDEO_ID]) - set(files[Columns.VIDEO_ID]))
    log.info(
        "%d/%d questions have local media, across %d videos",
        len(rows), len(qa), rows[Columns.VIDEO_ID].nunique(),
    )
    if missing:
        log.warning("%d videos absent locally, e.g. %s", len(missing), missing[:5])
    if rows.empty:
        raise SystemExit(f"no questions left after joining {split} to media in {DATASET_TO_MEDIA_DIR[dataset]}")

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
        "num_frames": args.num_frames,
        "question_first": args.question_first or None,  # None so it's only forwarded when set
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
        "chunks": f"{args.condition} from {args.embeddings}" if args.embeddings else "whole video",
        "transcript": "no" if args.no_transcript else "yes",
        "fps / frames": f"{args.fps} / {args.max_frames}",
        "results": out_path,
        "log": log_file,
        "revision": git_revision(),
        "resume": "no" if args.no_resume else "yes",
        "environment": environment(),
    })

    try:
        if args.embeddings:
            rows, transcripts = load_chunk_rows(args)
        else:
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

    # Decode failures are per-video warnings, so a run where every clip failed still gets
    # here and would otherwise exit 0 over an empty file. That has already cost two
    # overnight runs -- once to a missing video reader, once to a relative chunk path.
    if rows and not results:
        log.error("answered 0 of %d questions: every video was skipped. Check the decoder "
                  "and the chunk paths in the warnings above.", len(rows))
        return 1

    log.info("wrote %d results to %s", len(results), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
