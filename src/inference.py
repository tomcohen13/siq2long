"""
Batched multiple-choice inference over SIQ2 videos, backend-agnostic.

Questions are grouped by video so each clip is decoded once and reused across
all of its questions -- decoding dominates wall-clock, so this is the only
optimization that matters here. Batch size comes from `vlm.max_batch_size`,
which lets VideoLLaMA3 (1 question/pass) and Qwen (8) share this loop.

Predictions are the first 0-3 digit in the completion. An unparseable
completion is recorded as None and counted, never guessed -- a coin-flip
fallback would inflate accuracy by ~25% of the unparsed rate.
"""

import gc
import json
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from config import Columns
from src.vlm.base import VLM

DIGIT_RE = re.compile(r"[0-3]")

RESULT_COLUMNS = ["video", "qid", "pred", "gold", "correct", "raw"]


def parse(text: str) -> int | None:
    """First 0-3 in the output, or None. None is recorded, never guessed."""
    m = DIGIT_RE.search(text)
    return int(m.group()) if m else None


def group_by_video(rows: pd.DataFrame | list[dict]) -> dict[str, list[dict]]:
    """{video_id: [row, ...]}, input order preserved."""
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict(orient="records")

    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r[Columns.VIDEO_ID], []).append(r)
    return groups


def summarize(results: list[dict] | pd.DataFrame, n_examples: int = 5) -> float | None:
    """Print n / accuracy / unparsed rate. Returns accuracy, or None if unlabeled."""
    if isinstance(results, pd.DataFrame):
        results = results.to_dict(orient="records")
    n = len(results)
    if not n:
        print("n=0")
        return None

    unparsed = [r for r in results if r["pred"] is None]
    scored = [r for r in results if r["correct"] is not None]
    acc = sum(r["correct"] for r in scored) / len(scored) if scored else None

    acc_str = f"{acc:.4f}" if acc is not None else "n/a (unlabeled)"
    print(f"n={n}  accuracy={acc_str}  unparsed={len(unparsed)} ({len(unparsed) / n:.2%})")
    for r in unparsed[:n_examples]:
        print(f"  {r['qid']}: {r['raw']!r}")
    return acc


def _flush_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _done_qids(out_path: Path) -> set[str]:
    """qids already written to an interrupted run's JSONL."""
    if not out_path.exists():
        return set()
    with out_path.open() as f:
        return {json.loads(line)["qid"] for line in f if line.strip()}


def run(
    vlm: VLM,
    rows: pd.DataFrame | list[dict],
    transcripts: dict[str, str],
    *,
    use_transcript: bool = True,
    out_path: str | Path | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Generate an answer for every row and score it against `answer_idx`.

    Args:
        vlm: any VLM implementation; batch size is read from `vlm.max_batch_size`.
        rows: SIQ2 QA records with at least qid, vid_name, q, a0-a3, path_to_video,
            and answer_idx where labels exist (absent in the test split).
        transcripts: {video_id: transcript text}, e.g. from `load_transcripts`.
        use_transcript: pass the transcript to the model alongside the video.
        out_path: JSONL to append each batch to, so a crash mid-run loses nothing.
        resume: with `out_path`, skip qids already present in that file.

    Returns:
        DataFrame with columns video, qid, pred, gold, correct, raw. Videos that
        fail to decode are skipped with a printed reason and produce no rows.
    """
    groups = group_by_video(rows)

    out_path = Path(out_path) if out_path else None
    results: list[dict] = []
    if out_path and resume:
        skip = _done_qids(out_path)
        if skip:
            print(f"resuming: skipping {len(skip)} completed questions")
            groups = {
                vid: keep
                for vid, group in groups.items()
                if (keep := [r for r in group if r[Columns.QID] not in skip])
            }

    batch_size = max(1, vlm.max_batch_size)
    total = sum(len(g) for g in groups.values())
    progress = tqdm(total=total, unit="q")

    for video_id, group in groups.items():
        progress.set_description(video_id)
        try:
            clip = vlm.prepare_clip(
                str(group[0][Columns.VIDEO_PATH]),
                transcripts.get(video_id, ""),
            )
        except Exception as e:
            print(f"[skip] {video_id}: {type(e).__name__}: {e}")
            progress.update(len(group))
            continue

        try:
            for i in range(0, len(group), batch_size):
                batch = group[i : i + batch_size]
                texts = vlm.answer(clip, batch, use_transcript=use_transcript)

                batch_results = []
                for r, text in zip(batch, texts):
                    pred = parse(text)
                    gold = r.get(Columns.ANSWER_IDX)
                    gold = None if pd.isna(gold) else gold
                    batch_results.append({
                        "video": video_id,
                        "qid": r[Columns.QID],
                        "pred": pred,
                        "gold": gold,
                        "correct": None if gold is None else (pred is not None and pred == gold),
                        "raw": text.strip(),
                    })

                results.extend(batch_results)
                if out_path:
                    with out_path.open("a") as f:
                        for row in batch_results:
                            f.write(json.dumps(row) + "\n")

                progress.update(len(batch))
                scored = [r for r in results if r["correct"] is not None]
                if scored:
                    running = sum(r["correct"] for r in scored) / len(scored)
                    progress.set_postfix(acc=f"{running:.3f}")
        finally:
            del clip
            _flush_cuda()

    progress.close()
    summarize(results)
    return pd.DataFrame(results, columns=RESULT_COLUMNS)
