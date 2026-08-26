"""
Batched multiple-choice inference over SIQ2 videos, backend-agnostic.

Questions are grouped by video so each clip is decoded once and reused across
all of its questions -- decoding dominates wall-clock, so this is the only
optimization that matters here. Batch size comes from `vlm.max_batch_size`,
which lets VideoLLaMA3 (1 question/pass) and Qwen (8) share this loop.

Predictions are the first 0-3 digit in the completion. An unparseable
completion is recorded as None and counted, never guessed -- a coin-flip
fallback would inflate accuracy by ~25% of the unparsed rate.

Results are plain dicts, matching the JSONL written alongside them. Pandas
retypes None to NaN and int to np.int64, which makes "missing" mean two
different things in the same run; use `to_frame` when you want a DataFrame
for analysis.
"""

import gc
import json
import logging
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from config import Columns
from vlm.base import VLM

logger = logging.getLogger(__name__)

DIGIT_RE = re.compile(r"[0-3]")


def parse(text: str) -> int | None:
    """First 0-3 in the output, or None. None is recorded, never guessed."""
    m = DIGIT_RE.search(text)
    return int(m.group()) if m else None


@dataclass(slots=True)
class Result:
    """
    One scored question, and the schema of a JSONL line.

    `pred` is None when the completion held no 0-3 digit. `gold` and `correct`
    are None for unlabeled rows -- the test split ships no `answer_idx` -- which
    keeps them out of the accuracy denominator instead of counting them wrong.
    """

    video: str
    qid: str
    pred: int | None
    gold: int | None
    correct: bool | None
    raw: str

    @classmethod
    def score(cls, row: dict, video: str, completion: str) -> "Result":
        pred = parse(completion)
        gold = row.get(Columns.ANSWER_IDX)
        # pd.isna: a DataFrame of mixed labeled/unlabeled rows stores the gap as NaN.
        # int(): a numpy scalar from that same round-trip would make json.dumps raise.
        gold = None if pd.isna(gold) else int(gold)
        return cls(
            video=video,
            qid=row[Columns.QID],
            pred=pred,
            gold=gold,
            correct=None if gold is None else (pred is not None and pred == gold),
            raw=completion.strip(),
        )

    def as_dict(self) -> dict:
        return asdict(self)


def to_frame(results: list[Result]) -> pd.DataFrame:
    """
    Results as a DataFrame, for analysis only.

    Nullable dtypes are deliberate: plain int64/bool columns would turn every
    None into a NaN float, and NaN compares false against None and itself.
    """
    df = pd.DataFrame([r.as_dict() for r in results], columns=[f.name for f in fields(Result)])
    return df.astype({"pred": "Int64", "gold": "Int64", "correct": "boolean"})


def group_by_video(rows: pd.DataFrame | list[dict]) -> dict[str, list[dict]]:
    """{video_id: [row, ...]}, input order preserved."""
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict(orient="records")

    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r[Columns.VIDEO_ID], []).append(r)
    return groups


def summarize(results: list[Result | dict], n_examples: int = 5) -> float | None:
    """
    Print n / accuracy / unparsed rate. Returns accuracy, or None if unlabeled.

    Accepts dicts too, so a finished JSONL can be re-summarized without a rerun:
    `summarize([json.loads(l) for l in open(path)])`. A line whose keys don't
    match the schema raises here rather than skewing the numbers silently.
    """
    results = [r if isinstance(r, Result) else Result(**r) for r in results]
    n = len(results)
    if not n:
        logger.warning("nothing to summarize (n=0)")
        return None

    unparsed = [r for r in results if r.pred is None]
    scored = [r for r in results if r.correct is not None]
    acc = sum(r.correct for r in scored) / len(scored) if scored else None

    acc_str = f"{acc:.4f}" if acc is not None else "n/a (unlabeled)"
    logger.info(
        "n=%d  accuracy=%s  unparsed=%d (%.2f%%)",
        n, acc_str, len(unparsed), 100 * len(unparsed) / n,
    )
    if unparsed:
        # The completions themselves, since a high unparsed rate is almost always a
        # prompt or max_new_tokens problem rather than a model capability one.
        logger.warning("%d unparsed completions, first %d:", len(unparsed), min(n_examples, len(unparsed)))
        for r in unparsed[:n_examples]:
            logger.warning("  %s: %r", r.qid, r.raw)
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
) -> list[Result]:
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
        One `Result` per question answered this call -- on a resumed run that
        excludes rows already in `out_path`, which remains the full record.
        Videos that fail to decode are skipped with a printed reason and
        produce no results. Pass the list to `to_frame` for analysis.
    """
    groups = group_by_video(rows)

    out_path = Path(out_path) if out_path else None
    results: list[Result] = []
    if out_path and resume:
        skip = _done_qids(out_path)
        if skip:
            logger.info("resuming from %s: skipping %d completed questions", out_path, len(skip))
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
            # One line on the console even when dozens of clips fail; the traceback
            # goes to the file handler, which keeps DEBUG.
            logger.warning("skipping %s: %s: %s", video_id, type(e).__name__, e)
            logger.debug("traceback for %s", video_id, exc_info=True)
            progress.update(len(group))
            continue

        try:
            for i in range(0, len(group), batch_size):
                batch = group[i : i + batch_size]
                texts = vlm.answer(clip, batch, use_transcript=use_transcript)

                batch_results = [
                    Result.score(r, video_id, text) for r, text in zip(batch, texts)
                ]

                results.extend(batch_results)
                if out_path:
                    with out_path.open("a") as f:
                        for result in batch_results:
                            f.write(json.dumps(result.as_dict()) + "\n")

                progress.update(len(batch))
                scored = [r for r in results if r.correct is not None]
                if scored:
                    running = sum(r.correct for r in scored) / len(scored)
                    progress.set_postfix(acc=f"{running:.3f}")
                    # The bar is console-only; DEBUG leaves a progress trace in the file.
                    logger.debug(
                        "%s: %d/%d answered, running accuracy %.4f",
                        video_id, len(results), total, running,
                    )
        finally:
            del clip
            _flush_cuda()

    progress.close()
    summarize(results)
    return results
