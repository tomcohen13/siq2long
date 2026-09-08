"""
Encode chunks and queries (questions/questions+options) into one embedding cache.

Steps:
1. assemble the work list
2. encode each chunk's frames + transcript
3. encode each question
4. persist.

Chunks are stored as one flat table, keyed by `(vid, chunk_idx)` (not grouped per video).
"""

import logging
import pandas as pd
import torch

from pathlib import Path
from tqdm import tqdm

from config import ANSWER_KEYS, Columns
from data.transcripts import split_transcript_by_ranges
from data.videos import sample_windows
from encoders.base import DualEncoder, DualEncoderArtifact, DualEncoderOutput
from scoring import QUERY_TENSORS

logger = logging.getLogger(__name__)


class ChunkEncoding(DualEncoderOutput):
    """One video's chunks in both modalities, plus the text those transcripts came from."""

    texts: list[str]


class ArtifactTensors(DualEncoderOutput):
    """
    Every tensor an encode run produces: the chunk side and the query side.

    Chunk embeddings are one row per `(video, chunk)`; query embeddings are one row per
    question, in both query forms. Kept apart so scoring can pair any representation with
    any query form without re-encoding.
    """

    query_q: torch.Tensor
    query_qa: torch.Tensor


def render_query(row: dict, form: str) -> str:
    """
    The retrieval query text for one question, in one of the forms in `scoring.QUERY_TENSORS`.

    Every form is encoded because which one is fair is itself in question, and because the
    contrast between them is the diagnostic:

    - `question` -- the bare question. What a real system would have.
    - `question+options` -- also fair in a multiple-choice setting, but the options carry
      content the question does not.
    - `answer` -- the gold answer alone, a declarative statement about the clip and exactly
      the caption shape these encoders were pretrained on. Uses the label, so it is a
      **diagnostic only**; it isolates query form from query content (§3).

    Raises:
        ValueError: on an unknown form, or on `answer` for a row with no gold label (the
            test split has none) -- silently returning the question there would make the
            diagnostic compare a form against itself.
    """
    question = str(row[Columns.QUESTION])
    if form == "question":
        return question
    if form == "question+options":
        return " ".join([question, *(str(row[k]) for k in ANSWER_KEYS)])
    if form == "answer":
        gold = row.get(Columns.ANSWER_TEXT)
        if gold is None or pd.isna(gold):
            raise ValueError(f"no gold answer for {row.get(Columns.QID)!r}")
        return str(gold)
    raise ValueError(f"unknown query form {form!r}")


def render_queries_in_order(
    artifact: DualEncoderArtifact, qa: pd.DataFrame, form: str = "question+options"
) -> list[str]:
    """
    The query strings behind a cache's query embeddings, in its own qid order.

    The cache stores encoded queries, not their text, so anything that needs the strings
    back -- a lexical baseline, a query form added later -- rebuilds them through the same
    `render_query` the encoders used.
    """
    return [render_query(row, form) for row in rows_in_query_order(artifact, qa)]


def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def rows_in_query_order(artifact: DualEncoderArtifact, qa: pd.DataFrame) -> list[dict]:
    """
    QA rows ordered to match an artifact's query tensors, row for row.

    The query tensors are positional: row `i` is `meta["qids"][i]`. Anything that rebuilds
    query text later -- a lexical baseline, a query form added after the fact -- has to
    reproduce that order exactly, or it scores the wrong question against the right chunks.

    Raises:
        KeyError: if a cached qid is missing from `qa`. Reindexing past it would silently
            shift every row after it.
    """
    qa_by_qid = qa.drop_duplicates(Columns.QID).set_index(Columns.QID)
    qids = artifact["meta"]["qids"]
    absent = [qid for qid in qids if qid not in qa_by_qid.index]
    if absent:
        raise KeyError(f"{len(absent)} cached qids are not in qa, e.g. {absent[:5]}")
    return [qa_by_qid.loc[qid].to_dict() | {Columns.QID: qid} for qid in qids]

def encode_queries(
    encoder: DualEncoder, rows: list[dict], forms: list[str] | None = None
) -> dict[str, torch.Tensor]:
    """
    Encode each query form over the same rows, keyed by the tensor name it is stored under.

    Args:
        encoder: any `DualEncoder`; only its text tower is used.
        rows: QA records, in the order the artifact's `qids` are in.
        forms: which of `scoring.QUERY_TENSORS` to encode. Default is all of them, minus any that
            these rows cannot express -- the test split has no gold answer, so `answer` is
            dropped there rather than failing the run.
    """
    query_tensors = {}
    for form in forms if forms is not None else QUERY_TENSORS:
        try:
            query_texts = [render_query(row, form) for row in rows]
        except ValueError as reason:
            logger.warning("skipping query form %r: %s", form, reason)
            continue
        query_tensors[QUERY_TENSORS[form]] = torch.cat(
            [encoder.encode_texts(batch).cpu()
             for batch in _batches(query_texts, encoder.batch_size)]
        )
        logger.info("encoded %d queries as %r", len(query_texts), form)
    return query_tensors


def encode_chunks(
    encoder: DualEncoder,
    video_path: Path,
    transcript_path: Path,
    chunks: list[list[float]]
) -> ChunkEncoding:
    """
    Encode one video's chunks in both modalities.

    Frames are decoded per chunk straight from the full-length file -- no clip is written,
    which is what keeps the oracle window exactly on the SIQ2 trim. Batching is bounded by
    `encoder.batch_size` because the frames, not the embeddings, are what fills memory.

    Args:
        encoder: a DualEncoder-supporting model
        video_path: path to the full-length video
        transcript_path: path to the full-length transcript
        chunks: list of `[start, end]` ranges in seconds
    """
    texts = split_transcript_by_ranges(transcript_path, chunks)
    frames = sample_windows(video_path, chunks, encoder.num_frames)

    video_embeds = []
    text_embeds = []

    for batch_frames, batch_texts in zip(
        _batches(frames, encoder.batch_size), _batches(texts, encoder.batch_size)
    ):
        # encode and move to CPU
        video_embeds.append(encoder.encode_videos(batch_frames).cpu())
        text_embeds.append(encoder.encode_texts(batch_texts).cpu())

    video, text = torch.cat(video_embeds), torch.cat(text_embeds)
    # One row per chunk in both modalities, complain otherwise.
    if not len(video) == len(text) == len(chunks):
        raise RuntimeError(
            f"encoded {len(video)} video and {len(text)} text rows "
            f"for {len(chunks)} chunks of {video_path}"
        )
    out: ChunkEncoding = {
        "video_embeddings": video,
        "text_embeddings": text,
        "texts": texts,
    }
    return out


def encode(
    encoder: DualEncoder,
    manifest: dict[str, dict],
    qa: pd.DataFrame,
    files: pd.DataFrame,
    limit: int | None = None,
) -> tuple[ArtifactTensors, dict]:
    """
    Encode the chunks of every manifest video, and every question asked about them.

    Split selection stays with the caller: `load_qa` picks a split by filename, so there
    is nothing here to filter on and a `split` argument would only be a label this
    function could not enforce. It encodes exactly the rows it is handed.

    Args:
        encoder: any `DualEncoder`; nothing below is backbone-specific
        manifest: `{vid: {"chunks": [[s, e], ...], "oracle_idx": int, ...}}`
        qa: the QA rows to encode, as returned by `load_qa`
        files: `find_downloaded_files(...)`, giving video and transcript paths
        limit: first N videos only, for smoke tests

    Returns:
        `(tensors, meta)` ready for `DualEncoder.save`. A video whose frames or transcript
        fail to decode is skipped and recorded in `meta["failed"]` rather than taking the
        whole run down partway through.
    """
    paths = files.set_index(Columns.VIDEO_ID)
    vids = sorted(v for v in manifest if v in paths.index)
    if limit:
        vids = vids[:limit]
    logger.info("encoding %d videos with %s", len(vids), encoder.checkpoint)

    video_embs, text_embs, chunk_ids, chunk_texts, failed = [], [], [], [], []
    for vid in tqdm(vids, desc="chunks", unit="video"):
        chunks = manifest[vid]["chunks"]
        try:
            chunk_embeds = encode_chunks(
                encoder,
                video_path=paths.loc[vid, Columns.VIDEO_PATH],
                transcript_path=paths.loc[vid, Columns.TRANSCRIPT_PATH],
                chunks=chunks,
            )
        except Exception:
            logger.exception("could not encode %s; skipping", vid)
            failed.append(vid)
            continue
        video_embs.append(chunk_embeds["video_embeddings"])
        text_embs.append(chunk_embeds["text_embeddings"])
        chunk_ids.extend((vid, i) for i in range(len(chunks)))
        chunk_texts.extend(chunk_embeds["texts"])

    if not video_embs:
        raise RuntimeError("no video encoded successfully")

    encoded = {vid for vid, _ in chunk_ids}
    rows = qa[qa[Columns.VIDEO_ID].isin(encoded)].to_dict("records")
    logger.info("encoding %d questions over %d videos", len(rows), len(encoded))

    queries = encode_queries(encoder, rows)

    tensors: ArtifactTensors = {
        "video_embeddings": torch.cat(video_embs),
        "text_embeddings": torch.cat(text_embs),
        **queries,
    }
    meta = {
        "chunk_ids": chunk_ids,
        "chunk_texts": chunk_texts,
        "oracle_idx": {vid: manifest[vid]["oracle_idx"] for vid in encoded},
        "chunks": {vid: manifest[vid]["chunks"] for vid in encoded},
        "qids": [r[Columns.QID] for r in rows],
        "query_vid": [r[Columns.VIDEO_ID] for r in rows],
        "answer_idx": [r.get(Columns.ANSWER_IDX) for r in rows],
        "failed": failed,
    }
    return tensors, meta