"""
Encode a split's chunks and questions into one embedding cache.

Stages 0-3 of oracle find: assemble the work list, encode each chunk's frames and its
transcript, encode each question, persist. Scoring lives downstream and reads only the
cache, so trying another scoring variant costs seconds rather than a re-encode.

Chunks are stored as one **flat table** keyed by `(vid, chunk_idx)` rather than grouped
per video. That is what lets a retrieval pool be assembled from any subset later --
same-video distractors, distractors drawn from other videos, or a mix -- without encoding
anything again. Chunk transcripts are persisted alongside their embeddings so lexical
baselines run off the cache too, with no second pass over the VTTs.
"""

import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from config import ANSWER_KEYS, Columns
from data.transcripts import split_transcript_by_ranges
from data.videos import sample_windows
from encoders.base import DualEncoder

logger = logging.getLogger(__name__)


def render_query(row: dict, with_options: bool) -> str:
    """
    The retrieval query for one question.

    Both forms are encoded because which one is fair is itself in question: the options
    are available at retrieval time in a multiple-choice setting, but they also carry
    content the question alone does not.
    """
    question = str(row[Columns.QUESTION])
    if not with_options:
        return question
    return " ".join([question, *(str(row[k]) for k in ANSWER_KEYS)])


def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def encode_chunks(
    encoder: DualEncoder,
    video_path: Path,
    transcript_path: Path,
    chunks: list[list[float]]
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
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

    video_embeds, text_embeds = torch.cat(video_embeds), torch.cat(text_embeds)
    # A row per chunk in both modalities, or every downstream lookup by chunk index is
    # silently off. `encode_texts` accepts a bare string and returns one row for it, so
    # this misalignment is the kind that reaches scoring rather than raising here.
    if not len(video_embeds) == len(text_embeds) == len(chunks):
        raise RuntimeError(
            f"encoded {len(video_embeds)} video and {len(text_embeds)} text rows "
            f"for {len(chunks)} chunks of {video_path}"
        )
    return video_embeds, text_embeds, texts


def encode(
    encoder: DualEncoder,
    manifest: dict[str, dict],
    qa: pd.DataFrame,
    files: pd.DataFrame,
    limit: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict]:
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
            v_emb, t_emb, texts = encode_chunks(
                encoder,
                paths.loc[vid, Columns.VIDEO_PATH],
                paths.loc[vid, Columns.TRANSCRIPT_PATH],
                chunks,
            )
        except Exception:
            logger.exception("could not encode %s; skipping", vid)
            failed.append(vid)
            continue
        video_embs.append(v_emb)
        text_embs.append(t_emb)
        chunk_ids.extend((vid, i) for i in range(len(chunks)))
        chunk_texts.extend(texts)

    if not video_embs:
        raise RuntimeError("no video encoded successfully")

    encoded = {vid for vid, _ in chunk_ids}
    rows = qa[qa[Columns.VIDEO_ID].isin(encoded)].to_dict("records")
    logger.info("encoding %d questions over %d videos", len(rows), len(encoded))

    queries = {}
    for name, with_options in (("query_q", False), ("query_qa", True)):
        texts = [render_query(r, with_options) for r in rows]
        queries[name] = torch.cat(
            [encoder.encode_texts(b).cpu() for b in _batches(texts, encoder.batch_size)]
        )

    tensors = {
        "chunk_video": torch.cat(video_embs),
        "chunk_text": torch.cat(text_embs),
        **queries,
    }
    meta = {
        # (vid, chunk_idx) per row of the chunk tensors -- the addressing that lets any
        # pool of distractors be assembled at scoring time.
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