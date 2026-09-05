# Embedding caches

**This directory is empty in a fresh clone.** The `.pt` files are ~62 MB in total, too big
for git history, so they are published as release assets. Download them here and every
retrieval result in the paper reproduces with no video required.

| file | size | contents |
|---|---|---|
| `xclip_train.pt` | 40 MB | 507 train videos, 3,685 chunks, 3,594 questions |
| `xclip_val.pt` | 8.5 MB | 119 val videos, 776 chunks, 778 questions |
| `pe-video_val.pt` | 17 MB | the same val videos, PE-Video backbone |

    # TODO: replace with the release URL once it is cut
    curl -LO <release-url>/xclip_train.pt
    curl -LO <release-url>/xclip_val.pt
    curl -LO <release-url>/pe-video_val.pt

## Why these are published rather than rebuilt

Rebuilding them means re-downloading the source videos from YouTube, and those rot. SIQ2's
train split names 987 videos; 748 were still fetchable when we built this. Anyone repeating
that download later gets a smaller set and therefore different numbers, so a from-scratch
rebuild cannot confirm or contradict what we report. These caches can.

They are float32 on purpose. Half precision would halve the download, but `scoring.rank_of`
breaks ties pessimistically, so fp16 rounding would move a handful of ranks and the
published numbers would no longer match exactly.

## What is in one

A dict of `{checkpoint, num_frames, meta, tensors}` — read it with `DualEncoder.load`.
`tensors` holds one row per chunk (`video_embeddings`, `text_embeddings`) and one row per
question in each query form (`query_q`, `query_qa`, `query_a`); `meta` holds what makes
those rows addressable — `chunk_ids`, `chunk_texts`, `oracle_idx`, `qids`, `query_vid`.
The two row spaces are different heights and are joined only by video id.

## Rebuilding anyway

    python scripts/encode_chunks.py --encoder xclip --split val

Writes here by default. Pass `--media-dir` if your videos are not where
`PATH_TO_SIQ2LONG` points.
