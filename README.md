# SIQ2-Long

Video QA benchmarks hand the model its evidence. Social-IQ 2.0 pairs each question with the
sixty-second clip it was written about, so a model that scores well has shown it can
interpret a clip — not that it could have found one. **SIQ2-Long** removes that assumption:
it restores the full-length source videos, cuts them into one-minute chunks, and asks a
system to locate the answer-bearing chunk before answering.

We call the task **oracle finding**. Given a full video divided into 1-minute chunks and a
social reasoning question, identify the chunk the question was written about.

This repository holds the dataset recipe, the retrieval and answering pipelines, and the
controls behind the paper.

## The dataset

748 videos, median 5.9 minutes and at most 19.4, cut into 5,630 chunks. Chunking starts from
the original Social-IQ 2.0 window and works outward in both directions, so that window stays
intact as a chunk of its own — the **oracle**. Edge fragments under ten seconds are absorbed
into the neighbouring chunk; nothing is ever absorbed into the oracle, so its bounds always
match the original trim.

| split | videos | chunks | questions | drawn from |
|---|---|---|---|---|
| train | 406 | 2,953 | 2,876 | SIQ2 train |
| dev | 101 | 732 | 718 | SIQ2 train |
| test | 119 | 776 | 778 | SIQ2 validation |

A further 122 videos (1,169 chunks) carry no labelled questions and serve only as retrieval
distractors. Splits are by video, so no video appears in more than one and every chunk of a
held-out video is held out with it, including as a training negative.

Videos yielding fewer than 2 chunks are dropped — the oracle would be the only candidate and
retrieval would be free. Videos yielding more than 20 are dropped as well: ranking over a
ninety-chunk pool is a different problem, and a few very long videos would dominate any
video-weighted average.

**No clips are written to disk.** `siq2long/video_chunks.json` is the dataset: one entry per
video giving its duration, its chunk boundaries in seconds, and the index of its oracle.
Frames are decoded from the source file at read time. Cutting the chunks would snap their
edges to keyframes and drift the oracle off the original window.

Each chunk carries two modalities — the frames, and the subtitle lines falling inside its
window. Retrieval may use either or both.

### Attrition

Social-IQ 2.0 names 1,132 videos across its two labelled splits; 748 were still retrievable
when we built this. The benchmark distributes YouTube identifiers and availability decays,
so this attrition rather than annotation sets the size of the dataset. That is why the
embedding caches are published alongside the recipe: they are reproducible in a way the
download no longer is.

## Install

```bash
uv venv --python 3.12
uv pip install -e .          # or: export PYTHONPATH=$PWD/src
uv run pytest                # 307 tests, no GPU and no data needed
```

`src/` is the import root, so modules are `config`, `inference`, `vlm.*`, `data.*` with no
package prefix. There is deliberately no `src/__init__.py`: if the repo root were also
importable, `vlm.base` and `src.vlm.base` would be two distinct modules and `isinstance`
checks against `VLM` would fail across them.

## Data

The light half ships with this repository and is versioned with the code: QA rows
(`siq2long/qa/`), splits (`siq2long/splits/`), the original SIQ2 offsets (`trims.json`) and
the chunk manifest (`video_chunks.json`).

The heavy half does not. Point `PATH_TO_SIQ2LONG` at wherever you put it:

```bash
# .env, or the environment
PATH_TO_SIQ2LONG=/path/to/media    # expects video/ and transcript/ underneath
PATH_TO_SIQ2=/path/to/siq2         # the original 60s trims, for the SIQ2 baseline
```

Unset, both fall back inside the repo alongside the QA rows.

**Videos.** `scripts/download_all.py --mode siq2-full` fetches full-length video and
subtitles from YouTube. Expect a smaller set than ours; see Attrition above.

**Embedding caches.** ~62 MB of `.pt` files published as release assets. Download them into
`siq2long/embeddings/` and every retrieval result reproduces with no video required — this
is the recommended path.

```bash
# TODO: release URL
curl -LO <release-url>/xclip_train.pt      # 507 train+dev videos, 3,685 chunks
curl -LO <release-url>/xclip_val.pt        # 119 test videos, 776 chunks
curl -LO <release-url>/pe-video_val.pt     # the same videos, PE-Video backbone
```

> **Split naming.** The scripts take `--split train|val|test`, which are the *Social-IQ 2.0*
> split names, not this paper's. `--split val` is the paper's **test** split: the 119 videos
> drawn from SIQ2 validation. Cache files inherit that name, so `xclip_val.pt` holds the
> test-split embeddings. `siq2long/splits/` uses the paper's names.

## Pipeline

Each script is one step and writes an artifact the next one reads, so nothing re-runs the
expensive half.

```bash
# 1. Chunk manifest. ffprobe plus arithmetic, seconds on a laptop.
#    You almost certainly do not need this -- the manifest ships with the repo.
python scripts/build_manifest.py

# 2. Encode a split's chunks and questions into one cache. The expensive step:
#    decodes frames, runs the backbone. Skip it by downloading the caches above.
python scripts/encode_chunks.py --encoder xclip --split val

# 3. Score oracle finding. Tensor ops over the cache, so a full run takes seconds.
python scripts/eval_retrieval.py --cache siq2long/embeddings/xclip_val.pt --pool-type both

# 4. Answer each question from one retrieved chunk.
python scripts/run_inference.py --model qwen3-vl --dataset siq2long --split val \
    --embeddings siq2long/embeddings/xclip_val.pt --condition top1
```

`--condition oracle` is the ceiling a perfect retriever reaches, `top1` is what the retriever
returns, `random` is the floor. Re-running a command resumes from its output file, so a
preempted run costs only the batch in flight.

Also available: `scripts/train_adapter.py` fits a head on the frozen embeddings (minutes on a
laptop, since the backbone never runs), and `scripts/eval_baselines.py` scores the BM25 and
BGE text-only baselines over chunk transcripts.

Encoders: `xclip`, `pe-video`. Answerers: `qwen2.5-vl`, `qwen3-vl`, `internvl3`,
`llava-next-video`, `videollama3`.

## What we find

Numbers below are X-CLIP on the test split, 778 questions, ranking against the chunks of the
question's own video. The chance floor is 22.4% — pools run from 2 to 20 chunks, so it is
`mean(1/pool)` rather than a constant.

**Frozen encoders identify the video, not the minute.** Given a bare question, X-CLIP
recovers the oracle at **19.3%**, below the 22.4% floor. Against a same-sized pool drawn from
*other* videos it reaches **36.2%**. It can tell one video from another and cannot tell one
minute of a video from the next.

**Given a question, encoders rank uninformative chunks above the evidence.** 17.5% of chunks
carry under 20 transcript words or run under 45 seconds. For a bare question X-CLIP puts
those at mean pool position 0.396 and the oracle at 0.534 (0 = ranked first) — the ordering
is inverted. Supplying the answer options, which are declarative, reverses it. An
interrogative query has little descriptive content to match against.

**A trained head improves the ranking without consulting the question.** A residual head over
the frozen embeddings raises top-1 by **+21.3pp**, demoting uninformative chunks and
promoting oracle-like ones. Handed an *unrelated* question it retains most of that: about 85%
of the gain requires no question at all. What it learns is which chunks tend to be evidence,
not which chunk answers the question at hand.

**Perfect retrieval buys little.** Across four answerers, an oracle chunk beats a randomly
chosen one by 0.6 to 7.7 points — a neighbouring minute usually contains enough to answer.
The constraint is in selecting evidence, not in reasoning over it.

See the paper for the full tables, the controls, and the caveats.

## Layout

```
src/config.py      paths, column names, dataset registry
src/data/          QA loading, transcript handling, chunking, frame sampling
src/encoders/      dual encoders (X-CLIP, PE-Video) and the text-only baselines
src/vlm/           one module per answering backend, all implementing VLM
src/retrieval.py   encode a split into a cache
src/scoring.py     pools, ranks, retrieval metrics
src/adapters.py    trainable heads over frozen embeddings
src/train.py       the contrastive training loop
src/inference.py   the answering loop
src/stats.py       Wilson intervals, exact McNemar
src/plots.py       every figure in the paper
scripts/           one CLI per pipeline step
tests/             307 tests, no GPU or data required
```


Built on [Social-IQ 2.0](https://github.com/abwilf/Social-IQ-2.0-Challenge). The download
script is adapted from the original SIQ codebase.
