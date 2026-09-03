"""Base class for a dual-encoder"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch


class DualEncoderOutput(TypedDict):
    text_embeddings: torch.Tensor
    video_embeddings: torch.Tensor

class DualEncoderArtifact(TypedDict):
    """
    What `DualEncoder.save` writes / `retrieval.encode` returns.

    The tensors are the embeddings, and the meta is the non-tensor payload that makes
    them addressable: which (video, chunk) each row is, and the chunk transcripts BM25
    needs.
    """
    checkpoint: str
    num_frames: int
    meta: dict[str, Any]
    tensors: dict[str, torch.Tensor]


class DualEncoder(torch.nn.Module, ABC):
    """
    Two towers -- text and video -- projecting into one shared embedding space.

    Scoped to what the oracle-find pipeline actually does: embed a video's chunks, embed
    a query, compare them. So the contract is two encoders plus the guarantee that makes
    the comparison meaningful -- both return L2-normalized vectors of the same width, in
    the same space, so a dot product is cosine similarity. Everything a given backbone
    needs to honor that (processors, projections, temporal pooling) stays in its subclass.

    Subclasses are `nn.Module` so `.to(device)` and `.eval()` work uniformly and so the
    fusion adapters can wrap one, but nothing here assumes the weights are trainable --
    the backbone is frozen.
    """

    #: HF id (or equivalent) of the weights, recorded alongside saved embeddings.
    checkpoint: str

    #: Frames the video tower expects per chunk; the pipeline samples exactly this many.
    num_frames: int

    #: Items per forward pass. A VRAM hint for the pipeline, which does the batching --
    #: a 97-minute video is 97 chunks and will not fit in one pass.
    batch_size: int = 8

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @abstractmethod
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of strings. Returns (n_texts, dim), L2-normalized."""
        raise NotImplementedError("Must implement method within inheriting class")

    @abstractmethod
    def encode_videos(self, videos: list[np.ndarray]) -> torch.Tensor:
        """
        Encode a batch of video chunks. Returns (n_videos, dim), L2-normalized.

        Each element is one chunk's frames as a (num_frames, H, W, 3) uint8 array --
        the shape `data.videos.sample_frames` produces.
        """
        raise NotImplementedError("Must implement method within inheriting class")

    def similarity(self, queries: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """
        (n_queries, n_candidates) similarity scores.

        Plain cosine, which the normalization contract above reduces to a dot product.
        Override for a backbone that scores with a learned temperature or a non-cosine
        head; the pipeline reads scores only through here so ranking stays comparable.
        """
        return queries @ candidates.T

    def save(self, path: str | Path, meta: dict[str, Any] | None = None, **tensors) -> None:
        """
        Write named embedding tensors to `path`, stamped with what produced them.

        The stamp is the point. Embeddings are cached so every scoring variant -- fused
        or per-modality, question-only or question-plus-answers, same-video or cross-video
        distractors -- can be recomputed without touching video again. Two runs of that
        analysis are only comparable if the file says which checkpoint and frame count it
        came from, and swapping backbones means there will be several such files side by
        side. `meta` carries the non-tensor payload that makes the rows addressable: which
        (video, chunk) each row is, and the chunk transcripts BM25 needs.

        Tensors are detached and moved to CPU on the way out, so a saved cache never
        carries a CUDA device or an autograd graph into whatever loads it next.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact: DualEncoderArtifact = {
            "checkpoint": self.checkpoint,
            "num_frames": self.num_frames,
            "meta": meta or {},
            "tensors": {name: t.detach().cpu() for name, t in tensors.items()},
        }
        torch.save(artifact, path)

    @staticmethod
    def load(path: str | Path) -> DualEncoderArtifact:
        """
        Read back a `save` bundle: {"checkpoint", "num_frames", "meta", "tensors"}.

        `weights_only=False` because `meta` holds plain Python -- ids and transcripts --
        not just tensors. Only ever point this at a cache this repo wrote.
        """
        artifact: DualEncoderArtifact = torch.load(
            Path(path), map_location="cpu", weights_only=False
        )
        return artifact
