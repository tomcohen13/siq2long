"""
Trainable heads over frozen dual-encoder embeddings.

The backbone never runs here. Chunks and questions were encoded once into a cache, so
training is arithmetic on 512-d vectors. What is learned is only how to *move* those
vectors: a residual branch on each side of the dot product, plus the two weights that fuse
a chunk's video and transcript towers.

Every residual branch ends in a zero-initialized layer, so an untrained adapter reproduces
the frozen baseline bit for bit. That is deliberate. It means every point of movement is
attributable to training, and -- more importantly for this project -- that a *flat* result
is readable: if `question` retrieval does not improve, it did not improve from a verified
starting point, rather than from a representation we might have broken on the way in.

Two shapes are provided. `SmallAdapter` is a low-rank bottleneck, tens of thousands of
parameters, sized for the fact that there are only ~2.9k training questions. `BigAdapter`
is a wide MLP, low tens of millions. The comparison is itself a result: if the big one does
not beat the small one, the ceiling is the frozen representation rather than the head.
"""

import logging
from abc import ABC, abstractmethod
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders.base import DualEncoderArtifact
from scoring import QUERY_TENSORS

logger = logging.getLogger(__name__)

#: CLIP's temperature, and the scale the frozen baseline is implicitly scored at.
INITIAL_TEMPERATURE = 0.07

#: The floor CLIP uses, expressed as a temperature rather than as a logit scale of 100.
MIN_TEMPERATURE = 0.01


class DualEncoderAdapter(nn.Module, ABC):
    """
    Maps a cached query and a cached (video, transcript) chunk pair into one shared space.

    Query side:  `q' = normalize(q + residual_q(q))`
    Chunk side:  `c' = normalize(alpha*v + beta*t + residual_c([v; t]))`

    With both residuals at zero and `alpha = beta = 1`, `c'` is exactly the `fused`
    representation `scoring.select_chunk_embeddings` builds and `q'` is exactly the cached
    query -- which is where training starts.

    Subclasses supply `build_residual`; everything else is shared. A subclass must assign
    its own configuration *before* calling `super().__init__()`, because the base
    constructor calls `build_residual` while wiring the two branches.

    Args:
        dim: width of the cached embeddings. 512 for X-CLIP base.
        freeze_chunks: train the query side alone and leave chunks exactly as cached. A
            query-only adapter has no way to learn "what oracle chunks look like", so any
            gain it makes is necessarily query-dependent -- see the note below.
        checkpoint: the backbone these embeddings came from, carried for provenance.

    A note on what can go wrong. Trained on within-video pools, an adapter can raise top-1
    without learning anything about questions, by learning that oracle chunks are wordier,
    or longer, or rarely the short trailing one. A positional prior alone already scores
    31.7% on val against fused's 34.3%, so that failure would look like success. Two things
    catch it: `freeze_chunks=True`, which makes it structurally impossible, and scoring the
    trained adapter with mismatched queries, where a chunk prior keeps its gain and a real
    matcher collapses to chance.
    """

    def __init__(self, dim: int = 512, freeze_chunks: bool = False, checkpoint: str = ""):
        super().__init__()
        self.dim = dim
        self.freeze_chunks = freeze_chunks
        self.checkpoint = checkpoint

        # How the two towers are fused. The cache's `fused` weights them equally by
        # assumption; on train, transcript alone beats video alone (24.4% vs 21.0%), so
        # this is worth letting the data set.
        self.alpha = nn.Parameter(torch.ones(()))
        self.beta = nn.Parameter(torch.ones(()))

        self.query_residual = self.build_residual(dim, dim)
        self.chunk_residual = self.build_residual(2 * dim, dim)

        # Scores are cosines in [-1, 1]; the loss needs them spread before a softmax.
        # Learned in log space so it stays positive. Ranking is invariant to it, so this
        # changes the loss surface and never the reported metric.
        self.log_temperature = nn.Parameter(torch.tensor(INITIAL_TEMPERATURE).log())

        if freeze_chunks:
            self.alpha.requires_grad_(False)
            self.beta.requires_grad_(False)
            for p in self.chunk_residual.parameters():
                p.requires_grad_(False)

    @abstractmethod
    def build_residual(self, fan_in: int, fan_out: int) -> nn.Module:
        """
        The residual branch for one side.

        Must end in a zero-initialized layer, or the adapter will not start at the frozen
        baseline and a flat result stops being interpretable. Return `nn.Identity()` only
        for a configuration that deliberately has no residual path -- `encode_*` checks for
        it, since an identity branch would otherwise *double* the input rather than add zero.
        """

    @property
    def temperature(self) -> torch.Tensor:
        """The softmax scale. Floored, because a temperature at zero turns the loss to NaN."""
        return self.log_temperature.exp().clamp(min=MIN_TEMPERATURE)

    def encode_queries(self, queries: torch.Tensor) -> torch.Tensor:
        """`(n, dim)` cached queries -> `(n, dim)`, L2-normalized."""
        if isinstance(self.query_residual, nn.Identity):
            return F.normalize(queries, dim=-1)
        return F.normalize(queries + self.query_residual(queries), dim=-1)

    def encode_chunks(self, video: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """`(n, dim)` video and transcript embeddings -> one `(n, dim)` chunk, L2-normalized."""
        fused = self.alpha * video + self.beta * text
        if isinstance(self.chunk_residual, nn.Identity):
            return F.normalize(fused, dim=-1)
        return F.normalize(fused + self.chunk_residual(torch.cat([video, text], dim=-1)), dim=-1)

    def forward(
        self, queries: torch.Tensor, video: torch.Tensor, text: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Both sides at once. Returns `(queries, chunks)`, in that order."""
        return self.encode_queries(queries), self.encode_chunks(video, text)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SmallAdapter(DualEncoderAdapter):
    """
    A low-rank residual: down to `rank` dimensions and back up.

    Sized against the data rather than the hardware. There are ~2.9k training questions, so
    `rank=8` is ~20k parameters and `rank=32` is ~82k. Setting `rank=0` removes both
    branches entirely, leaving `alpha`, `beta` and the temperature -- a three-parameter
    model that can only relearn how to weight the two towers, which is a useful floor.

    Args:
        rank: width of the bottleneck. 0 disables the residual branches.
        dropout: inside the branch only, so it never disturbs the identity path.
    """

    def __init__(self, dim: int = 512, rank: int = 8, dropout: float = 0.0, **kwargs):
        # Set before super().__init__(), which calls build_residual.
        self.rank = rank
        self.dropout = dropout
        super().__init__(dim=dim, **kwargs)

    def build_residual(self, fan_in: int, fan_out: int) -> nn.Module:
        if self.rank < 0:
            raise ValueError(f"rank must be non-negative, got {self.rank}")
        if self.rank == 0:
            return nn.Identity()
        up = nn.Linear(self.rank, fan_out, bias=False)
        nn.init.zeros_(up.weight)
        return nn.Sequential(
            nn.Linear(fan_in, self.rank, bias=False),
            nn.GELU(),
            nn.Dropout(self.dropout),
            up,
        )


class BigAdapter(DualEncoderAdapter):
    """
    Two hidden layers of `hidden` units, then a zero-initialized projection.

    Roughly 13M parameters at the default width, against ~2.9k training questions, so it
    will overfit without early stopping -- that is expected and is what the dev split is
    for. It exists to answer one question: is capacity the binding constraint? If it does
    not beat `SmallAdapter`, the frozen representation is the ceiling, not the head. Depth
    is fixed at two because this is not an architecture paper and one point of comparison
    is enough to answer that.

    LayerNorm on the branch input keeps the wide path stable; it sits inside the residual,
    so the identity path is untouched and an untrained model still reproduces the baseline.

    Args:
        hidden: width of both hidden layers.
        dropout: applied after each of them.
    """

    def __init__(self, dim: int = 512, hidden: int = 2048, dropout: float = 0.1, **kwargs):
        # Set before super().__init__(), which calls build_residual.
        self.hidden = hidden
        self.dropout = dropout
        super().__init__(dim=dim, **kwargs)

    def build_residual(self, fan_in: int, fan_out: int) -> nn.Module:
        out = nn.Linear(self.hidden, fan_out)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        return nn.Sequential(
            nn.LayerNorm(fan_in),
            nn.Linear(fan_in, self.hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            out,
        )


#: Selectable by name from the CLI.
ADAPTERS = {"small": SmallAdapter, "big": BigAdapter}


@torch.no_grad()
def adapt_artifact(adapter: DualEncoderAdapter, artifact: DualEncoderArtifact) -> DualEncoderArtifact:
    """
    Push a whole cache through a trained adapter, returning a cache of the same shape.

    This is what keeps evaluation honest: the result is an ordinary artifact, so
    `oracle_ranks`, `question_pools`, `eval_retrieval.py` and the McNemar tests all run on
    the adapted model unchanged, and adapted and frozen numbers come out of the same code.

    The adapted chunk vector is written to **both** tower keys. After adaptation the towers
    are no longer separable -- fusion is learned rather than a mean taken afterwards -- so
    all three of `scoring.REPRESENTATIONS` collapse to the same vector, and `fused` (which
    renormalizes `v + t`) returns it exactly.
    """
    adapter.eval()
    tensors = artifact["tensors"]
    chunks = adapter.encode_chunks(tensors["video_embeddings"], tensors["text_embeddings"])

    adapted = {"video_embeddings": chunks, "text_embeddings": chunks}
    for name in QUERY_TENSORS.values():
        if name in tensors:
            adapted[name] = adapter.encode_queries(tensors[name])

    # Debug, not info: the training loop calls this once an epoch to score dev.
    logger.debug(
        "adapted %d chunks and %d query forms through %s (%d trainable parameters)",
        len(chunks), len(adapted) - 2, type(adapter).__name__, adapter.trainable_parameters(),
    )
    return {
        "checkpoint": artifact["checkpoint"],
        "num_frames": artifact["num_frames"],
        "meta": deepcopy(artifact["meta"]),
        "tensors": adapted,
    }
