"""Perception Encoder (video branch) text + video encoders (weights frozen)."""
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, PeVideoModel

from encoders.base import DualEncoder, DualEncoderOutput


class PEVideoEncoder(DualEncoder):
    """
    Meta's Perception Encoder video branch, wrapped as a two-tower encoder.

    `PeVideoModel` exposes no per-tower getters.
    So the two branches below reproduce forward's per modality:

        video: video_encoder(...).pooler_output -> video_head
        text:  text_model(..., output_hidden_states=True).hidden_states[-1][:, 0]
               -> text_video_head
    """

    CHECKPOINT = "facebook/pe-av-large"
    NUM_FRAMES = 16

    def __init__(self, checkpoint: str = CHECKPOINT):
        super().__init__()
        self.checkpoint = checkpoint
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.model, info = PeVideoModel.from_pretrained(checkpoint, output_loading_info=True)
        # Unexpected keys are fine (audio tower) but not missing.
        if info["missing_keys"]:
            raise RuntimeError(f"{checkpoint} left weights uninitialized: {info['missing_keys'][:5]}")
        self.model.requires_grad_(False)
        self.num_frames = self.NUM_FRAMES

    @property
    def max_text_tokens(self) -> int:
        """Text tower's context window -- the ceiling on how much transcript survives."""
        return self.model.config.text_config.max_position_embeddings

    @torch.inference_mode()
    def forward(self, frames: np.ndarray, transcript: str) -> DualEncoderOutput:
        """One chunk through both towers. Convenience over the batch methods below."""
        return {
            "text_embeddings": self.encode_texts([transcript]),
            "video_embeddings": self.encode_videos([frames]),
        }

    @torch.inference_mode()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of strings. Returns (n_texts, embed_size)."""
        text_inputs = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        text_out = self.model.text_model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            output_hidden_states=True,
        )
        text_emb = self.model.text_video_head(text_out.hidden_states[-1][:, 0])
        return F.normalize(text_emb, dim=-1)

    @torch.inference_mode()
    def encode_videos(self, videos: list[np.ndarray]) -> torch.Tensor:
        """Encode a batch of video chunks. Returns (n_chunks, embed_size)."""
        # `padding_mask_videos` is only produced when return_tensors is set; without it
        # the processor hands back a list of per-clip tensors and no mask.
        video_inputs = self.processor.video_processor(
            list(videos),
            num_frames=self.num_frames,
            return_tensors="pt",
        ).to(self.device)
        video_out = self.model.video_encoder(
            pixel_values_videos=video_inputs["pixel_values_videos"],
            padding_mask_videos=video_inputs.get("padding_mask_videos"),
        )
        video_emb = self.model.video_head(video_out.pooler_output)
        return F.normalize(video_emb, dim=-1)
