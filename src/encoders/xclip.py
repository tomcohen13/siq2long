"""X-CLIP text + video encoders (weights frozen)."""
import numpy as np
import torch
import torch.nn.functional as F
from transformers import XCLIPModel, XCLIPProcessor

from encoders.base import DualEncoder, DualEncoderOutput


class XCLIPEncoder(DualEncoder):
    """XCLIP-based dual encoder"""

    CHECKPOINT = "microsoft/xclip-base-patch16-16-frames"

    def __init__(self, checkpoint: str = CHECKPOINT):
        super().__init__()
        self.checkpoint = checkpoint
        # Slow tokenizer avoids subtle mismatches vs the fast Rust tokenizer on some inputs.
        self.processor = XCLIPProcessor.from_pretrained(checkpoint, use_fast=False)
        self.model = XCLIPModel.from_pretrained(checkpoint)
        self.model.requires_grad_(False)
        self.num_frames = self.model.config.vision_config.num_frames

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
        text_out = self.model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        )
        return F.normalize(text_out.pooler_output, dim=-1)

    @torch.inference_mode()
    def encode_videos(self, videos: list[np.ndarray]) -> torch.Tensor:
        """Encode a batch of video chunks. Returns (n_chunks, embed_size)."""
        # The image processor wants each video as a sequence of frames; `list(v)` unpacks
        # the (num_frames, H, W, 3) arrays our sampler returns and is a no-op on lists.
        frames = [list(v) for v in videos]
        video_inputs = self.processor.image_processor(frames, return_tensors="pt").to(self.device)
        video_out = self.model.get_video_features(pixel_values=video_inputs["pixel_values"])
        return F.normalize(video_out.pooler_output, dim=-1)
