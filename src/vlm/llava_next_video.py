"""LLaVA-NeXT-Video-7B, driven with frames we decode ourselves."""

import logging

import torch
from transformers import AutoProcessor, LlavaNextVideoForConditionalGeneration

from data.utils import sample_frames
from vlm.base import VLM

logger = logging.getLogger(__name__)


class LlavaNextVideo(VLM):
    """
    LLaVA-NeXT-Video-7B. Vicuna backbone, transformers-native, no remote code.

    Frames are decoded here and handed to the processor as a [T, H, W, C] array.
    transformers only reaches for its own decoder when the input isn't already a
    video array (`video_processing_utils._decode_and_sample_videos`), so this keeps
    decoding under our control -- no torchcodec, no CUDA ABI to match -- and it
    decodes each clip once for all of that clip's questions, which is what
    `prepare_clip` exists for.

    `num_frames` is fixed rather than derived from a target fps, because this model
    was trained on a fixed frame budget: sampling 58 frames just because the clip
    is 58 seconds long would be off-recipe. Clip length is the variable in the
    degradation study; the frame budget is a property of the model. That means
    `run_eval.py`'s `--fps` / `--max-frames` don't apply here and will be logged as
    ignored, which is the intended, visible behaviour.
    """

    CHECKPOINT: str = "llava-hf/LLaVA-NeXT-Video-7B-hf"
    max_batch_size = 1

    def __init__(
        self,
        num_frames: int = 32,
        max_new_tokens: int = 32,
        dtype=torch.bfloat16,
        max_batch_size: int = 1,
        question_first: bool = False,
    ):
        self.num_frames = num_frames
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.question_first = question_first

        self.model = self.attn = None
        for attn in ("flash_attention_2", "sdpa"):
            try:
                self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                    self.CHECKPOINT,
                    device_map="auto",
                    dtype=dtype,
                    attn_implementation=attn,
                )
                self.attn = attn
                break
            except (ImportError, ValueError) as e:
                logger.warning("couldn't load %s with %s: %s", self.CHECKPOINT, attn, e)
                continue
        if self.model is None:
            raise RuntimeError(f"could not load {self.CHECKPOINT} with any attention backend")

        self.model.eval()
        gen_config = self.model.generation_config
        gen_config.do_sample = False
        gen_config.temperature = gen_config.top_p = gen_config.top_k = None

        self.processor = AutoProcessor.from_pretrained(self.CHECKPOINT)
        # Left padding: generation after right-side pad tokens returns garbage, and a
        # uniform prompt length is what makes the slice in `answer` valid.
        self.processor.tokenizer.padding_side = "left"

    def prepare_clip(self, video_path, transcript):
        frames = sample_frames(video_path, self.num_frames)
        # Logged so a run's methods section can state what each model actually saw.
        logger.debug(
            "%s: sampled %d frames at %dx%d", video_path, len(frames), *frames.shape[1:3]
        )
        return {"frames": frames, "transcript": transcript.strip()}

    @torch.inference_mode()
    def answer(self, clip: dict, rows: list[dict], use_transcript: bool = True) -> list[str]:
        assert len(rows) <= self.max_batch_size, f"{len(rows)} rows > cap {self.max_batch_size}"

        texts = []
        for row in rows:
            transcript = clip["transcript"] if use_transcript else ""
            # A bare `{"type": "video"}` placeholder: the frames travel via `videos=`.
            content = [
                {"type": "video"} if kind == "video" else {"type": "text", "text": value}
                for kind, value in self.content_parts(row, transcript)
            ]
            texts.append(
                self.processor.apply_chat_template(
                    [{"role": "user", "content": content}], add_generation_prompt=True
                )
            )

        inputs = self.processor(
            text=texts,
            videos=[clip["frames"]] * len(texts),
            do_sample_frames=False,  # already sampled; don't let the processor resample
            padding=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=self.dtype)

        outputs = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        gen_ids = outputs[:, inputs["input_ids"].shape[1] :].cpu()
        return self.processor.batch_decode(gen_ids, skip_special_tokens=True)
