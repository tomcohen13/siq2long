"""InternVL3-8B via the transformers-native (`-hf`) checkpoint."""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from data.utils import get_duration
from vlm.base import VLM


class InternVL3_8B(VLM):
    """
    InternVL3-8B. One question per forward pass.

    Frame sampling is pinned explicitly. `InternVLVideoProcessor.do_sample_frames`
    defaults to False for backwards compatibility and `apply_chat_template` only
    turns it on when `fps` or `num_frames` is passed, so leaving it unset would
    quietly feed this model a different frame budget than the Qwen backends and
    make any gap in the results table unattributable.
    """

    CHECKPOINT: str = "OpenGVLab/InternVL3-8B-hf"
    max_batch_size = 1

    def __init__(
        self,
        fps: float = 1.0,
        max_frames: int = 128,
        max_new_tokens: int = 32,
        dtype=torch.bfloat16,
        load_in_4bit: bool = False,
        max_batch_size: int = 1,
    ):
        self.fps = fps
        self.max_frames = max_frames
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.max_batch_size = max_batch_size

        # bf16 by default, unlike the HF examples: they quantize to fit small GPUs,
        # but running this model in NF4 while Qwen and VideoLLaMA3 run bf16 makes a
        # quantization artifact indistinguishable from a model difference. ~16GB in
        # bf16 fits an 80GB card alongside the video activations.
        quantization_config = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
            if load_in_4bit
            else None
        )

        self.model = self.attn = None
        for attn in ("flash_attention_2", "sdpa"):
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.CHECKPOINT,
                    device_map="auto",
                    dtype=dtype,
                    attn_implementation=attn,
                    quantization_config=quantization_config,
                )
                self.attn = attn
                break
            except (ImportError, ValueError) as e:
                print(f"couldn't load {self.CHECKPOINT} with {attn}: {e}")
                continue
        if self.model is None:
            raise RuntimeError(f"could not load {self.CHECKPOINT} with any attention backend")

        self.model.eval()
        gen_config = self.model.generation_config
        gen_config.do_sample = False
        gen_config.temperature = gen_config.top_p = gen_config.top_k = None

        self.processor = AutoProcessor.from_pretrained(self.CHECKPOINT)
        # Left padding: generation after right-side pad tokens returns garbage, and
        # a uniform prompt length is what makes the slice in `answer` valid.
        self.processor.tokenizer.padding_side = "left"

    def prepare_clip(self, video_path, transcript):
        """
        Path, transcript, and a resolved frame count. The processor decodes per call.

        The count is derived here rather than passing `fps` straight through:
        `sample_frames` treats `num_frames` and `fps` as mutually exclusive, and
        `num_frames` wins whenever the processor config sets it. Deriving it from
        the duration keeps both the target rate and the `max_frames` cap, and stays
        under the clip's own frame total -- a fixed count above that raises, which
        SIQ2-Long's short tail chunks would hit.
        """
        duration = get_duration(video_path)
        num_frames = max(1, min(self.max_frames, int(duration * self.fps)))
        return {
            "video_path": video_path,
            "transcript": transcript.strip(),
            "num_frames": num_frames,
        }

    def _content(self, clip: dict, row: dict, use_transcript: bool) -> list[dict]:
        content = [{"type": "video", "path": clip["video_path"]}]
        if use_transcript:
            content.append({"type": "text", "text": f"Transcript:\n{clip['transcript']}"})
        content.append({"type": "text", "text": self.render_question(row)})
        return content

    @torch.inference_mode()
    def answer(self, clip: dict, rows: list[dict], use_transcript: bool = True) -> list[str]:
        """
        B questions on one clip -> one left-padded batch.

        Batching here buys GPU utilization only: `apply_chat_template` reloads the
        video once per conversation, so neither decode nor vision encoding is shared
        across the batch. Memory scales with B x the clip's token count, and this
        model tiles frames (~256 tokens each), so a 128-frame clip is ~33k tokens
        per sequence -- hence the cap defaults to 1. Raise it once you've measured
        headroom for the clip lengths you're running.
        """
        assert len(rows) <= self.max_batch_size, f"{len(rows)} rows > cap {self.max_batch_size}"

        conversations = [
            [{"role": "user", "content": self._content(clip, row, use_transcript)}]
            for row in rows
        ]
        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            num_frames=clip["num_frames"],  # also flips do_sample_frames to True
        ).to(self.model.device, dtype=self.dtype)

        outputs = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        gen_ids = outputs[:, inputs["input_ids"].shape[1] :].cpu()
        return self.processor.batch_decode(gen_ids, skip_special_tokens=True)
