import torch

from src.models.vlm import VLM

NAME = "DAMO-NLP-SG/VideoLLaMA3-7B"
SYSTEM = "You are a helpful assistant."


class VideoLlama3(VLM):
    """VideoLLaMA3 via the DAMO remote-code path. One question per forward pass."""

    max_batch_size = 1

    def __init__(
        self,
        fps: float = 1.0,
        max_frames: int = 128,
        max_new_tokens: int = 32,
        dtype=torch.bfloat16
    ):
        from transformers import AutoModelForCausalLM, AutoProcessor

        try:
            attn = "flash_attention_2"
            self.model = AutoModelForCausalLM.from_pretrained(
                NAME, trust_remote_code=True, device_map="auto",
                torch_dtype=dtype, attn_implementation=attn,
            )
        except (ImportError, ValueError):
            attn = "sdpa"
            self.model = AutoModelForCausalLM.from_pretrained(
                NAME, trust_remote_code=True, device_map="auto",
                torch_dtype=dtype, attn_implementation=attn,
            )

        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(NAME, trust_remote_code=True)

        self.fps = fps
        self.max_frames = max_frames
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.attn = attn

    def prepare_clip(self, video_path, transcript):
        return {"video_path": video_path, "transcript": transcript}

    def answer(self, clip: dict, rows: list[dict], use_transcript: bool = True) -> list[str]:
        return [self._answer_one(clip, row, use_transcript) for row in rows]

    def _answer_one(self, clip: dict, row: dict, use_transcript: bool) -> str:
        content = [{
            "type": "video",
            "video": {
                "video_path": clip["video_path"],
                "fps": self.fps,
                "max_frames": self.max_frames,
            },
        }]
        if use_transcript:
            transcript = clip["transcript"]
            content.append({"type": "text", "text": f"Transcript:\n{transcript}"})
        content.append({"type": "text", "text": self.render_question(row)})

        conversation = [
            # {"role": "system", "content": SYSTEM},
            {"role": "user", "content": content},
        ]

        inputs = self.processor(
            conversation=conversation,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        device = self.model.device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        # This path builds inputs_embeds, so `out` is usually completion-only.
        # Slice defensively rather than assuming either convention.
        prompt_len = inputs["input_ids"].shape[1]
        if out.shape[1] > prompt_len:
            out = out[:, prompt_len:]

        return self.processor.batch_decode(out.cpu(), skip_special_tokens=True)[0]
