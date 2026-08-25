import os
import torch

from enum import StrEnum
from transformers import AutoProcessor

os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchcodec"  # must precede the import below
from qwen_vl_utils import process_vision_info

from src.models.vlm import VLM

class QwenModels(StrEnum):
    Qwen25VL7BInstruct = "Qwen/Qwen2.5-VL-7B-Instruct"
    Qwen3VL8bInstruct = "Qwen/Qwen3-VL-8B-Instruct"


class QwenVLM(VLM):

    CHECKPOINT: str
    return_video_metadata: bool
    do_resize: bool

    @staticmethod
    def _model_cls():
        raise NotImplementedError

    def _load(self):

        model_cls = self._model_cls()
        for attn in ("flash_attention_2", "sdpa"):
            try:
                self.model = model_cls.from_pretrained(
                    self.CHECKPOINT, device_map="auto",
                    torch_dtype=self.dtype, attn_implementation=attn,
                )
                self.attn = attn
                break
            except (ImportError, ValueError) as e:
                print(f"couldn't load {self.CHECKPOINT}: {e}")
                continue

        self.model.eval()
        gen_config = self.model.generation_config
        gen_config.do_sample = False
        gen_config.temperature = gen_config.top_p = gen_config.top_k = None

        self.processor = AutoProcessor.from_pretrained(self.CHECKPOINT)
        self.processor.tokenizer.padding_side = "left"

    def __init__(
        self,
        fps: float = 1.0,
        max_frames: int = 128,
        max_new_tokens: int = 32,
        max_pixels: int = 448 * 448,
        dtype=torch.bfloat16,
        max_batch_size: int = 8,
    ):

        self.fps = fps
        self.max_frames = max_frames
        self.max_new_tokens = max_new_tokens
        self.max_pixels = max_pixels
        self.dtype = dtype
        self.max_batch_size = max_batch_size

        self._load()

    def prepare_clip(self, video_path, transcript):
        """Decode video once; transcript rides along. Both reused across the clip's questions."""

        msg = {
            "type": "video",
            "video": f"file://{video_path}",
            "fps": self.fps,
            "max_pixels": self.max_pixels,
            "max_frames": self.max_frames,
        }
        _, video_inputs, video_kwargs = process_vision_info(
            [{"role": "user", "content": [msg]}],
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=self.return_video_metadata,
        )
        if self.return_video_metadata:
            videos, metadatas = zip(*video_inputs)
            videos, metadatas = list(videos), list(metadatas)
        else:
            videos, metadatas = video_inputs, None
        
        if isinstance(video_kwargs.get("fps"), (list, tuple)):
            video_kwargs["fps"] = video_kwargs["fps"][0]

        return {
            "msg": msg,
            "videos": videos,
            "metadatas": metadatas,
            "kwargs": video_kwargs,
            "transcript": transcript.strip()
        }

    def _encode(self, clip, rows, use_transcript=True):
        """One decoded clip + its transcript, B questions -> a single left-padded batch."""
        prefix = [clip["msg"]]
        if use_transcript:
            prefix.append({"type": "text", "text": f"Transcript:\n{clip['transcript']}"})

        texts = []
        for r in rows:
            prompt = self.render_question(r)
            texts.append(
                self.processor.apply_chat_template(
                    [{"role": "user", "content": prefix + [{"type": "text", "text": prompt}]}],
                    tokenize=False, add_generation_prompt=True,
                )
            )
        b = len(texts)
        extra = {}
        if clip["metadatas"] is not None:
            extra["video_metadata"] = clip["metadatas"] * b
        if not self.do_resize:
            extra["do_resize"] = False

        inputs = self.processor(
            text=texts,
            videos=clip["videos"] * b,
            padding=True, return_tensors="pt",
            **extra,
            **clip["kwargs"],
        )
        return inputs.to(self.model.device, dtype=self.dtype)

    @torch.inference_mode()
    def answer(self, clip: dict, rows: list[dict], use_transcript: bool = True):
        assert len(rows) <= self.max_batch_size, f"{len(rows)} rows > cap {self.max_batch_size}"
        inputs = self._encode(clip, rows, use_transcript=use_transcript)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        gen_ids = outputs[:, inputs["input_ids"].shape[1]:].cpu()
        texts = self.processor.batch_decode(gen_ids, skip_special_tokens=True)
        return texts


class Qwen25VL(QwenVLM):

    CHECKPOINT = QwenModels.Qwen25VL7BInstruct
    return_video_metadata = False
    do_resize = True

    @staticmethod
    def _model_cls():
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration


class Qwen3VL(QwenVLM):
    """Qwen3-VL-8B-Instruct"""

    CHECKPOINT = QwenModels.Qwen3VL8bInstruct
    return_video_metadata = True
    do_resize = False

    @staticmethod
    def _model_cls():
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
