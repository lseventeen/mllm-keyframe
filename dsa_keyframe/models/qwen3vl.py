"""
Qwen3-VL 模型加载与推理封装
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from ..config import Config


class Qwen3VLModel:
    def __init__(self, config: Config):
        self.config = config
        print(f"[INFO] 正在加载 Qwen3-VL 模型: {config.model_name} ...")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.model_name,
            dtype=config.dtype,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(config.model_name)

        # 设置图像分辨率预算
        self.processor.image_processor.size = {
            "longest_edge": config.max_pixels,
            "shortest_edge": config.min_pixels
        }

        self.model.eval()
        print(f"[INFO] 模型加载完成")

    def generate(self, messages: list) -> str:
        """
        统一推理接口
        使用 process_vision_info 处理图像（image_patch_size=16 为 Qwen3-VL 必须参数）
        """
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Qwen3-VL 专用：image_patch_size=16，同时处理视频元数据
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True
        )

        # 解包视频元数据
        video_metadatas = None
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)

        # do_resize=False：qwen-vl-utils 已完成 resize，避免重复操作
        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs
        ).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()
