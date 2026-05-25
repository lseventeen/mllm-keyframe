"""
Qwen3-VL 模型加载与推理封装
"""

import numpy as np
import torch
from PIL import Image as PILImage
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

    @staticmethod
    def _convert_images_in_messages(messages: list) -> list:
        """
        将消息内容中的 numpy.ndarray 图像转换为 PIL.Image，
        因为 qwen_vl_utils.fetch_image 不接受 numpy.ndarray。
        仅对含有 numpy 图像的 part 做浅拷贝，不修改原始输入。
        """
        result = []
        for message in messages:
            new_parts = []
            for part in message.get("content", []):
                if part.get("type") == "image" and isinstance(part.get("image"), np.ndarray):
                    arr = part["image"]

                    # 规范化到 uint8
                    if arr.dtype == np.float32 or arr.dtype == np.float64:
                        # 浮点数组：若值域在 [0, 1] 则缩放，否则按实际范围归一化
                        if arr.max() <= 1.0 and arr.min() >= 0.0:
                            arr = (arr * 255).round().astype(np.uint8)
                        else:
                            low, high = arr.min(), arr.max()
                            arr = ((arr - low) / (high - low + 1e-8) * 255).round().astype(np.uint8)
                    elif arr.dtype != np.uint8:
                        # 整数类型（如 uint16）：线性映射到 [0, 255]
                        low, high = arr.min(), arr.max()
                        arr = ((arr - low) / (high - low + 1e-8) * 255).round().astype(np.uint8)

                    # 转为 3 通道 RGB
                    if arr.ndim == 2:
                        arr = np.stack([arr] * 3, axis=-1)
                    elif arr.ndim == 3 and arr.shape[2] == 1:
                        arr = np.repeat(arr, 3, axis=2)
                    elif arr.ndim == 3 and arr.shape[2] == 4:
                        arr = arr[:, :, :3]  # RGBA -> RGB，丢弃 alpha 通道
                    elif arr.ndim == 3 and arr.shape[2] != 3:
                        raise ValueError(
                            f"不支持的图像通道数 {arr.shape[2]}，期望 1、3 或 4 通道"
                        )

                    try:
                        pil_img = PILImage.fromarray(arr)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"无法将 numpy 数组（shape={arr.shape}, dtype={arr.dtype}）"
                            f"转换为 PIL 图像：{exc}"
                        ) from exc

                    new_parts.append({**part, "image": pil_img})
                else:
                    new_parts.append(part)
            result.append({**message, "content": new_parts})
        return result

    def generate(self, messages: list) -> str:
        """
        统一推理接口
        使用 process_vision_info 处理图像（image_patch_size=16 为 Qwen3-VL 必须参数）
        """
        messages = self._convert_images_in_messages(messages)
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
