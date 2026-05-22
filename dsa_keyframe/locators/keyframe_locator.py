"""
DSA 关键帧定位核心逻辑
- 方案一：逐帧评分（score_mode=True）
- 方案二：滑动窗口多帧对比 + 投票（默认）
"""

import re
from PIL import Image

from ..config import Config
from ..models.qwen3vl import Qwen3VLModel
from ..processors.frame_processor import FrameProcessor


class DSAKeyframeLocator:
    def __init__(self, model: Qwen3VLModel, config: Config):
        self.model = model
        self.config = config

    # ----------------------------------------------------------
    # 方案一：逐帧评分
    # ----------------------------------------------------------
    def score_single_frame(self, image: Image.Image) -> float:
        """
        对单帧 DSA 图像进行质量评分（0~10）
        评分依据：造影剂充盈程度 + 血管清晰度
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            "This is a single frame from a DSA (Digital Subtraction Angiography) sequence.\n"
                            "Please evaluate this frame based on:\n"
                            "1. Contrast agent filling: how completely the vessels are filled.\n"
                            "2. Vessel visibility: how clearly the vessel boundaries are visible.\n"
                            "Rate from 0 to 10 (integer). 10=peak filling, clearly visible. 0=no contrast.\n"
                            "Output ONLY the integer score."
                        )
                    }
                ]
            }
        ]
        result = self.model.generate(messages)
        match = re.search(r"\d+(\.\d+)?", result)
        return min(max(float(match.group()), 0.0), 10.0) if match else 0.0

    def locate_by_scoring(
        self,
        frames: list[Image.Image],
        indices: list[int]
    ) -> list[int]:
        """逐帧打分，返回 top_k 关键帧索引（按时间顺序）"""
        print(f"[INFO] 开始逐帧评分，共 {len(frames)} 帧...")
        scores = []
        for frame, idx in zip(frames, indices):
            processed = FrameProcessor.preprocess_dsa_frame(frame)
            score = self.score_single_frame(processed)
            scores.append((idx, score))
            print(f"  帧 {idx:4d} -> 得分: {score:.1f}")

        scores.sort(key=lambda x: -x[1])
        return sorted([idx for idx, _ in scores[:self.config.top_k]])

    # ----------------------------------------------------------
    # 方案二：多帧对比（默认）
    # ----------------------------------------------------------
    def compare_window(
        self,
        window_frames: list[Image.Image],
        window_indices: list[int]
    ) -> int:
        """
        在滑动窗口内，让模型选出最佳关键帧
        Returns: 该帧的原始帧索引
        """
        n = len(window_frames)
        content = []
        for i, img in enumerate(window_frames):
            content.append({"type": "text", "text": f"[Frame {i+1}]"})
            content.append({"type": "image", "image": img})

        content.append({
            "type": "text",
            "text": (
                f"Above are {n} consecutive DSA (Digital Subtraction Angiography) frames.\n"
                "Identify the KEY FRAME defined as:\n"
                "  - Peak contrast agent filling in the target vessel\n"
                "  - Clearest vessel structure and boundaries\n"
                "  - Best diagnostic value\n"
                f"Reply with ONLY a number from 1 to {n}. No explanation."
            )
        })

        messages = [{"role": "user", "content": content}]
        result = self.model.generate(messages)

        match = re.search(r"\d+", result)
        if match:
            chosen = max(0, min(int(match.group()) - 1, n - 1))
            return window_indices[chosen]
        return window_indices[n // 2]  # 默认返回窗口中间帧

    def locate_by_comparison(
        self,
        frames: list[Image.Image],
        indices: list[int]
    ) -> list[int]:
        """滑动窗口多帧对比 + 投票，返回 top_k 关键帧索引"""
        print(f"[INFO] 开始滑动窗口对比，共 {len(frames)} 帧，"
              f"窗口={self.config.window_size}，步长={self.config.window_stride}")

        vote_count: dict[int, int] = {}
        ws = self.config.window_size
        stride = self.config.window_stride
        total_windows = max(1, (len(frames) - ws) // stride + 1)

        for w_idx, i in enumerate(range(0, max(1, len(frames) - ws + 1), stride)):
            window_frames = [
                FrameProcessor.preprocess_dsa_frame(f)
                for f in frames[i:i + ws]
            ]
            window_indices = indices[i:i + ws]
            best_idx = self.compare_window(window_frames, window_indices)
            vote_count[best_idx] = vote_count.get(best_idx, 0) + 1
            print(f"  窗口 [{w_idx+1}/{total_windows}] "
                  f"帧范围: {window_indices[0]}~{window_indices[-1]} "
                  f"-> 最佳帧: {best_idx} (票数: {vote_count[best_idx]})")

        sorted_candidates = sorted(vote_count.items(), key=lambda x: -x[1])
        return sorted([idx for idx, _ in sorted_candidates[:self.config.top_k]])

    # ----------------------------------------------------------
    # 统一入口
    # ----------------------------------------------------------
    def locate(
        self,
        frames: list[Image.Image],
        indices: list[int]
    ) -> list[int]:
        """根据 config.score_mode 选择定位策略"""
        if self.config.score_mode:
            return self.locate_by_scoring(frames, indices)
        return self.locate_by_comparison(frames, indices)
