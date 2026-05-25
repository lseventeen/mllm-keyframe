"""
DSA 关键帧定位核心逻辑
- 方案一：逐帧评分（score_mode=True）
- 方案二：滑动窗口多帧对比 + 投票（默认）
"""

import re
import numpy as np

from ..config import Config
from ..models.qwen3vl import Qwen3VLModel


class DSAKeyframeLocator:
    def __init__(self, model: Qwen3VLModel, config: Config):
        self.model = model
        self.config = config

    # ----------------------------------------------------------
    # 方案一：逐帧评分
    # ----------------------------------------------------------
    def score_single_frame(self, image: np.ndarray) -> float:
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
        frames: list[np.ndarray],
        indices: list[int]
    ) -> list[int]:
        """逐帧打分，返回 top_k 关键帧索引（按时间顺序）"""
        print(f"[INFO] 开始逐帧评分，共 {len(frames)} 帧...")
        scores = []
        for frame, idx in zip(frames, indices):
            score = self.score_single_frame(frame)
            scores.append((idx, score))
            print(f"  帧 {idx:4d} -> 得分: {score:.1f}")

        scores.sort(key=lambda x: -x[1])
        return sorted([idx for idx, _ in scores[:self.config.top_k]])

    # ----------------------------------------------------------
    # 方案二：多帧对比（默认）
    # ----------------------------------------------------------
    def compare_window(
        self,
        window_frames: list[np.ndarray],
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
        frames: list[np.ndarray],
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
            window_frames = frames[i:i + ws]
            window_indices = indices[i:i + ws]
            best_idx = self.compare_window(window_frames, window_indices)
            vote_count[best_idx] = vote_count.get(best_idx, 0) + 1
            print(f"  窗口 [{w_idx+1}/{total_windows}] "
                  f"帧范围: {window_indices[0]}~{window_indices[-1]} "
                  f"-> 最佳帧: {best_idx} (票数: {vote_count[best_idx]})")

        sorted_candidates = sorted(vote_count.items(), key=lambda x: -x[1])
        return sorted([idx for idx, _ in sorted_candidates[:self.config.top_k]])

    # ----------------------------------------------------------
    # 方案三：全序列一次性选帧
    # ----------------------------------------------------------
    def locate_by_global_selection(
        self,
        frames: list[np.ndarray],
        indices: list[int]
    ) -> list[int]:
        """
        将整个序列（或均匀采样后的子集）一次性送入模型，
        让模型综合对比所有帧后直接选出 top_k 关键帧。

        当帧数超过 config.global_max_frames 时，先均匀采样
        降至 global_max_frames 帧，再送入模型；选出的帧号
        会映射回原始 indices。
        """
        n_total = len(frames)
        max_f = self.config.global_max_frames

        # 均匀采样（帧数过多时）
        if n_total > max_f:
            sample_pos = [round(i * (n_total - 1) / (max_f - 1)) for i in range(max_f)]
            sel_frames = [frames[p] for p in sample_pos]
            sel_indices = [indices[p] for p in sample_pos]
            print(f"[INFO] 全序列模式：帧数 {n_total} 超过上限 {max_f}，"
                  f"均匀采样至 {max_f} 帧后送入模型")
        else:
            sel_frames = frames
            sel_indices = indices
            print(f"[INFO] 全序列模式：将全部 {n_total} 帧一次性送入模型")

        n = len(sel_frames)
        content = []
        for i, img in enumerate(sel_frames):
            content.append({"type": "text", "text": f"[Frame {i + 1}]"})
            content.append({"type": "image", "image": img})

        content.append({
            "type": "text",
            "text": (
                f"Above are all {n} frames of a DSA (Digital Subtraction Angiography) sequence "
                f"in chronological order.\n"
                f"Please select the {self.config.top_k} KEY FRAME(S) that best satisfy:\n"
                f"  1. Peak contrast agent filling in the target vessel\n"
                f"  2. Clearest vessel structure and boundaries\n"
                f"  3. Highest diagnostic value\n"
                f"Consider ALL frames together before deciding.\n"
                f"Reply with ONLY {self.config.top_k} frame number(s) from 1 to {n}, "
                f"separated by spaces or commas. No explanation."
            )
        })

        messages = [{"role": "user", "content": content}]
        result = self.model.generate(messages)
        print(f"[INFO] 模型回复: {result}")

        chosen_nums = [int(m) for m in re.findall(r"\d+", result)]
        # 去重、裁剪到有效范围，记录并丢弃越界值
        valid_set = set()
        for c in chosen_nums:
            clamped = max(1, min(c, n))
            if clamped != c:
                print(f"[WARN] 模型返回帧号 {c} 超出范围 [1, {n}]，已修正为 {clamped}")
            valid_set.add(clamped)
        valid = sorted(valid_set)
        # 取前 top_k 个（按帧号升序）
        valid = valid[:self.config.top_k]
        # 不足 top_k 时补充（从剩余帧中间位置填满）
        if len(valid) < self.config.top_k:
            fallback = [i + 1 for i in range(n) if (i + 1) not in set(valid)]
            need = self.config.top_k - len(valid)
            if fallback:
                mid = max(0, len(fallback) // 2 - need // 2)
                valid.extend(fallback[mid:mid + need])
            valid.sort()

        # 将局部帧编号映射回原始 indices
        result_indices = sorted(sel_indices[c - 1] for c in valid)
        print(f"[INFO] 全序列选帧结果（原始帧索引）: {result_indices}")
        return result_indices

    # ----------------------------------------------------------
    # 统一入口
    # ----------------------------------------------------------
    def locate(
        self,
        frames: list[np.ndarray],
        indices: list[int]
    ) -> list[int]:
        """根据 config 选择定位策略：
        - global_mode=True  → 全序列一次性选帧（方案三）
        - score_mode=True   → 逐帧打分（方案一）
        - 默认              → 滑动窗口多帧对比（方案二）
        """
        if self.config.global_mode:
            return self.locate_by_global_selection(frames, indices)
        if self.config.score_mode:
            return self.locate_by_scoring(frames, indices)
        return self.locate_by_comparison(frames, indices)
