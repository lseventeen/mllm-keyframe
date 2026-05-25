"""
帧处理工具：图像读取、DSA 预处理、关键帧保存
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tifffile
from PIL import Image


class FrameProcessor:
    @staticmethod
    def _to_tensor(image: np.ndarray) -> torch.Tensor:
        """将单帧 numpy 图像转为 float32 tensor [H, W, C]（3 通道）"""
        img_np = image
        if img_np.ndim == 2:
            img_np = img_np[:, :, None]
        if img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)
        return torch.from_numpy(img_np.astype(np.float32))

    @staticmethod
    def _log_transform_tensor(image: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(image, min=1e-6))

    @staticmethod
    def _normalize_to_uint8(image: torch.Tensor) -> torch.Tensor:
        """逐帧归一化：支持单帧 [H,W,C] 和批次 [N,H,W,C]"""
        if image.ndim == 3:
            min_val = image.amin()
            max_val = image.amax()
            if max_val > min_val:
                return (image - min_val) / (max_val - min_val) * 255.0
            return torch.zeros_like(image)
        # 批次：逐帧独立归一化
        min_val = image.amin(dim=(1, 2, 3), keepdim=True)
        max_val = image.amax(dim=(1, 2, 3), keepdim=True)
        mask = (max_val > min_val).squeeze()
        result = torch.zeros_like(image)
        if mask.any():
            result[mask] = (
                (image[mask] - min_val[mask]) / (max_val[mask] - min_val[mask]) * 255.0
            )
        return result

    @staticmethod
    def _batch_resize_long_edge(images: torch.Tensor, long_edge: int | None) -> torch.Tensor:
        """批量 Resize [N,H,W,C]；所有帧尺寸相同时一次完成"""
        if long_edge is None or long_edge <= 0:
            return images

        h, w = int(images.shape[1]), int(images.shape[2])
        max_edge = max(w, h)
        if max_edge == long_edge:
            return images

        scale = long_edge / float(max_edge)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        # [N,H,W,C] -> [N,C,H,W]
        nchw = images.permute(0, 3, 1, 2).float()
        resized = F.interpolate(nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return resized.permute(0, 2, 3, 1)

    @staticmethod
    def process_dsa_sequence(
        frames: list[np.ndarray],
        indices: list[int],
        base_frame_number: int = 3,
        resize_long_edge: int | None = None,
        save_dir: str | None = None
    ) -> tuple[list[np.ndarray], list[int]]:
        """
        DSA 序列预处理（批量 GPU/CPU 加速）：
        - 对每帧做 log 变换
        - 以第 base_frame_number 帧为参考做减影，仅保留之后的帧
        - 逐帧独立归一化到 uint8
        - 按长边 resize，保持宽高比
        - 可选保存各处理步骤的中间图像
        """
        if not frames:
            return [], []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base_idx = base_frame_number - 1
        if base_idx < 0 or base_idx >= len(frames):
            raise ValueError(f"基准帧序号超出范围: {base_frame_number}")
        if base_idx + 1 >= len(frames):
            raise ValueError("序列过短，无法生成减影结果")

        # 基准帧 log 变换
        base_tensor = FrameProcessor._to_tensor(frames[base_idx]).to(device)
        base_log = FrameProcessor._log_transform_tensor(base_tensor)  # [H, W, C]

        frames_to_process = frames[base_idx + 1:]
        processed_indices = indices[base_idx + 1:]

        # ---- 批量处理：stack -> log -> diff -> normalize -> resize ----
        # 校验所有帧尺寸一致（DSA 序列同一设备采集，尺寸应相同）
        ref_shape = frames_to_process[0].shape[:2]
        if any(f.shape[:2] != ref_shape for f in frames_to_process[1:]):
            raise ValueError(
                f"帧尺寸不一致，无法批量处理。首帧 HxW={ref_shape}，"
                f"请确认所有帧来自同一 DSA 采集序列。"
            )

        batch = torch.stack(
            [FrameProcessor._to_tensor(f).to(device) for f in frames_to_process]
        )  # [N, H, W, C]

        log_batch = FrameProcessor._log_transform_tensor(batch)           # [N, H, W, C]
        diff_batch = log_batch - base_log.unsqueeze(0)                    # broadcast [N, H, W, C]
        diff_norm_batch = FrameProcessor._normalize_to_uint8(diff_batch)  # [N, H, W, C]
        resized_batch = FrameProcessor._batch_resize_long_edge(diff_norm_batch, resize_long_edge)
        # 转回 CPU numpy
        processed_np = resized_batch.byte().cpu().numpy()  # [N, H', W', C]
        processed_frames = [processed_np[i] for i in range(len(frames_to_process))]

        # ---- 可选：保存各步骤中间结果 ----
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            step_dirs = {
                "log": os.path.join(save_dir, "step_log"),
                "diff_norm": os.path.join(save_dir, "step_diff_norm"),
                "resized": os.path.join(save_dir, "step_resized"),
            }
            for path in step_dirs.values():
                os.makedirs(path, exist_ok=True)

            # 保存基准帧 log 可视化
            base_log_vis = FrameProcessor._normalize_to_uint8(base_log)
            Image.fromarray(base_log_vis.byte().cpu().numpy()).save(
                os.path.join(step_dirs["log"], f"base_log_frame{indices[base_idx]}.png")
            )

            log_norm_batch = FrameProcessor._normalize_to_uint8(log_batch)  # [N, H, W, C]
            log_norm_np = log_norm_batch.byte().cpu().numpy()
            diff_norm_np = diff_norm_batch.byte().cpu().numpy()

            for i, frame_idx in enumerate(processed_indices):
                Image.fromarray(log_norm_np[i]).save(
                    os.path.join(step_dirs["log"], f"log_frame{frame_idx}.png")
                )
                Image.fromarray(diff_norm_np[i]).save(
                    os.path.join(step_dirs["diff_norm"], f"diff_norm_frame{frame_idx}.png")
                )
                Image.fromarray(processed_frames[i]).save(
                    os.path.join(step_dirs["resized"], f"resized_frame{frame_idx}.png")
                )

            for frame_idx, frame in zip(processed_indices, processed_frames):
                path = os.path.join(save_dir, f"processed_frame{frame_idx}.png")
                Image.fromarray(frame).save(path)
            print(f"[INFO] 预处理后图像已保存: {save_dir}")

        return processed_frames, processed_indices

    @staticmethod
    def extract_frames_from_folder(
        folder_path: str,
        extensions: tuple = (".tif", ".tiff")
    ) -> tuple[list[np.ndarray], list[int]]:
        """
        从图像文件夹中读取帧（按文件名排序）
        Args:
            folder_path: 图像文件夹路径
            extensions: 支持的图像扩展名
        Returns:
            (frames, indices): numpy 图像列表 和 对应帧索引列表
        """
        folder = Path(folder_path)
        image_files = sorted([
            f for f in folder.iterdir()
            if f.suffix.lower() in extensions
        ])
        if not image_files:
            raise FileNotFoundError(f"文件夹中无图像文件: {folder_path}")

        frames = [tifffile.imread(str(f)) for f in image_files]
        indices = list(range(len(frames)))
        print(f"[INFO] 从文件夹加载帧数: {len(frames)}")
        return frames, indices

    @staticmethod
    def extract_frames_from_tiff(
        tiff_path: str,
        sample_interval: int = 1
    ) -> tuple[list[np.ndarray], list[int]]:
        """
        从多页 TIFF 文件中读取帧
        Args:
            tiff_path: TIFF 文件路径
            sample_interval: 抽帧间隔，1 表示每帧都取
        Returns:
            (frames, indices): numpy 图像列表 和 对应帧索引列表
        """
        video = tifffile.imread(tiff_path)
        if video.ndim == 2:
            video = video[np.newaxis, ...]
        if video.ndim != 3:
            raise ValueError(f"TIFF 维度不支持: {video.shape}")

        frames, indices = [], []
        for frame_idx, frame in enumerate(video):
            if frame_idx % sample_interval == 0:
                frames.append(frame)
                indices.append(frame_idx)

        if not frames:
            raise ValueError(f"TIFF 中未读取到任何帧: {tiff_path}")

        print(f"[INFO] 从 TIFF 加载帧数: {len(frames)}")
        return frames, indices

    @staticmethod
    def save_keyframes(
        frames: list[np.ndarray],
        indices: list[int],
        output_dir: str
    ):
        """
        保存关键帧到本地
        Args:
            frames: 关键帧 numpy 图像列表
            indices: 对应帧索引
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        for rank, (frame_idx, frame) in enumerate(zip(indices, frames)):
            path = os.path.join(output_dir, f"keyframe_rank{rank+1}_frame{frame_idx}.png")
            Image.fromarray(frame).save(path)
            print(f"[INFO] 已保存关键帧: {path}")
