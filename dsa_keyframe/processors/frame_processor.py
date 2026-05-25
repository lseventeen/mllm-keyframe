"""
帧处理工具：图像读取、DSA 预处理、关键帧保存
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tifffile


class FrameProcessor:
    @staticmethod
    def _to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
        img_np = image
        if img_np.ndim == 2:
            img_np = img_np[:, :, None]
        if img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)
        img_np = img_np.astype(np.float32)
        return torch.from_numpy(img_np).to(device)

    @staticmethod
    def _log_transform_tensor(image: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(image, min=1e-6))

    @staticmethod
    def _normalize_to_uint8(image: torch.Tensor) -> torch.Tensor:
        min_val = torch.amin(image)
        max_val = torch.amax(image)
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val) * 255.0
        else:
            image = torch.zeros_like(image)
        return image

    @staticmethod
    def _resize_long_edge_tensor(image: torch.Tensor, long_edge: int | None) -> torch.Tensor:
        if long_edge is None or long_edge <= 0:
            return image

        h, w = int(image.shape[0]), int(image.shape[1])
        max_edge = max(w, h)
        if max_edge == long_edge:
            return image

        scale = long_edge / float(max_edge)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        chw = image.permute(2, 0, 1).unsqueeze(0)
        resized = F.interpolate(chw, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return resized.squeeze(0).permute(1, 2, 0)

    @staticmethod
    def process_dsa_sequence(
        frames: list[np.ndarray],
        indices: list[int],
        base_frame_number: int = 3,
        resize_long_edge: int | None = None,
        save_dir: str | None = None
    ) -> tuple[list[np.ndarray], list[int]]:
        """
        DSA 序列预处理：
        - 对每帧做 log 变换
        - 以第 3 帧为参考做减影（当前帧 - 参考帧），仅保留第 4 帧及以后
        - 按长边进行 resize，保持宽高比
        - 可选保存处理后的图像
        """
        if not frames:
            return []

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，无法在 GPU 上执行预处理")

        device = torch.device("cuda")

        base_idx = base_frame_number - 1
        if base_idx < 0 or base_idx >= len(frames):
            raise ValueError(f"基准帧序号超出范围: {base_frame_number}")

        base_tensor = FrameProcessor._to_tensor(frames[base_idx], device)
        base_log = FrameProcessor._log_transform_tensor(base_tensor)
        processed_frames: list[np.ndarray] = []

        if base_idx + 1 >= len(frames):
            raise ValueError("序列过短，无法生成减影结果")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            step_dirs = {
                "log": os.path.join(save_dir, "step_log"),
                "diff_raw": os.path.join(save_dir, "step_diff_raw"),
                "diff_norm": os.path.join(save_dir, "step_diff_norm"),
                "resized": os.path.join(save_dir, "step_resized"),
            }
            for path in step_dirs.values():
                os.makedirs(path, exist_ok=True)

            base_log_vis = FrameProcessor._normalize_to_uint8(base_log)
            tifffile.imwrite(
                os.path.join(step_dirs["log"], f"base_log_frame{indices[base_idx]}.tiff"),
                base_log_vis.byte().cpu().numpy()
            )

        for i, frame in enumerate(frames[base_idx + 1:], start=base_idx + 1):
            tensor = FrameProcessor._to_tensor(frame, device)
            log_tensor = FrameProcessor._log_transform_tensor(tensor)
            diff_tensor = log_tensor - base_log
            diff_norm = FrameProcessor._normalize_to_uint8(diff_tensor)
            resized = FrameProcessor._resize_long_edge_tensor(diff_norm, resize_long_edge)
            proc_img = resized.byte().cpu().numpy()
            processed_frames.append(proc_img)

            if save_dir:
                log_vis = FrameProcessor._normalize_to_uint8(log_tensor)
                tifffile.imwrite(
                    os.path.join(step_dirs["log"], f"log_frame{indices[i]}.tiff"),
                    log_vis.byte().cpu().numpy()
                )
                diff_raw_vis = FrameProcessor._normalize_to_uint8(diff_tensor)
                tifffile.imwrite(
                    os.path.join(step_dirs["diff_raw"], f"diff_raw_frame{indices[i]}.tiff"),
                    diff_raw_vis.byte().cpu().numpy()
                )
                tifffile.imwrite(
                    os.path.join(step_dirs["diff_norm"], f"diff_norm_frame{indices[i]}.tiff"),
                    diff_norm.byte().cpu().numpy()
                )
                tifffile.imwrite(
                    os.path.join(step_dirs["resized"], f"resized_frame{indices[i]}.tiff"),
                    proc_img
                )

        processed_indices = indices[base_idx + 1:]

        if save_dir:
            for frame_idx, frame in zip(processed_indices, processed_frames):
                path = os.path.join(save_dir, f"processed_frame{frame_idx}.tiff")
                tifffile.imwrite(path, frame)
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
            (frames, indices): PIL 图像列表 和 对应帧索引列表
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
            (frames, indices): PIL 图像列表 和 对应帧索引列表
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
            frames: 关键帧 PIL 图像列表
            indices: 对应帧索引
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        for rank, (frame_idx, frame) in enumerate(zip(indices, frames)):
            path = os.path.join(output_dir, f"keyframe_rank{rank+1}_frame{frame_idx}.tiff")
            tifffile.imwrite(path, frame)
            print(f"[INFO] 已保存关键帧: {path}")
