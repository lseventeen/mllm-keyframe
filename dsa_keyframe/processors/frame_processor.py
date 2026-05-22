"""
帧处理工具：视频/图像读取、DSA 预处理、关键帧保存
"""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path


class FrameProcessor:

    @staticmethod
    def extract_frames_from_video(
        video_path: str,
        sample_interval: int = 1
    ) -> tuple[list[Image.Image], list[int]]:
        """
        从视频文件中抽取帧
        Args:
            video_path: 视频文件路径
            sample_interval: 抽帧间隔，1 表示每帧都取
        Returns:
            (frames, indices): PIL 图像列表 和 对应帧索引列表
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频文件: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] 视频信息: 总帧数={total_frames}, FPS={fps:.1f}")

        frames, indices = [], []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
                indices.append(frame_idx)
            frame_idx += 1

        cap.release()
        print(f"[INFO] 抽取帧数: {len(frames)}")
        return frames, indices

    @staticmethod
    def extract_frames_from_folder(
        folder_path: str,
        extensions: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    ) -> tuple[list[Image.Image], list[int]]:
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

        frames = [Image.open(f).convert("RGB") for f in image_files]
        indices = list(range(len(frames)))
        print(f"[INFO] 从文件夹加载帧数: {len(frames)}")
        return frames, indices

    @staticmethod
    def preprocess_dsa_frame(image: Image.Image) -> Image.Image:
        """
        DSA 图像预处理：
        - 转灰度
        - CLAHE 对比度增强（让血管更清晰）
        - 转回 RGB（模型输入要求）
        """
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))

    @staticmethod
    def save_keyframes(
        frames: list[Image.Image],
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
            path = os.path.join(output_dir, f"keyframe_rank{rank+1}_frame{frame_idx}.png")
            frame.save(path)
            print(f"[INFO] 已保存关键帧: {path}")
