"""
结果可视化：生成关键帧对比图（TIFF）
"""

import os

import numpy as np
import tifffile

from ..processors.frame_processor import FrameProcessor


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return np.stack([frame] * 3, axis=-1)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return np.repeat(frame, 3, axis=2)
    return frame


def _add_red_border(frame: np.ndarray, thickness: int = 3) -> np.ndarray:
    bordered = frame.copy()
    h, w = bordered.shape[:2]
    t = min(thickness, h // 2, w // 2)
    bordered[:t, :, :] = [255, 0, 0]
    bordered[-t:, :, :] = [255, 0, 0]
    bordered[:, :t, :] = [255, 0, 0]
    bordered[:, -t:, :] = [255, 0, 0]
    return bordered


def visualize_results(
    all_frames: list[np.ndarray],
    all_indices: list[int],
    keyframe_indices: list[int],
    output_dir: str
):
    """
    保存关键帧并生成可视化对比图
    Args:
        all_frames: 所有帧的 numpy 图像列表
        all_indices: 对应帧索引列表
        keyframe_indices: 关键帧索引列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    frame_map = {idx: frame for idx, frame in zip(all_indices, all_frames)}

    # 1. 保存单独关键帧
    keyframes = [frame_map[i] for i in keyframe_indices if i in frame_map]
    FrameProcessor.save_keyframes(keyframes, keyframe_indices, output_dir)

    # 2. 生成横向对比图
    panels = []
    for frame_idx in keyframe_indices:
        if frame_idx not in frame_map:
            continue
        rgb = _to_rgb(frame_map[frame_idx]).astype(np.uint8)
        panels.append(_add_red_border(rgb))

    if not panels:
        return

    # 若各帧高度不同，先 padding 到相同高度（底部补零）
    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        h = p.shape[0]
        if h < max_h:
            pad = np.zeros((max_h - h, p.shape[1], p.shape[2]), dtype=p.dtype)
            p = np.concatenate([p, pad], axis=0)
        padded.append(p)

    viz = np.concatenate(padded, axis=1)
    viz_path = os.path.join(output_dir, "keyframes_visualization.tiff")
    tifffile.imwrite(viz_path, viz)
    print(f"[INFO] 可视化结果已保存: {viz_path}")
