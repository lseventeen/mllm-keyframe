"""
结果可视化：生成关键帧对比图
"""

import os
import matplotlib.pyplot as plt
from PIL import Image

from ..processors.frame_processor import FrameProcessor


def visualize_results(
    all_frames: list[Image.Image],
    all_indices: list[int],
    keyframe_indices: list[int],
    output_dir: str
):
    """
    保存关键帧并生成可视化对比图
    Args:
        all_frames: 所有帧的 PIL 图像列表
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
    n = len(keyframe_indices)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, frame_idx in zip(axes, keyframe_indices):
        if frame_idx in frame_map:
            ax.imshow(frame_map[frame_idx], cmap="gray")
        ax.set_title(f"Key Frame\nIndex: {frame_idx}", fontsize=12, color="red")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(3)

    plt.suptitle("DSA Keyframe Localization Results (Qwen3-VL)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    viz_path = os.path.join(output_dir, "keyframes_visualization.png")
    plt.savefig(viz_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] 可视化结果已保存: {viz_path}")
