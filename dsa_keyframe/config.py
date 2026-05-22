"""
全局配置项
"""

from dataclasses import dataclass


@dataclass
class Config:
    # 模型配置
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"  # 可选 2B / 4B / 8B / 32B / 30B-A3B / 235B-A22B
    dtype: str = "auto"                              # 推荐 "auto"，自动选择 bf16/fp16

    # 抽帧配置
    sample_interval: int = 1                         # 抽帧间隔（1=每帧都处理）

    # 滑动窗口配置
    window_size: int = 5                             # 滑动窗口大小
    window_stride: int = 2                           # 滑动步长

    # 输出配置
    top_k: int = 3                                   # 最终输出关键帧数量
    output_dir: str = "./keyframes_output"           # 关键帧保存目录

    # 推理配置
    max_new_tokens: int = 20                         # 模型最大生成 token 数
    score_mode: bool = False                         # True=逐帧打分模式, False=多帧对比模式（默认）

    # 图像分辨率控制（视显存大小调整）
    min_pixels: int = 256 * 32 * 32                  # 最小像素数
    max_pixels: int = 1280 * 32 * 32                 # 最大像素数
