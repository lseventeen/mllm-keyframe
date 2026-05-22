"""
DSA 关键帧定位系统 - 主入口
支持命令行调用和代码调用两种方式
"""

import argparse
from pathlib import Path

from .config import Config
from .models import Qwen3VLModel
from .locators import DSAKeyframeLocator
from .processors import FrameProcessor
from .utils import visualize_results


def run(input_path: str, config: Config = None) -> list[int]:
    """
    主流程函数
    Args:
        input_path: 视频文件路径 或 图像帧文件夹路径
        config: 配置项，None 时使用默认配置
    Returns:
        关键帧索引列表（按时间顺序）
    """
    if config is None:
        config = Config()

    # 1. 加载模型
    model = Qwen3VLModel(config)
    locator = DSAKeyframeLocator(model, config)

    # 2. 读取输入
    p = Path(input_path)
    if p.is_dir():
        frames, indices = FrameProcessor.extract_frames_from_folder(str(p))
    elif p.is_file():
        frames, indices = FrameProcessor.extract_frames_from_video(str(p), config.sample_interval)
    else:
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    if not frames:
        raise ValueError("未能加载任何帧，请检查输入路径")

    # 3. 关键帧定位
    mode_str = "逐帧评分" if config.score_mode else "多帧对比"
    print(f"\n[INFO] 开始关键帧定位，模式: {mode_str}")
    keyframe_indices = locator.locate(frames, indices)

    # 4. 打印结果
    print(f"\n{'='*50}")
    print(f"[RESULT] 关键帧定位完成！")
    print(f"[RESULT] 关键帧索引（按时间顺序）: {keyframe_indices}")
    print(f"{'='*50}\n")

    # 5. 可视化 & 保存
    visualize_results(frames, indices, keyframe_indices, config.output_dir)

    return keyframe_indices


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="DSA 关键帧定位（基于 Qwen3-VL）")
    parser.add_argument("--input",       type=str, required=True,
                        help="输入路径：视频文件或图像帧文件夹")
    parser.add_argument("--model",       type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="模型名称或本地权重路径")
    parser.add_argument("--top_k",       type=int, default=3,
                        help="输出关键帧数量（默认 3）")
    parser.add_argument("--interval",    type=int, default=1,
                        help="抽帧间隔，仅视频输入有效（默认 1=每帧）")
    parser.add_argument("--window_size", type=int, default=5,
                        help="滑动窗口大小（默认 5）")
    parser.add_argument("--output_dir",  type=str, default="./keyframes_output",
                        help="输出目录（默认 ./keyframes_output）")
    parser.add_argument("--score_mode",  action="store_true",
                        help="启用逐帧评分模式（默认为多帧对比模式）")
    args = parser.parse_args()

    config = Config(
        model_name=args.model,
        sample_interval=args.interval,
        window_size=args.window_size,
        top_k=args.top_k,
        output_dir=args.output_dir,
        score_mode=args.score_mode,
    )
    run(input_path=args.input, config=config)


if __name__ == "__main__":
    main()
