# DSA 关键帧定位系统（Qwen3-VL）

基于 **Qwen3-VL** 多模态大模型，对 DSA（数字减影血管造影）序列自动定位造影剂充盈最佳、血管显示最清晰的关键帧。

## 目录结构

```
dsa_keyframe/
├── README.md
├── requirements.txt
├── main.py                      # 命令行入口 & run() 函数
├── config.py                    # 全局配置 Config dataclass
├── models/
│   ├── __init__.py
│   └── qwen3vl.py               # Qwen3-VL 模型加载与推理
├── processors/
│   ├── __init__.py
│   └── frame_processor.py       # 帧读取、DSA 预处理、保存
├── locators/
│   ├── __init__.py
│   └── keyframe_locator.py      # 关键帧定位（评分/对比两种策略）
└── utils/
    ├── __init__.py
    └── visualization.py         # 可视化对比图生成
```

## 安装

```bash
pip install -r requirements.txt
# 可选：更快的视频解码
pip install "qwen-vl-utils[decord]==0.0.14"
```

## 使用方法

### 命令行

```bash
# 视频输入（多帧对比模式，默认）
python -m dsa_keyframe.main --input ./dsa.mp4 --top_k 3

# 图像文件夹（逐帧评分模式）
python -m dsa_keyframe.main --input ./dsa_frames/ --score_mode

# 使用本地模型权重
python -m dsa_keyframe.main --input ./dsa.mp4 --model /path/to/Qwen3-VL-8B-Instruct
```

### 代码调用

```python
from dsa_keyframe.main import run
from dsa_keyframe.config import Config

config = Config(
    model_name="Qwen/Qwen3-VL-8B-Instruct",
    top_k=3,
    score_mode=False,
    output_dir="./results"
)
keyframe_indices = run("./dsa.mp4", config)
print("关键帧位置:", keyframe_indices)
```

## 可选模型

| 模型 | 显存需求 | 特点 |
|------|---------|------|
| `Qwen3-VL-2B-Instruct` | ~6 GB | 资源极有限时使用 |
| `Qwen3-VL-4B-Instruct` | ~10 GB | 轻量推荐 |
| **`Qwen3-VL-8B-Instruct`** | ~18 GB | **默认推荐** |
| `Qwen3-VL-32B-Instruct` | ~70 GB | 高精度需求 |
| `Qwen3-VL-30B-A3B-Instruct` | ~30 GB | MoE 架构，效率高 |

## 两种定位模式

| 模式 | 参数 | 说明 |
|------|------|------|
| 多帧对比（默认） | 不加 `--score_mode` | 滑动窗口内让模型选最佳帧，投票决定 |
| 逐帧评分 | `--score_mode` | 每帧单独打分 0~10，取最高分 |
