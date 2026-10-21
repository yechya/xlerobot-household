# 已验证运行环境 (Verified Environment)

以下版本来自本机实测（2026-10-21），用于复现实验与 TensorRT 编译。换平台、换 JetPack 或换
TensorRT 版本后必须重新验证，不能假定二进制兼容。

## 边缘端（真机推理）

| 组件 | 实测版本 |
|------|----------|
| 设备 | NVIDIA Jetson AGX Orin Developer Kit (aarch64) |
| L4T / JetPack | R36.4.7（JetPack 6.x 系列） |
| CUDA Toolkit | 12.6 (V12.6.68) |
| TensorRT | 10.3.0 (`trtexec` v100300) |
| Python | 3.10.12 |
| PyTorch | 2.3.0（CUDA 12.4 构建） |
| TorchVision | 0.18.0a0+6043bc2 |
| LeRobot | 0.3.3 |
| Transformers | 4.51.3 |
| ONNX / ONNX Runtime | 1.21.0 / 1.23.2 |
| NumPy | 1.26.0 |
| safetensors | 0.7.0 |
| accelerate | 1.12.0 |
| huggingface-hub | 0.36.0 |
| pyzmq | 27.1.0 |
| OpenCV | 4.11.0.86 |

说明：

- PyTorch 为 CUDA 12.4 构建，运行在 CUDA 12.6 驱动上；不要混用其他 JetPack 版本的 wheel。
- `torch_wheels/` 与 TensorRT engine 不进入 git；需按本机 JetPack 版本自行准备或从 release 获取。
- TensorRT 10 与 conda 环境可能需要 `LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libnvinfer.so.10`；
  `lerobot_0.3.3_pzy/envs.sh` 已按存在性条件加载。
- `trtexec` 路径：`/usr/src/tensorrt/bin/trtexec`。
- Engine 与构建时的 TensorRT 版本、GPU 架构、ONNX opset、shape profile 绑定；升级后必须重建并
  重新跑数值对齐。

## 主机端（框架与测试）

`robot-agent` 只依赖 Python 标准库与 PyYAML；测试额外需要 pytest、numpy、Pillow。支持 Python 3.10+。

```bash
cd robot-agent
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

CI 在 Ubuntu 上以 Python 3.10 与 3.12 运行同一套主机侧测试；不覆盖真实硬件、Jetson 或 TensorRT。

## 训练端

训练工作区（`lerobot_train` / 3090 服务器）与边缘端环境不同，通常使用较新的 CUDA/PyTorch。
训练产出的 `pretrained_model` 目录包含 `config.json`、`model.safetensors`、pre/postprocessor 与
`train_config.json`；部署端必须保持相同的输入特征名、相机顺序、动作维度和归一化统计。

## 版本采集

运行以下脚本生成当前机器的版本快照（用于 issue 与复现记录）：

```bash
PYTHON=~/miniconda3/envs/lerobot_pzy/bin/python bash scripts/capture_environment.sh
```
