# LeRobot π0.5 (pi05) Jetson Integration Patch

本目录把 `lerobot_ycy` 工作区相对上游的 π0.5 定制改动整理成可审阅补丁，避免发布完整 fork、
旧历史和大文件。

- 上游：`https://github.com/huggingface/lerobot.git`
- 基线提交：`473f1bd0e`（`docs: improve assets (#2777)`）
- 补丁：`patches/0001-pi05-jetson-integration.patch`（41 个文件，约 5.8k 行）
- 校验和：见 `manifest.json`

> 这是与 `integrations/lerobot_jetson/` **相互独立**的工作区补丁。两者基于不同的上游提交，
> 请分别应用到独立的 LeRobot checkout，不要叠加应用。

## 包含内容

- π0.5 的 ONNX 导出、BF16/INT8 TensorRT engine 构建、量化校准与数值校验脚本；
- π0.5 policy 的 TensorRT 专家解码器与配置改动；
- 数据集/可视化/录制相关的必要修改；
- `xlerobot` host/client 与底盘配置（ycy 版本）；
- `envs.sh`、依赖清单与运行脚本。

## 不包含

- 训练数据、checkpoint、TensorRT engine、标定缓存；
- `examples/` 下的大量遥操作演示脚本（与 pzy 补丁重复，未纳入本补丁）；
- `thrid-party/`、`xone/` 等外部克隆；
- 开发日志 `TRT_PI05_INTEGRATION_PROGRESS.md`；
- `CLAUDE.md` 等本机代理配置。

## 应用

```bash
./apply.sh /path/to/lerobot-pi05
cd /path/to/lerobot-pi05
```

手动方式：

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout 473f1bd0e
git apply /path/to/0001-pi05-jetson-integration.patch
```

验证：

```bash
./verify.sh
```

## 安装与运行

1. 按 [已验证环境](../../docs/environment.md) 准备 JetPack/CUDA/TensorRT 与 Python。
2. 安装与 JetPack 匹配的 PyTorch/TorchVision wheel，然后 `pip install -e .`。
3. 复制 `.env.example` 为 `.env` 并填写本机代理、VLM 与相机配置。
4. `source envs.sh`（激活 `lerobot_ycy`，加载 `.env`）。
5. 按脚本头部注释设置 `POLICY_PATH`、`DATASET_PATH` 与 `LEROBOT_POLICY_ROOT`。

## 已知边界

- 权重与数据未包含；策略强过拟合，不能直接跨硬件使用。
- TensorRT engine 与 JetPack/TensorRT 版本绑定；换版本需重新导出与编译。
- 补丁不保证与上游后续版本兼容；升级上游需重新基线化。
- 真实动作必须保留物理急停与看门狗；这不是认证安全系统。

## 许可证

上游 LeRobot 与本补丁均按 Apache-2.0 分发；上游版权与 NOTICE 要求继续适用。
