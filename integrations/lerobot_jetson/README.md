# LeRobot Jetson / XLeRobot Integration Patch

本目录把 `lerobot_0.3.3_pzy` 工作区相对上游的定制改动整理成**可审阅补丁**，避免发布 221 GB
的完整 fork、旧历史和大文件。

- 上游：`https://github.com/huggingface/lerobot.git`
- 基线提交：`1033680a5`（`chore: move errors to utils (#2017)`）
- 补丁：`patches/0001-xlerobot-jetson-integration.patch`（54 个文件，约 11k 行）
- 校验和：见 `manifest.json`

## 包含内容

- SmolVLA 的 TensorRT/ONNX 导出、推理后端、INT8 量化与数值校验脚本；
- `xlerobot` 双臂 + 三全向轮底盘的 robot/teleoperator 配置与驱动；
- VR、Xbox、键盘遥操作与数据录制脚本；
- 三路相机、底盘控制、数据集评测与性能分析脚本；
- Jetson 安装脚本、环境激活脚本与依赖调整。

## 不包含

- 训练数据、checkpoint、TensorRT engine、标定缓存；
- `thrid-party/XLeRobot` 上游克隆（可选，见下）；
- 根目录重复的 `modeling_smolvla.py` / `smolvlm_with_expert.py` 副本；
- 绝对路径符号链接 `src/lerobot/robots/xlerobot`（原链接指向本机 `thrid-party`，不可移植）。

## 应用

```bash
./apply.sh /path/to/lerobot
cd /path/to/lerobot
git status --short
```

脚本会克隆上游、检出基线提交并应用补丁。已有目录不会被覆盖。

手动方式：

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout 1033680a5
git apply /path/to/0001-xlerobot-jetson-integration.patch
```

验证补丁可干净应用（不落盘）：

```bash
./verify.sh
```

## 安装

1. 按 [已验证环境](../../docs/environment.md) 准备 JetPack、CUDA、TensorRT 与 Python。
2. 可选：使用 `locks/` 中的精确环境锁（linux-aarch64，随 JetPack 变化，不要跨平台复用）。
3. 准备与 JetPack 匹配的 PyTorch/TorchVision wheel，安装后执行 `pip install -e .`。
4. 复制环境模板并填写本机配置：

```bash
cp /path/to/xlerobot-repo/.env.example .env
# 编辑 .env：代理、VLM、相机 URL、串口
source envs.sh
```

`envs.sh` 会自动探测 Conda，优先激活 `lerobot_pzy`，并加载仓库根目录或上级目录的 `.env`。
串口权限建议使用 udev 规则，而不是 `sudo chmod 666`。

4. 可选：克隆 XLeRobot 以使用其仿真/硬件资产：

```bash
mkdir -p thrid-party
git clone https://github.com/Vector-Wangel/XLeRobot.git thrid-party/XLeRobot
```

补丁中的 `src/lerobot/model/SO101Robot.py` 是指向 `thrid-party/XLeRobot/.../SO101Robot.py`
的相对符号链接；使用 VR/Xbox 遥操作前必须完成上述克隆，否则该模块导入会失败。其他策略与
TensorRT 脚本不依赖它。

## 相机与串口

补丁中的示例脚本保留了实验室设备号（`/dev/video0/2/4`、`/dev/ttyACM0..3`）。复现前必须按
[硬件清单](../../docs/hardware_bom.md) 与 `robot-agent/docs/hardware-profile.md` 重新确认映射。

## 已知边界

- 权重与数据未包含；策略强过拟合，不能直接跨硬件使用，见 [模型权重范围](../../docs/model-weights.md)。
- TensorRT engine 与 JetPack/TensorRT 版本绑定，换版本需重新导出与编译。
- 补丁不保证与上游后续版本兼容；升级上游需要重新基线化。
- 这是集成补丁，不是经认证的安全系统；真实动作必须保留物理急停。

## 许可证

上游 LeRobot 与本补丁均按 Apache-2.0 分发；上游版权与 NOTICE 要求继续适用。XLeRobot 资产按其
Apache-2.0 许可证使用。
