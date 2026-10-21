# Dependency Locks

这些锁来自已验证的 Jetson AGX Orin 环境（L4T R36.4.7 / CUDA 12.6 / TensorRT 10.3.0），用于
复现与故障排查。它们**不是跨平台通用锁**：换 JetPack、GPU 架构或 CUDA 版本必须重新生成。

## 文件

| 文件 | 内容 |
|------|------|
| `conda-explicit.lock` | conda 精确 URL 锁（linux-aarch64，177 行） |
| `pip-freeze.lock` | pip 包名==版本（排除 editable 的 `lerobot`） |

## 使用

```bash
# 1. 新建 conda 环境（可能需要把清华镜像替换为官方 channel）
conda create --name lerobot_pzy --file locks/conda-explicit.lock

# 2. 安装与 JetPack 匹配的 PyTorch/TorchVision wheel
#    参考 integrations/lerobot_jetson/README.md 与 install_jetson.sh

# 3. 安装剩余 pip 包
pip install -r locks/pip-freeze.lock

# 4. 安装打过补丁的 LeRobot（editable）
pip install -e . --no-deps
```

`pip-freeze.lock` 中的 `torch`/`torchvision` 版本必须与已安装的 Jetson wheel 一致；如果 pip
尝试从 PyPI 重新下载，先安装 wheel 再执行第 3 步，或使用 `--no-deps` 逐项安装。

## 重新生成

```bash
conda list --explicit -p "$CONDA_PREFIX" > locks/conda-explicit.lock
python - <<'PY'
import json, subprocess, sys
packages = json.loads(subprocess.check_output(
    [sys.executable, "-m", "pip", "list", "--format=json"], text=True))
for item in sorted(packages, key=lambda x: x["name"].lower()):
    if item["name"].lower() in {"lerobot", "xlerobot-agent"}:
        continue
    print(f'{item["name"]}=={item["version"]}')
PY
```

生成后检查是否包含 `file://`、`/home/<user>`、token 或私有代理；这些不能进入公开锁文件。
