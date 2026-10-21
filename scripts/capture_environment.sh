#!/usr/bin/env bash
# Print a reproducible environment snapshot. Never prints tokens or proxy values.
set -u

echo "## System"
uname -srm
[ -r /proc/device-tree/model ] && echo "device: $(tr -d '\0' < /proc/device-tree/model)"
[ -r /etc/os-release ] && grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release
[ -r /etc/nv_tegra_release ] && echo "l4t: $(head -n 1 /etc/nv_tegra_release)"

echo
echo "## CUDA / TensorRT"
nvcc --version 2>/dev/null | tail -n 2 || echo "nvcc: not found"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
if [ -x "$TRTEXEC" ]; then
  "$TRTEXEC" --version 2>/dev/null | grep -m 1 'TensorRT v' || true
else
  echo "trtexec: not found"
fi

echo
echo "## Python"
PYTHON="${PYTHON:-python3}"
"$PYTHON" --version
"$PYTHON" - <<'PY' 2>/dev/null || true
import importlib.metadata as m
for name in (
    "torch", "torchvision", "lerobot", "transformers", "onnx", "onnxruntime",
    "numpy", "safetensors", "accelerate", "huggingface-hub", "PyYAML", "pyzmq",
):
    try:
        print(f"{name}: {m.version(name)}")
    except Exception:
        print(f"{name}: missing")
PY

echo
echo "## Conda environments"
CONDA_BIN="${CONDA_BIN:-$(command -v conda 2>/dev/null || true)}"
if [ -z "$CONDA_BIN" ] && [ -x "$HOME/miniconda3/bin/conda" ]; then
  CONDA_BIN="$HOME/miniconda3/bin/conda"
fi
if [ -n "$CONDA_BIN" ]; then
  "$CONDA_BIN" env list 2>/dev/null | sed '/^#/d'
else
  echo "conda: not found"
fi

echo
echo "## Git"
git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo "not a git worktree"
