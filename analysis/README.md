# 实验分析脚本

本目录把 [docs/results.md](../docs/results.md) 中可由录制数据复算的数字对应到脚本和输出文件。所有脚本都已在
Jetson AGX Orin 上对原始数据重新运行，本目录中的输出文件就是这次运行的结果。

关节数值为 LeRobot 归一化舵机位置：关节 −100～100，夹爪 0～100。

## 结果与脚本对照

| 结果 | 脚本 | 输出 | 复算值 |
| --- | --- | --- | --- |
| SmolVLA TensorRT 混合推理（20 帧实机观测） | [`trt/summarize_smolvla_trt.py`](trt/summarize_smolvla_trt.py) | [`trt/smolvla_pick_up_bottle_fp16_summary.json`](trt/smolvla_pick_up_bottle_fp16_summary.json)、`_per_frame.csv`、`_latency.png` | 836.47 → 333.92 ms（2.50×）；余弦相似度均值 0.99999989、最差 0.99999880；相对 L2 均值 4.41×10⁻⁴、最差 1.53×10⁻³；平均绝对误差 0.0169；第 0 帧 453.2 ms（预热） |
| 实机有效控制频率（瓶子入筐） | [`control_frequency/control_frequency.py`](control_frequency/control_frequency.py) | [`control_frequency/control_frequency.csv`](control_frequency/control_frequency.csv) | SmolVLA 227 / 20 s = 11.35 Hz → 247 / 20 s = 12.35 Hz（+8.8%）；π0.5 614 / 60 s = 10.23 Hz → 701 / 60 s = 11.68 Hz（+14.2%） |
| 平滑窗口 1 / 3 / 5 | [`smoothing/smoothing_metrics.py`](smoothing/smoothing_metrics.py) | [`smoothing/smoothing_metrics.csv`](smoothing/smoothing_metrics.csv) | 控制帧 616 / 553 / 657；指令变化率 0.386 / 0.265 / 0.416；关节加速度 0.172 / 0.132 / 0.210；跟踪误差 1.681 / 1.365 / 1.726 |
| 三层误差模型 | [`jitter/jitter_analysis.py`](jitter/jitter_analysis.py) | [`jitter/jitter_summary.json`](jitter/jitter_summary.json)、[`chunk_edges.csv`](jitter/chunk_edges.csv)、[`phases.csv`](jitter/phases.csv)、`chunk_jitter.png` | 3 个回合共 1826 帧；跟踪误差与指令变化率 r = 0.9418，与动作幅度 r = 0.0087；包含块切换的末尾 10 帧比开头 10 帧高 +64% / +115% / +173%；过渡 / 精调阶段误差 3.605 / 1.244；右夹爪开度小于 20 的帧占 82.0%（1498 / 1826） |

[docs/results.md](../docs/results.md) 中末尾抖动的 +172% 是按两位小数的均值计算的，不取整计算为 +173%。

## 各量定义

- **TensorRT 验证**：PyTorch 与 TensorRT 后端对同一组 20 帧实机观测推理，比较最终下发给舵机的 12 维动作。
  原始逐帧记录在 [`trt/raw/`](trt/raw/)，由补丁中的 `scripts/verify_trt_with_real_data.py` 生成
  （见 [integrations/lerobot_jetson](../integrations/lerobot_jetson/README.md)）。
- **有效控制频率**：录制帧数 ÷ 录制时长。LeRobot 录制循环每个控制步写一帧，推理越慢，固定时长内的帧越少。
- **平滑指标**：指令变化率 = 相邻帧指令差的绝对值均值；关节加速度 = 实际关节位置二阶差分的绝对值均值；
  跟踪误差 = 指令与实际位置之差的绝对值均值；均按 12 个关节平均。
- **抖动分析**：相关系数、阶段划分和块首尾比较的公式写在 `jitter_analysis.py` 文件头。块首尾比较按 50 帧一个动作块，
  比较块内右臂指令二阶差分的前 10 个值和后 10 个值。

## 运行方法

依赖：Python 3.10+，`numpy pandas pyarrow matplotlib`；从 rerun 录制文件读取时另需 `rerun-sdk==0.22.*`。

```bash
cd analysis/trt          # 只依赖 raw/ 中的结果文件，可直接运行
python summarize_smolvla_trt.py raw/smolvla_pick_up_bottle_verify_real_data.json \
    --name smolvla_pick_up_bottle_fp16 --out-dir . --plot --trt-label "TensorRT FP16 (hybrid)"

cd ../control_frequency  # $DATA 为 LeRobot 数据集所在目录
python control_frequency.py --out-dir . \
    --run SmolVLA PyTorch         $DATA/eval_smolvla_pick_up_bottle_three_py_20s  20 \
    --run SmolVLA "TensorRT FP16" $DATA/eval_smolvla_pick_up_bottle_three_trt_20s 20 \
    --run pi0.5   PyTorch         $DATA/eval_pi05_pytorch_60s 60 \
    --run pi0.5   "TensorRT BF16" $DATA/eval_pi05_trt_60s     60

cd ../smoothing
python smoothing_metrics.py --out-dir . \
    --episode 1 $DATA/eval_smolvla_pick_up_bottle 0 \
    --episode 3 $DATA/eval_smolvla_pick_up_bottle 1 \
    --episode 5 $DATA/eval_smolvla_pick_up_bottle 2

cd ../jitter
python jitter_analysis.py --dataset $DATA/eval_smolvla_pick_up_bottle --out-dir . --plot
```

## 数据

公开：本目录中的脚本、`trt/raw/` 中的逐帧验证记录（绝对路径字段已缩为数据集名）以及全部派生 JSON / CSV / 图。
未公开：LeRobot 数据集（含三路相机视频）、rerun 录制文件、模型权重与 TensorRT 引擎。脚本的输入路径都是命令行参数，
拿到同格式的数据即可运行。
