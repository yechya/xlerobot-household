# 评测协议 (Benchmarking)

目标：让“成功率”可复核，而不是引用精剪视频。仓库当前**未发布**正式结果；本文件定义协议与报告
模板，避免把 demo 当作性能证明。

## 任务级指标

| 指标 | 定义 |
|------|------|
| 成功率 | 后置条件证据通过且无需人工接管的 episode 比例 |
| 接管率 | 操作者介入（遥操作、复位、急停后重启）的 episode 比例 |
| 前置失败率 | 目标/工作区/人员条件未通过，未进入策略执行 |
| 策略失败率 | 策略已执行但未满足后置条件 |
| 后置失败率 | 策略返回成功但证据未通过 |
| 时延 | 端到端动作时延与推理时延的 p50 / p95，注明后端与设备 |

每个任务至少报告：episode 数、成功数、成功率、95% 置信区间、接管率、失败阶段分布。

## 样本量与置信区间

小样本不能用裸成功率下结论。使用 Wilson 区间：

$$
\hat{p}=\frac{k}{n},\qquad
\frac{\hat{p}+\frac{z^2}{2n}\pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n}+\frac{z^2}{4n^2}}}{1+\frac{z^2}{n}}
$$

其中 $k$ 为成功数，$n$ 为有效 episode 数，$z=1.96$（95%）。报告示例：

| 任务 | n | 成功 | 成功率 | 95% CI | 接管率 |
|------|---:|---:|---:|---|---:|
| `fold_simple` | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| `stack_contrast` | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| `pick_up_bottle_towel` | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| `pick_up_bottle` | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| `dock_table` | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

## 环境切片

至少在以下维度分层，不能只报总体：

- 光照：明亮 / 昏暗 / 混合；
- 物体位置：左 / 中 / 右，近 / 远；
- 物体外观：至少 3 个变体；
- 人员/障碍：无人、旁观、轻微遮挡；
- 相机与桌面高度：标称、允许偏差范围。

## 失败分类

使用 episode 的 `failure_stage`，不要把所有失败归为“模型不行”：

1. `precondition_evidence`：目标不可见、工作区不净；
2. `scene_resolve`：坐标/语义解析失败；
3. `task_backend_execute`：策略执行失败；
4. `postcondition_evidence`：策略返回但证据不通过；
5. `navigate` / `dock` / `dock_table`：移动或对接失败；
6. `recovery`：恢复动作失败；
7. `operator_stop` / `estop`：人工中止，单独统计，不混入模型失败。

## 原始证据

每个 episode 至少保留：

- episode manifest 与 execution binding；
- 三路相机原始帧或视频（按隐私要求脱敏）；
- 策略 fingerprint、benchmark report、失败阶段；
- 操作者接管记录与时间戳。

## 报告模板

```text
commit:
task:
policy fingerprint:
dataset version / episode range:
camera mapping + calibration hash:
table profile:
backend (PyTorch / TensorRT BF16 / INT8):
device + JetPack + TensorRT:
n / successes / Wilson 95% CI:
intervention rate:
failure stage counts:
latency p50 / p95:
raw episodes path + sha256:
```

## 禁止事项

- 不得只报告成功片段或最好一次运行；
- 不得把不同硬件、不同相机安装、不同任务定义的结果合并；
- 不得在没有原始 episode 与失败分类的情况下引用成功率；
- 不得用仿真结果替代实机结果，除非明确标注。
