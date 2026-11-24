# 评测协议 (Benchmarking)

目标：让“成功率”可复核。本文件定义评测协议与报告模板；本项目的实机任务结果见 [results.md](results.md)。

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

## 操作步骤

### 0. 冻结部署与指纹

先固定代码、权重、相机映射、桌面 profile 与标定，然后记录：

```bash
robotctl capabilities > capabilities.json
python - <<'PY'
import json
c = json.load(open("capabilities.json"))
for task_id, task in c["task_catalog"]["tasks"].items():
    print(task_id, task["fingerprint"], task["executor"])
print(json.dumps(c.get("policy_registry", {}), indent=2, ensure_ascii=False))
PY
```

把任务 fingerprint 与 policy fingerprint 写入 benchmark spec 的 `task_fingerprint` 与
`binding_equals.policy_fingerprint`。两者任一变化，旧 episode 不再计入新报告。

### 1. 配置切片

每个切片使用一个 episode context 文件，并重启 `robotd`（context 在启动时加载）：

```bash
export ROBOT_AGENT_EPISODE_DIR=/data/bench/fold_simple/episodes
export ROBOT_AGENT_EPISODE_CONTEXT=/data/bench/fold_simple/bright-layout-a.yaml
systemctl --user restart robotd
```

context 的 `tags` 必须覆盖 benchmark spec 的 `coverage` 键：

```yaml
version: 1
deployment_id: lab-kitchen-1
tags:
  lighting: bright
  layout: layout-a
  object_variant: cloth-a
```

每个切片至少 10 个 episode；推荐每任务 30–50 个，先跑 pilot 再扩量。

### 2. 采集

每个 episode 走完整观察 → 授权 → 执行流程：

```bash
robotctl observe "确认桌面目标、人员和障碍物状态"
robotctl arm fold_simple --confirm "CONFIRM fold_simple"
robotctl execute fold_simple --real
```

执行结束后确认 manifest 写入 `ROBOT_AGENT_EPISODE_DIR`，并检查：

- `success` 与 `failure_stage` 是否符合实际；
- `human_intervention` 是否在 `stop` / `pause` / `estop` 后为 true；
- `task_fingerprint` 与 `execution_binding` 是否与冻结值一致。

### 3. 绑定原始证据

为需要 benchmark 证据的任务建立 evidence spec（参考
`config/episode-evidence.example.yaml`），然后绑定：

```bash
robot-episode-evidence index --episode /data/bench/.../episode-xxx.json \
  --spec /data/bench/.../evidence.yaml
robot-episode-evidence check --episodes /data/bench/.../episodes \
  --require-kind rgb_video --require-kind robot_state
```

### 4. 评估

```bash
robot-benchmark-evaluate /data/bench/fold_simple/benchmark.yaml \
  /data/bench/fold_simple/episodes \
  --output /data/bench/fold_simple/report.json
```

报告会按 coverage 切片检查样本量、成功率、接管率与失败阶段比例。第一次运行建议使用
“报告基线”门禁（较低阈值），确认统计口径后再提高到发布阈值。报告 JSON 与原始 manifest
一起归档；不要只保留通过项。

### 5. 发布

在 `reports/`（或独立数据仓库）中提供：

- 报告 JSON 与 SHA-256；
- 每任务 n / 成功 / Wilson CI / 接管率 / 失败分类；
- 原始 episode 目录的哈希清单（原始数据可因隐私不公开，但哈希与统计必须可核对）；
- 失败样本的脱敏视频或截图；
- 明确的限制：硬件、标定、光照、物体集合、权重指纹。

## 推荐节奏

| 阶段 | 工作量 | 目标 |
|------|--------|------|
| Pilot | 每任务 5–10 episode | 验证 manifest、失败分类、证据链 |
| Baseline | 每任务 30–50 episode | 得到可报告的 CI 与失败分布 |
| Release gate | 覆盖全部切片 | 设定并冻结发布阈值 |
| Regression | 每次权重/相机/固件变更后重跑 | 防止性能回退 |

## 禁止事项

- 不得只报告成功片段或最好一次运行；
- 不得把不同硬件、不同相机安装、不同任务定义的结果合并；
- 不得在没有原始 episode 与失败分类的情况下引用成功率；
- 不得用仿真结果替代实机结果，除非明确标注。
