# 复现指南 (Reproduction Guide)

目标：让外部读者能在**没有本实验室硬件**的情况下复现框架行为；在有同构硬件时按标定流程复现任务。
模型权重复用边界见 [model-weights.md](model-weights.md)，版本矩阵见 [environment.md](environment.md)。

## 0. 复现范围

| 可复现 | 需要额外资产 |
|--------|--------------|
| `robot-agent` 安装、测试、dry-run、mission 编排 | 无 |
| Web 控制台与 CLI | 三路相机快照 URL（可用假 URL 验证接口） |
| 语义桌沿对接逻辑 | 真实底盘、相机、VLM 端点与标定 |
| 4 项家务技能 | 对应权重、数据集、机械臂标定、相机安装 |
| TensorRT 加速 | 匹配 JetPack/TensorRT 的 Jetson、ONNX、engine 构建环境 |
| 论文/视频中的成功率 | 原始 episode、benchmark 报告与失败样本 |

## 1. 框架安装与验证

```bash
git clone <repository-url>
cd <repository>/robot-agent
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

预期：`362 passed`（版本更新后以实际输出为准）。再在仓库根目录检查文档链接：

```bash
python scripts/check_markdown_links.py
```

## 2. 配置

```bash
cp .env.example .env
# 编辑 .env：VLM_API_URL / VLM_MODEL / VLM_API_TOKEN
# 三路相机快照：ROBOT_AGENT_{HEAD,LEFT,RIGHT}_CAMERA_SNAPSHOT_URL
```

`VLM_API_URL` 与 `VLM_MODEL` 没有默认值；缺省时观察直接失败关闭，不会向未知端点发送图像。

## 3. 无硬件 dry-run

```bash
python -m robot_agent.service          # 终端 A
python -m robot_agent.cli status       # 终端 B
python -m robot_agent.cli dry-run "把瓶子放进筐里"
python -m robot_agent.cli mission-dry-run <mission_id>
```

dry-run 不发送硬件命令；用于验证任务匹配、catalog、mission 图与 Token 流程。

## 4. 有硬件的最小闭环

1. 按 [hardware_bom.md](hardware_bom.md) 完成供电、共地、急停与串口绑定。
2. 按 [hardware-profile.md](../robot-agent/docs/hardware-profile.md) 确认三路相机映射。
3. 标定双臂零位与夹爪行程，写入 LeRobot 标定目录。
4. 标定桌沿 profile（`config/tables.example.yaml` 的 standoff 与容差），安装为
   `ROBOT_AGENT_TABLE_CATALOG`。
5. 先执行观察，再 arm，再执行：

```bash
robotctl observe "确认桌面目标、人员和障碍物状态"
robotctl arm dock_table --confirm "CONFIRM dock_table"
robotctl execute dock_table --real
```

6. 验证 `stop` / `estop`、进程组取消、看门狗归零与物理急停。

## 5. 训练与 TensorRT

先重建定制 LeRobot 工作区：

```bash
./integrations/lerobot_jetson/apply.sh /path/to/lerobot
./integrations/lerobot_jetson/verify.sh
```

训练工作区、数据集与权重不在本仓库分发。当前流程：

1. 遥操作采集 episode，检查相机映射与动作维度；
2. 使用 `smolvla` / `pi05` 配置训练，记录 seed、数据版本与归一化统计；
3. 导出 ONNX，固定 shape profile 与 opset；
4. 用 `trtexec` 构建 BF16/INT8 engine，记录 TensorRT 版本；
5. 对比 PyTorch 与 TensorRT 输出误差，跑 benchmark 门禁；
6. 通过 policy registry 绑定权重 SHA-256 与部署指纹。

## 6. 当前复现缺口

- `lerobot_0.3.3_pzy` 已整理为 `integrations/lerobot_jetson/` 补丁（基线 `1033680a5`）；
  π0.5 已整理为 `integrations/lerobot_pi05/` 补丁（基线 `473f1bd0e`）。两者需应用到独立 checkout。
- `XLeRobot` 与训练数据、checkpoint、engine、标定缓存仍需单独获取。
- 没有真实硬件的自动化 HIL；CI 只覆盖主机侧逻辑。
- 尚未发布按任务统计的成功率与置信区间；在补齐前，不应引用具体性能数字。

## 7. 复现记录模板

提交 issue 或论文附录时附上：

```text
commit:
environment snapshot: bash scripts/capture_environment.sh
camera mapping:
arm calibration hash:
table profile:
policy fingerprint:
benchmark report:
failure episodes:
```
