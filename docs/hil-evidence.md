# 硬件在环与急停证据 (HIL Evidence)

本文件定义真实硬件安全行为的验收记录。**当前仓库没有公开的 HIL 报告**；在补齐以下证据前，
不得宣称“已通过硬件在环验证”。

## 必测场景

| 场景 | 操作 | 通过标准 |
|------|------|----------|
| 软件急停 | 执行中调用 `robotctl estop` | 进程组终止、底盘零速、机械臂停止、审计记录 |
| 操作者停止 | 执行中调用 `robotctl stop` | 当前动作取消，后续 mission 节点不再执行 |
| 看门狗断连 | 断开控制心跳 / 杀死 client | 250ms 内速度归零，500ms 内硬件停车 |
| 电源/通信丢失 | 断开机械臂或底盘串口 | 适配器报错，任务失败关闭，不继续发送动作 |
| 相机丢失 | 拔掉一路相机 | 观察失败，需观察的任务拒绝执行 |
| 人员进入 | 安全策略要求的最小距离内出现人员 | motion/manipulation 被拒绝或停止 |
| 物理急停 | 按下硬件急停 | 执行机构动力切断，软件状态进入 `ESTOP` |
| 复位 | 物理急停释放后 `robotctl reset` | 仅在确认字符串匹配后恢复，旧 token 失效 |

## 执行步骤

### ROS 2 场景

```bash
robot-hil-record /data/hil/heartbeat_loss.jsonl \
  --scenario-id heartbeat_loss --scenario heartbeat_loss
# 注入对应故障（停止 heartbeat / kill 进程 / 断相机）
robot-hil-evaluate /data/hil/heartbeat_loss.jsonl \
  --policy config/hil-policy.example.yaml \
  --output /data/hil/heartbeat_loss.report.json
```

对 `config/hil-policy.example.yaml` 的 `required_scenarios` 逐个执行：
`heartbeat_loss`、`robotd_killed`、`nav2_killed`、`sensor_loss`。

### 物理急停与断电

这些场景不由 `robot-hil-record` 自动覆盖，需要人工记录：

1. 机器人低速执行 `dock_table` 或一个已授权任务；
2. 按下物理急停；
3. 确认执行机构断电、软件进入 `ESTOP`、审计日志写入；
4. 释放急停并执行 `robotctl reset --confirm "RESET ESTOP"`；
5. 验证旧 token 已失效、需要重新观察和 arm；
6. 保存视频、审计日志与 `episode` 清单，填写下方记录模板。

### 记录归档

把每个场景的 JSONL、报告、日志与视频哈希写入 `docs/hil-evidence.md` 的副本或独立
`reports/hil-<date>.md`，并在 release 说明中链接。未执行的场景标记 `BLOCKED`，不要留空。

## 记录格式

```text
测试日期:
commit:
硬件序列号 / 标定哈希:
固件与 JetPack 版本:
场景:
操作步骤:
预期行为:
实际行为:
恢复步骤:
日志路径 + sha256:
视频/照片路径 + sha256:
结论: PASS / FAIL / BLOCKED
```

## 证据保存

- 审计日志：`$ROBOT_AGENT_STATE_DIR/audit.jsonl`；
- episode manifest 与 benchmark 报告；
- 原始视频与串口日志（脱敏后）；
- 失败场景必须保留，不得只保留通过项。

## 自动化边界

`robot-hil-evaluate` 与 `robot-agent/docs/m1-nav2-hil.md` 覆盖部分 Nav2 验收检查；它们不能替代
物理急停、断电与人员安全测试。CI 只在主机侧运行，不连接真实硬件或 Jetson。
