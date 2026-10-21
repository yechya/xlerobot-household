# 本地策略配置示例

这份记录对应当前开发机的策略资产登记，不代表已经允许机械臂执行。路径中的 `$HOME` 需替换为
实际用户目录；这些本地文件不由本仓库分发。

## 已登记资产

- 运行环境：`$HOME/miniconda3/envs/lerobot_pzy`
- 直接入口：`$HOME/miniconda3/envs/lerobot_pzy/bin/lerobot-record`
- 策略工作区：`$HOME/project/lerobot_0.3.3_pzy`
- 精确 Conda 清单：`$HOME/.config/robot-agent/lerobot_pzy.conda.lock`
- 注册表：`$HOME/.config/robot-agent/policies.yaml`

已登记四个本地 SmolVLA checkpoint：`fold_simple`、`stack_contrast`、
`pick_up_bottle_towel` 和 `pick_up_bottle`。模型摘要、可执行文件摘要、运行时清单摘要和三路
RGB 配置都由 `robot-policy-materialize` 计算，不能手工填写。

当前注册表中的四个 `enabled` 都是 `false`。完整资产校验在临时全启用副本上通过，但这只证明
文件完整性，不证明任务质量、碰撞安全或机械臂可执行性。

## 当前运行时

`$HOME/.config/robot-agent/robot-agent.env` 已设置：

```bash
ROBOT_AGENT_POLICY_REGISTRY=$HOME/.config/robot-agent/policies.yaml
```

重启 `robotd` 后，`robotctl capabilities` 会展示四个策略 fingerprint；授权仍会因为策略禁用而
拒绝，Dry-run 不受影响：

```bash
systemctl --user restart robotd-software.service
robotctl capabilities
robotctl dry-run fold_simple
```

## 启用前置条件

当前已发现 `/dev/ttyACM0` 和 `/dev/ttyACM1`，但当前用户尚未加入 `dialout` 组；同时还没有完成
串口身份、舵机标定、急停/停止行为、策略前置/后置条件和 benchmark 验收。因此不能把本地副本
直接改成 `enabled: true`，也不能用占位策略或伪造报告绕过门禁。

完成硬件和策略验收后，再针对单个任务生成新的部署副本：

```bash
robot-policy-materialize policies.template.yaml /etc/robot-agent/policies.yaml \
  --enable fold_simple
robot-policy-check /etc/robot-agent/policies.yaml
```

只把通过人工复核、RGB 观察、停止/急停和 benchmark 的任务加入 `--enable`，并在变更后重启
`robotd`。任何模型、运行时、串口或相机映射变化都必须重新 materialize 和校验。
