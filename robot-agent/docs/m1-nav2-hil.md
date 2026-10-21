# M1 Nav2 and Hardware-in-Loop Gate

M1 的目标不是让机器人尽快在家里自由移动，而是先证明底盘在可重复、可取消、可审计的条件下可靠移动。当前机器未安装 ROS 2/Nav2，本阶段实现只完成控制边界和离线测试，不允许据此启用真机运动。

## Why This Order

1. **先固定命名目标，再开放任务规划。** `robotctl` 只接受版本化目录中的位置和 dock，不接受任意坐标。这样地图变更可审查，令牌能绑定明确目标，也能防止模型输出未经验证的位姿。
2. **先建立停止链路，再测试成功路径。** Nav2 action cancellation、零速度发布、XLeRobot host 的 500ms watchdog、MCU watchdog 和物理急停必须层层独立；否则上层进程崩溃会留下持续运动风险。
3. **先仿真和 HIL，再进入居家实测。** 导航成功率不是唯一门槛，传感器丢失、TF 错误、action 超时、网络阻塞和进程退出都必须验证能停车。
4. **先可靠到达桌边，再做移动操作。** 抓取策略依赖可重复的相机视角和机械臂工作空间。桌边对接误差未收敛时，增加 VLA 或 MoveIt 只会把导航误差传递给操作层。

## Runtime Contract

```bash
robotctl motion-catalog
robotctl navigate kitchen_table_staging
robotctl arm-motion navigate kitchen_table_staging \
  --confirm "CONFIRM motion:navigate:kitchen_table_staging"
robotctl navigate kitchen_table_staging --real
robotctl stop
```

dry-run 只验证目标存在。真实命令需要近期观察、完全匹配的人工确认和一次性 token。`stop`、`estop` 与 daemon 退出都会取消 active goal 并发布零速度。

## Bring-Up Sequence

1. 安装与底盘驱动匹配的 ROS 2 发行版、Nav2、`opennav_docking_msgs` 和传感器驱动。
2. 建立 URDF、TF、轮速里程计、IMU 和激光或深度障碍物层，先验证时间戳与坐标树。
3. 在 Gazebo 或 Isaac Sim 中验证命名导航、取消、超时、恢复行为和 watchdog。
4. 将机器人架空或断开驱动使能做 HIL，验证命令输出和失联归零，不允许轮子接触地面。
5. 在围挡测试区以最低速度执行 100 次故障注入，包括杀死 `robotd`、杀死 Nav2、拔掉传感器和阻塞 action。
6. 通过后再标定少量家庭命名位置和充电 dock，逐点放行，不开放任意坐标接口。

仓库已提供参数化 ROS 2 launch、标定预检、被动 HIL 记录器与报告门禁，具体命令见 [ros2-bringup.md](ros2-bringup.md)。当前示例保持未审批状态，不会通过真机启动预检。

`robot-hil-evaluate` 先生成原始评测结果。通过人工复核后，使用 `robotctl
capabilities` 返回的 `deployment.motion_fingerprint` 创建有时效的部署验收：

```bash
robot-acceptance-bind nav2_hil <motion-fingerprint> \
  /var/lib/robot-agent/hil/nav2-result.json \
  /var/lib/robot-agent/hil/nav2-acceptance.json \
  --report-id nav2-home-v1 --valid-seconds 2592000
```

`ROBOT_AGENT_HIL_REPORT` 必须指向 acceptance 文件，而不是裸评测 JSON。文件绑定完整
payload、motion fingerprint、生成时间、有效期和自身 SHA-256。地图或导航标定改变、
文件被编辑或验收过期时，ROS 2 motion backend 保持不可用。

## Acceptance Gate

- 100 次固定路线导航成功率至少 95%，碰撞为 0。
- 任一上层心跳丢失后，底盘在 500ms 安全预算内停止；最终门限以实测制动距离为准。
- `stop` 和 `estop` 在导航、恢复行为及 docking 各阶段均有效。
- TF、定位、传感器或 action 故障均返回可审计原因，不静默重试无限次。
- 桌边 staging pose 的位置与航向误差满足后续手眼标定和机械臂工作空间要求。

ROS heartbeat watchdog 是用户态补充层，不是主安全控制器。现有 XLeRobot host watchdog、MCU watchdog、驱动器使能和物理急停必须在 `robotd` 与 ROS 2 全部失效时仍能独立停车。
