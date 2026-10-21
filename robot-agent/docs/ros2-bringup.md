# ROS 2 Bringup and HIL Procedure

本流程只适用于围挡测试区。首次 HIL 必须将底盘架空或断开驱动使能，确认轮子不能推动机器人后再开始。仓库中的示例 footprint、地图路径和标定状态不可直接用于真机。

## Build

在已安装 ROS 2、Nav2、`opennav_docking_msgs`、`xacro` 和 `colcon` 的目标机器上：

```bash
cd /opt/robot-agent
.venv/bin/pip install -e .
cd ros2_ws
colcon build --symlink-install --packages-select xlerobot_agent_bringup
source install/setup.bash
```

ROS 2 使用的 Python 环境必须能导入 `robot_agent`。不要让系统 ROS 和虚拟环境分别加载不兼容版本的 `rclpy`。

## Calibrate

1. 从 `config/bringup.example.yaml` 创建部署 manifest。
2. 提供正式 URDF/xacro、占据栅格地图、Nav2 参数和 base-bridge 参数文件。
3. 实测三轮半径、轮心到旋转中心距离、base footprint 和 collision-monitor stop polygon。
4. 验证 `base_link -> odom -> map`、IMU、里程计与 `/scan` 的时间戳和方向。
5. 由另一名操作者复核后填写 reviewer、date，并把 `approved` 改为 `true`。

运行硬门禁：

```bash
robot-bringup-check /etc/robot-agent/bringup.yaml
```

只有输出 `"valid": true` 才允许启动：

```bash
ros2 launch xlerobot_agent_bringup navigation.launch.py \
  manifest:=/etc/robot-agent/bringup.yaml use_sim_time:=false
```

launch 会再次执行同一预检，避免绕过 CLI 检查。Nav2 controller 必须发布到 `cmd_vel_nav`，collision monitor 是唯一允许发布最终 `/xlerobot/base/cmd_vel_safe` 的 Nav2 节点。

当前 ZMQ host 没有认证能力，因此 bridge 只允许 loopback 地址；host 与 ROS 2 必须运行在同一受控机器。协议和里程计限制见 [base-bridge.md](base-bridge.md)。

## RGB World-Model Bridge

RGB 分割结果写入世界模型的节点是独立、默认关闭的。它不在
`navigation.launch.py` 中自动启动，也不会驱动底盘或机械臂。只有显式提供绝对世界模型
路径时才会运行：

```bash
ros2 run xlerobot_agent_bringup rgb_world_model_bridge --ros-args \
  -p world_model_path:=/var/lib/robot-agent/world.yaml \
  -p segmentation_topic:=/xlerobot/segmentation
```

节点只接受新鲜的 `SegmentationFrame`，并通过带哈希的 observation-fusion 事件保存标签、
置信度、相机帧和 2D ROI。未来帧、过期帧、重复 observation ID 或非法 mask 会被拒绝；
空帧不会删除已有对象。它不产生深度、三维坐标、可达性或接触事实。

## Record HIL

每个故障场景单独启动被动记录器。记录器不会注入故障或发送运动命令：

```bash
robot-hil-record /var/lib/robot-agent/hil/events.jsonl \
  --scenario-id heartbeat-001 --scenario heartbeat_loss
```

导航开始并达到经批准的最低测试速度后，由操作者执行预先评审的故障步骤，再立即发布标记：

```bash
ros2 topic pub --once /robot_agent/hil/fault std_msgs/msg/String \
  "{data: heartbeat_loss}"
```

依次记录：

- `heartbeat_loss`：停止 gateway 心跳，保留底盘 host 与 recorder。
- `robotd_killed`：终止 `robotd`，验证 watchdog 独立停车。
- `nav2_killed`：终止 Nav2 controller/action 进程，验证低层停车。
- `sensor_loss`：在受控环境断开测试传感器，验证 action abort 与停车。

碰撞检测链路应向 `/robot_agent/hil/collision` 发布 `std_msgs/Bool`。记录器从 `/odom` 计算平移和角速度，并监听 `navigate_to_pose/_action/status`。

## Evaluate

```bash
robot-hil-evaluate /var/lib/robot-agent/hil/events.jsonl \
  --policy /etc/robot-agent/hil-policy.yaml \
  --output /var/lib/robot-agent/hil/latest-report.json
```

报告必须覆盖四类场景、碰撞为零、在 500ms 内首次归零、持续至少 200ms，并在 2 秒内得到 canceled、aborted 或 rejected action 终态。稀疏速度采样不能证明稳定停车。

通过人工复核后才可配置：

```bash
ROBOT_AGENT_MOTION_BACKEND=ros2
ROBOT_AGENT_BRINGUP_MANIFEST=/etc/robot-agent/bringup.yaml
ROBOT_AGENT_HIL_REPORT=/var/lib/robot-agent/hil/latest-report.json
```

`robotd` 每次启动都会重新检查 manifest 和报告。如果任一场景缺失或参数超过 commissioning 上限，motion backend 保持 unavailable。
