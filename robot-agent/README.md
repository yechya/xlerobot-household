# XLeRobot Agent Runtime

面向居家服务机器人的独立任务运行时。它不依赖任何对话式编程工具，也不要求 MCP 或网络控制面。

核心边界：

```text
operator / application
        |
     robotctl
        |
  Unix socket (0600)
        |
      robotd  -------- append-only audit
        |
 cancellable task process
        |
 hardware adapters / ROS 2 bridge
        |
 MCU watchdog + physical emergency stop
```

`robotd` 是任务授权和进程监管边界，不是硬实时电机控制器。限位、碰撞检测、失联停车和物理急停必须在硬件控制器或 ROS 2 控制层独立成立。

## 当前能力

- 默认 dry-run，未知任务默认拒绝。
- 单实例 Unix Socket 服务，CLI 只作为客户端。
- 白名单任务、一次性 token、近期观察要求和严格状态机。
- 真实任务运行在独立进程组，可由另一个连接执行 `stop` 或 `estop`。
- token 文件、socket、状态目录和审计日志均限制为当前用户访问。
- VLM 通过 OpenAI-compatible HTTP API 直接访问摄像头快照。
- 网页可监管 LeRobot 双臂遥操作数据集录制，支持新建/续录、回合进度、日志和相机所有权恢复。
- 底盘、全景云台和策略执行器通过显式适配器命令接入，未配置时拒绝真实动作。
- 三全向底盘 bridge 完成 ROS rad/s 与 host deg/s 转换、250ms 归零和轮速里程计发布。
- 版本化 policy registry 将任务与 checkpoint 解耦，并把模型 SHA-256/fingerprint 绑定到一次性授权。
- task catalog 保留可扩展的任务、安全和视觉证据契约；当前接受 `learned_policy` 与确定性 `table_docking` executor。
- 默认目录保留原项目四个任务：`fold_simple`、`stack_contrast`、`pick_up_bottle_towel` 和 `pick_up_bottle`，并新增 `dock_table` 语义桌沿对接。
- 语义桌沿对接由闭环 RGB 观察驱动，低置信度不动作，最大步长/步数/时限/停滞保护全部有界；校准参数来自版本化 table catalog 并进入部署指纹。
- mission `retry` 可声明 recovery 动作：任务重试前必须先完成一次新的 VLM 观察或已注册恢复任务，否则停止整条任务链。
- 硬件基线是三路普通 RGB 摄像头；策略授权会绑定三路相机配置，不接受深度、点云或力传感器字段。
- RGB segmentation publisher 可输出 source-tagged 2D mask，但不会伪造深度、3D 位姿或接触力。
- 分割模型通过独立 Python entry-point 包接入；distribution/version、权重摘要、查询映射、topic、frame、实例上限和推理 deadline 均由 manifest 固定，任务新增 label 不新增核心代码分支。
- 独立 loopback HTTP client 与模型无关服务端通过 provider entry point 连接 RGB 分割模型，不进入控制核心依赖。
- 真实任务写入不可变 episode；版本化 benchmark 按环境切片、成功率、接管率和失败阶段门禁 learned policy 晋级。
- 经审核的媒体/状态 evidence bundle 将原始文件、时间范围和 SHA 绑定到 episode；V2 benchmark 和生产 learned-policy 门禁要求匹配的原始证据。
- 可扩展 replay plan 先筛选 episode，再按具名 route 分流失败，并验证 episode/evidence/source-set/report 哈希和安全相对路径，供仿真、评测和数据导出器离线消费。
- 独立 replay runtime 支持证据导出和 RGB 视频片段处理，不再暴露 RGB-D/rosbag 几何管线。
- 结构化 safety interlock 在 arm/execute/mission 节点边界检查人员、宠物、隐私、传感器健康和物理急停，缺证据默认拒绝。
- RGB-D、MoveIt/MTC、Servo、wrench 接触、几何叠碗、擦拭和拉抽屉实现已移除；重新引入前必须先确认真实硬件与数据接口。

## 安装

```bash
cd robot-agent
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

开发模式可直接使用当前 Python 环境：

```bash
python -m robot_agent.service
python -m robot_agent.cli status
python -m robot_agent.cli dry-run "把瓶子放进筐里"
```

软件部署（不连接真机硬件）可使用用户级安装脚本：

```bash
./deploy/install-software.sh --enable
```

该 profile 固定 dry-run 和 disabled motion，真实策略不会被授权。完整说明见
[docs/user-guide.md](docs/user-guide.md) 的“软件部署”章节。

安装后对应命令为：

```bash
robotd
robotctl status
robot-web --port 8765
robotctl capabilities
robotctl plan "把瓶子放进筐里"
robotctl dry-run "把瓶子放进筐里"
robot-policy-materialize policies.template.yaml /etc/robot-agent/policies.yaml
robotctl world-state
robotctl world-events --since 0 --limit 100
robot-episode-evidence --help
robot-episode-replay-plan --help
robot-episode-replay --help
robot-data-triage --help
robot-acceptance-bind --help
```

## 真实任务流程

真实操作默认禁用。必须先配置硬件适配器，并依次完成观察、授权和执行：

```bash
robotctl observe "确认桌面目标、人员和障碍物状态"
robotctl arm pick_up_bottle --confirm "CONFIRM pick_up_bottle"
robotctl execute pick_up_bottle --real
robotctl status
```

`arm` 得到的一次性 token 会写入权限为 `0600` 的本地缓存，`execute --real` 使用后立即删除。`stop` 与 `estop` 不需要任务 token。

Nav2 真实导航采用相同授权流程，并且 token 与操作和命名目标精确绑定：

```bash
robotctl arm-motion navigate kitchen_table_staging \
  --confirm "CONFIRM motion:navigate:kitchen_table_staging"
robotctl navigate kitchen_table_staging --real
```

默认 `ROBOT_AGENT_MOTION_BACKEND=disabled`。完成仿真与 HIL 门禁前不得改为 `ros2`，示例坐标也不得直接用于真机。

启用 ROS 2 backend 还必须配置通过预检的 `ROBOT_AGENT_BRINGUP_MANIFEST` 和覆盖全部故障场景的 `ROBOT_AGENT_HIL_REPORT`。完整流程见 [docs/ros2-bringup.md](docs/ros2-bringup.md)。

## 配置

参考 [config/robot-agent.env.example](config/robot-agent.env.example)。重要变量：

| 变量 | 作用 |
|------|------|
| `ROBOT_AGENT_STATE_DIR` | token、审计和世界模型目录 |
| `ROBOT_AGENT_RUNTIME_DIR` | Unix Socket 目录 |
| `ROBOT_AGENT_CHASSIS_COMMAND` | 底盘适配器命令 |
| `ROBOT_AGENT_PANORAMA_COMMAND` | 云台全景适配器命令 |
| `ROBOT_AGENT_EMERGENCY_STOP_COMMAND` | 独立急停命令 |
| `ROBOT_AGENT_MOTION_BACKEND` | `disabled` 或可选的 `ros2` Nav2 gateway |
| `ROBOT_AGENT_NAVIGATION_CATALOG` | 版本化命名位置和 dock 目录 |
| `ROBOT_AGENT_WATCHDOG_TIMEOUT` | ROS heartbeat 零速度保护超时 |
| `ROBOT_AGENT_BRINGUP_MANIFEST` | 已复核的 ROS 2/Nav2 标定 manifest |
| `ROBOT_AGENT_HIL_REPORT` | 绑定当前 motion domain 指纹且未过期的 Nav2 HIL acceptance |
| `ROBOT_AGENT_POLICY_REGISTRY` | 已审核的 learned-policy 部署注册表 |
| `ROBOT_AGENT_DATASET_RECORDING_CONFIG` | LeRobot 遥操作录制运行时、主臂端口、限制和相机服务切换配置 |
| `ROBOT_AGENT_TASK_CATALOG` | 可扩展任务、安全和证据契约目录 |
| `ROBOT_AGENT_SEMANTIC_MAP_CONFIG` | 启动前审核并原子同步的房间、台面、容器和区域地图资产 |
| `ROBOT_AGENT_MISSION_CATALOG` | 跨房间命名 motion + task 的版本化任务图 |
| `ROBOT_AGENT_TABLE_CATALOG` | 语义桌沿对接的版本化标定 profile（standoff、容差、步长、时限） |
| `ROBOT_AGENT_EPISODE_DIR` / `ROBOT_AGENT_EPISODE_CONTEXT` | 不可变执行清单目录与当前环境标签 |
| `ROBOT_AGENT_BENCHMARK_REPORT` | learned policy 晋级报告文件或按任务组织的目录；生产默认必需 |
| `ROBOT_AGENT_REQUIRE_POLICY_EVIDENCE` | 要求 learned policy 晋级报告绑定 episode 媒体/状态证据；生产默认启用 |
| `ROBOT_AGENT_SAFETY_POLICY` / `ROBOT_AGENT_SAFETY_SNAPSHOT` | 人员/宠物/隐私结构化门禁策略与最新快照 |
| `VLM_API_URL` / `VLM_API_TOKEN` | OpenAI-compatible VLM 服务 |
| `ROBOT_AGENT_{HEAD,LEFT,RIGHT}_CAMERA_SNAPSHOT_URL` | 三路 RGB 快照地址，缺一路则观察失败 |

适配器协议见 [docs/adapter-contracts.md](docs/adapter-contracts.md)。生产部署示例见 [deploy/robotd.service](deploy/robotd.service)。

## 验证

```bash
pytest -q
python -m compileall -q robot_agent
```

## 模型权重

本运行时不会自动下载、发现或猜测模型。所有 learned policy 必须通过 `ROBOT_AGENT_POLICY_REGISTRY`
显式绑定任务、checkpoint SHA-256、推理后端与部署指纹，并通过 benchmark 门禁。

仓库默认不附带权重。实验室 checkpoint 是参考产物：它们在固定相机、光照、桌面高度与机械臂标定下
采集的少量 episode 上微调，跨硬件或跨场景直接使用大概率失败。外部用户应重新采集数据、标定并训练，
而不是把权重当作即插即用成品。完整说明见 [模型权重范围](../docs/model-weights.md)。

`dock_table`、mission 重试/恢复、急停与 Token 授权属于框架能力，不依赖特定权重，但仍需按现场标定
table profile 与设备映射。

## 项目方向

- [Agent 系统使用说明](docs/user-guide.md)
- [机械臂控制网页使用说明](docs/web-ui-user-guide.md)
- [网页数据集录制说明](docs/dataset-recording.md)
- [本机策略配置记录](docs/local-policy-configuration.md)

- [架构与安全边界](docs/architecture.md)
- [已确认硬件基线](docs/hardware-profile.md)
- [具身智能居家机器人路线图](docs/embodied-home-roadmap.md)
- [M1 Nav2 与硬件在环门禁](docs/m1-nav2-hil.md)
- [ROS 2 bringup 与 HIL 操作流程](docs/ros2-bringup.md)
- [XLeRobot 底盘 bridge 与传输边界](docs/base-bridge.md)
- [Learned policy 注册、校验与晋级](docs/policy-registry.md)
- [统一 episode、benchmark 与策略晋级门禁](docs/data-flywheel.md)
- [插件化离线回放、隔离 rosbag2 与失败片段](docs/offline-replay-runtime.md)
- [可扩展任务目录](docs/task-catalog.md)
- [跨房间移动操作 mission graph](docs/missions.md)
- [人员、宠物、隐私与物理急停 safety interlock](docs/safety-interlock.md)
- [语义实例世界模型与事件回放](docs/semantic-world-model.md)
- [部署目录交叉预检与指纹](docs/deployment-validation.md)
- [可插拔 segmentation 后端协议](docs/segmentation-backends.md)
- [本机 HTTP segmentation 参考插件](docs/segmentation-http.md)
