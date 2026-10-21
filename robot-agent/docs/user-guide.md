# Agent 系统使用说明

本文说明当前 `robot-agent` 的安装、配置、检查、dry-run、真实执行和数据评估流程。
系统已经独立于 Claude Code、MCP 和其他对话式编程工具；所有控制请求通过本机权限受限的
Unix Socket 进入 `robotd`。

## 1. 先看清边界

当前已确认的视觉硬件只有三路普通 RGB 摄像头：

| 角色 | 设备 |
| --- | --- |
| `left_cam` | `/dev/video4` |
| `right_cam` | `/dev/video0` |
| `head_cam` | `/dev/video2` |

系统当前不依赖、也不会自动假设深度相机、LiDAR、点云、IMU、力/力矩传感器、触觉传感器或
MoveIt。RGB 分割只能提供标签、置信度和二维区域，不能证明距离、三维位姿、可达性或接触力。

`robotd` 是任务授权和进程监管边界，不是硬实时控制器。碰撞监控、驱动器限位、MCU/主机
watchdog 和物理急停必须在底层独立成立。

## 2. 安装

在开发机或机器人软件环境中执行：

```bash
cd robot-agent
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

安装后可使用以下入口：

```text
robotd                    启动任务服务
robotctl                  查询状态、规划、观察、授权和执行
robot-policy-check        校验 learned-policy 注册表
robot-policy-materialize  从本地文件计算并封存策略摘要
robot-episode-check       校验 episode 清单
robot-episode-evidence    建立或检查 episode 证据
robot-benchmark-evaluate  生成策略晋级评估
robot-experiment-evaluate 生成 RGB/VLA 对照实验报告
robot-web                 启动本地图形控制台
```

如果没有安装 console script，可以使用等价的模块命令，例如：

```bash
python -m robot_agent.service
python -m robot_agent.cli status
```

## 3.0 软件部署（无真机硬件）

当前阶段可以先部署完整的 `robotd` 软件运行时，不连接摄像头、机械臂、底盘或 ROS 2。安装
脚本创建独立虚拟环境、用户级配置、状态目录和可选的 systemd user service；该 profile 固定
`MOTION_BACKEND=disabled`，并关闭真实策略、benchmark 和 safety interlock 门禁，真实动作不会
被放行：

```bash
cd robot-agent
./deploy/install-software.sh --enable
```

没有 user systemd 会话时省略 `--enable`，再手动启动服务：

```bash
./deploy/install-software.sh
systemctl --user daemon-reload
systemctl --user enable --now robotd-software.service
```

### 3.0.1 接入三路 RGB 摄像头

在确认设备映射为 `/dev/video4`（左臂）、`/dev/video0`（右臂）和 `/dev/video2`（头部）后，
使用带相机选项的安装脚本。它会安装独立的 `robot-camera.service`，通过 V4L2/ffmpeg 维护三路
最新 JPEG，并把观察所需的快照地址写入同一份环境文件：

```bash
cd robot-agent
./deploy/install-software.sh --with-camera --enable
systemctl --user status robot-camera.service
curl http://127.0.0.1:8766/health
```

服务默认只监听 `127.0.0.1:8766`，可通过 `ROBOT_AGENT_CAMERA_*` 变量调整采集分辨率和帧率。
网页预览和 VLM 观察共享后台采集进程；浏览器不会再次打开 `/dev/video*`，因此打开预览不会和
观察或其他相机调用争抢设备。单路相机不可用时，健康接口会给出对应设备和错误信息。

停止相机采集但保留 robotd：

```bash
systemctl --user disable --now robot-camera.service
```

在终端加载同一份客户端配置并检查部署：

```bash
set -a
. "$HOME/.config/robot-agent/robot-agent.env"
set +a
robotctl health
robotctl status
robotctl capabilities
robotctl dry-run pick_up_bottle
```

此部署只验证服务、Unix socket、任务目录、状态机和 dry-run；`observe`/`inspect-plan` 在没有
相机快照地址和 VLM token 时会明确失败。停止或查看日志：

```bash
systemctl --user stop robotd-software.service
systemctl --user status robotd-software.service
```

### 3.0.2 网页控制台

面向操作者的完整网页操作手册见 [机械臂控制网页使用说明](web-ui-user-guide.md)。下面保留最短启动方式。

在已加载 `robot-agent.env` 的终端启动本地网页服务：

```bash
robot-web --host 127.0.0.1 --port 8765
```

打开 <http://127.0.0.1:8765/>，可以查看状态和任务目录，提交目标规划、观察、dry-run、授权、
执行、暂停、停止、急停和复位请求，也可以监管 LeRobot 遥操作数据集录制。网页只调用 `robotd`
的白名单 API，不直接访问执行器；默认只监听本机回环地址。浏览器中的授权 token 只保存在当前
页面内存，刷新页面后需要重新授权。录制配置和操作流程见
[网页数据集录制说明](dataset-recording.md)。

如果使用自定义 socket，启动时显式传入：

```bash
robot-web --socket /path/to/robotd.sock --port 8765
```

## 3.1 生产服务启动

部署机通常使用仓库里的 systemd 单元。先把经过审核的环境文件和目录配置安装到
`/etc/robot-agent/`，确认服务用户对三路摄像头、机械臂串口和状态目录有最小必要权限，
再启动服务：

```bash
sudo install -d -m 0750 /etc/robot-agent
sudo install -m 0640 config/robot-agent.env.example /etc/robot-agent/robot-agent.env
# 编辑 env、task catalog、policy registry、safety policy 和其他部署文件
sudo systemctl daemon-reload
sudo systemctl enable --now robot-watchdog.service
sudo systemctl enable --now robotd.service
robotctl health
```

`robotd.service` 默认以 `robot` 用户运行，并通过 `robot-watchdog.service` 提供 ROS 心跳补充
保护。systemd 单元不会自动把示例配置变成可用硬件配置；真实动作仍受策略、观察、benchmark、
safety snapshot 和一次性 token 共同门控。查看日志：

```bash
sudo journalctl -u robotd.service -f
sudo journalctl -u robot-watchdog.service -f
```

## 3. 开发机首次启动

开发机只做软件联调时，可以使用临时目录和 dry-run 配置。下面的配置只适用于没有真实
动作的开发/测试环境，不要复制到真机生产配置：

```bash
export ROBOT_AGENT_STATE_DIR="$PWD/.runtime/state"
export ROBOT_AGENT_RUNTIME_DIR="$PWD/.runtime/run"
export ROBOT_AGENT_SOCKET="$PWD/.runtime/run/robotd.sock"
export ROBOT_AGENT_EPISODE_DIR="$PWD/.runtime/episodes"
export ROBOT_AGENT_MOTION_BACKEND=disabled
export ROBOT_AGENT_REQUIRE_POLICY_BENCHMARK=false
export ROBOT_AGENT_REQUIRE_POLICY_EVIDENCE=false
export ROBOT_AGENT_REQUIRE_SAFETY_INTERLOCK=false
```

终端 A 启动服务：

```bash
python -m robot_agent.service
```

终端 B 检查服务：

```bash
robotctl health
robotctl status
robotctl capabilities
```

正常情况下会看到 `boundary: standalone-robotd`。`capabilities` 是最重要的总览，包含任务
目录、策略目录、mission 目录、部署 fingerprint 和 benchmark 门禁状态。

## 4. 日常命令

### 4.1 查询和规划

```bash
robotctl status
robotctl health
robotctl capabilities
robotctl world-state
robotctl world-events --since 0 --limit 100
robotctl motion-catalog
robotctl mission-catalog
```

自然语言只用于意图匹配和规划展示，不会直接生成未经审核的硬件动作：

```bash
robotctl plan "把瓶子放进筐里"
robotctl plan "把桌面上的物品整理好"  # 抽象目标自动使用 head_cam 视觉计划
robotctl dry-run "把瓶子放进筐里"
```

`dry-run` 会完整走任务契约、前置证据、任务调度和后置证据管线，最后一步只提交
dummy 指令，不启动真实策略、不驱动机械臂和底盘。涉及操作证据的任务需要三路相机快照和
VLM token；缺少它们时会在证据阶段失败，而不会退回旧的假设成功结果。未匹配的抽象目标会
先用 head_cam 生成只包含已注册任务的 `visual_plan`，再逐个执行同一管线。`robotctl execute
TASK_ID` 不带 `--real` 时也只是 dry-run。

### 4.2 当前四个任务

内置任务 ID 为：

| ID | 作用 |
| --- | --- |
| `fold_simple` | 折叠展开的布/毛巾 |
| `stack_contrast` | 将小碗叠放到盘子上 |
| `pick_up_bottle_towel` | 将瓶子放到毛巾上 |
| `pick_up_bottle` | 将瓶子放入前方筐中 |

任务必须来自 task catalog。未知任务、未注册 executor 或不符合任务参数约束的请求会被拒绝。

## 5. 三路 RGB 观察

`robotctl observe` 会同时读取三路快照，并将它们发送给 OpenAI-compatible VLM。快照 URL
不是 `/dev/video*`，而是由相机服务提供的 HTTP snapshot 地址。配置以下变量：

```bash
export ROBOT_AGENT_HEAD_CAMERA_SNAPSHOT_URL=http://127.0.0.1:8080/head/snapshot
export ROBOT_AGENT_LEFT_CAMERA_SNAPSHOT_URL=http://127.0.0.1:8080/left/snapshot
export ROBOT_AGENT_RIGHT_CAMERA_SNAPSHOT_URL=http://127.0.0.1:8080/right/snapshot
export VLM_API_URL=https://your-vlm.example/v1/chat/completions
export VLM_API_TOKEN='替换为实际令牌'
export VLM_MODEL=your-vision-model
export VLM_TIMEOUT=30
# Reasoning 模型需调大，例如 DeepSeek 用 8192；非 reasoning 保持 2048
export VLM_MAX_TOKENS=2048
```

执行观察：

```bash
robotctl observe "确认桌面目标、人员和障碍物状态"
```

缺少任一路快照 URL、VLM 令牌、相机响应为空或请求超时，观察会失败；在默认配置下，失败的
观察不会满足真实任务的近期观察要求。VLM 是任务语义观察器，不是独立的碰撞安全传感器。

### 5.1 抽象指令视觉计划

综合指令可以先由 `head_cam` 的单张 RGB 图像辅助分解。规划器只允许选择 task catalog 中
已经注册的任务，不会生成坐标、参数、脚本或新的动作；规划阶段不会驱动任何硬件：

```bash
robotctl plan "把桌面上的物品整理好"
# 从返回的 visual_plan.plan_id 复制计划 ID
robotctl arm-plan PLAN_ID --confirm "CONFIRM plan:PLAN_ID"
robotctl execute-plan PLAN_ID --real
```

也可以直接使用 `robotctl inspect-plan` 获取同样的视觉计划，它适合需要明确区分“规划”和
“普通 plan”调用的应用。

这条链路分为五步：

1. `robotd` 只读取 `head_cam` 快照，并把抽象指令、世界模型摘要和已注册任务目录发送给 VLM。
2. VLM 必须返回严格的 `{"steps": [...]}` JSON；未知任务、重复任务、低置信度或额外字段都会被拒绝。
3. `robotd` 将每个步骤绑定到任务 fingerprint、策略 fingerprint、相机图像摘要和计划摘要。
4. 操作者查看返回的 `task_ids`、`confidence`、`reason` 和 `plan_sha256` 后，才执行 `arm-plan`。
5. `execute-plan` 按顺序执行已注册任务，每一步都重新检查绑定和 safety gate，并使用一次性 token。

如果 VLM 无法从图像证明可执行任务，会返回空 `steps`，此计划不能 arm。计划只保存在当前
`robotd` 进程内，服务重启后需要重新 `inspect-plan`；生产配置若要求新鲜观察，arm 前仍须先
完成一次三路 RGB `robotctl observe`。

## 6. 真实机械臂任务

真实执行必须同时满足以下条件：

1. 任务已在 task catalog 中，并且 `executor: learned_policy`。
2. policy registry 中的可执行文件、运行时 lockfile、模型文件和三路 RGB 配置均已审核。
3. 对应策略已启用，且模型摘要与本机文件一致。
4. 生产默认要求通过 benchmark，并绑定 RGB 视频/状态证据。
5. 生产默认要求有效的 safety policy 和新鲜 safety snapshot。
6. `robotctl observe` 在观察 TTL 内成功完成。
7. 操作者明确确认任务 ID。

### 6.1 准备策略注册表

从模板复制一份部署副本，替换真实路径、串口、相机配置和模型目录。示例文件中的全零摘要
和 `enabled: false` 不能直接用于真机：

```bash
cp config/policies.example.yaml /tmp/policies.template.yaml
# 编辑 /tmp/policies.template.yaml，填入真实 executable、runtime_manifest、model_path 等路径
robot-policy-materialize /tmp/policies.template.yaml /etc/robot-agent/policies.yaml \
  --enable pick_up_bottle
robot-policy-check /etc/robot-agent/policies.yaml --schema-only
robot-policy-check /etc/robot-agent/policies.yaml
```

只对已完成离线回放、RGB 前置/后置条件检查和人工复核的任务使用 `--enable`。如果路径或
摘要不匹配，`robotd` 会拒绝授权。

### 6.2 启动前确认

```bash
robotctl capabilities
robotctl status
robotctl observe "确认目标瓶子、筐、人员和工作区状态"
robotctl dry-run pick_up_bottle
```

在 `capabilities` 中检查目标任务、policy fingerprint、deployment fingerprint 和 benchmark
报告是否处于可用状态。不要用手工伪造的 safety snapshot 或 episode 来绕过门禁。

### 6.3 一次真实执行

```bash
robotctl arm pick_up_bottle \
  --confirm "CONFIRM pick_up_bottle"
robotctl execute pick_up_bottle --real
robotctl status
```

`arm` 创建一次性 token，并把它写入权限为 `0600` 的本地 token cache；`execute --real` 使用
后立即删除 token。token 绑定任务参数、策略 fingerprint、部署 fingerprint 和相关报告，
修改这些内容后旧 token 会失效。

不要把 token 放进 shell 历史、日志、环境变量或任务进程参数。自动化场景可使用
`--token-file PATH`，但文件必须保持仅当前用户可读。

## 7. 停止、急停和恢复

这些命令不需要任务 token：

```bash
robotctl pause
robotctl stop
robotctl estop
```

- `pause` 暂停服务状态并停止当前执行请求。
- `stop` 取消当前任务并进入禁用状态。
- `estop` 触发软件急停并锁存状态，同时取消任务和运动目标。

物理急停触发后，先确认机械臂、底盘和周围人员状态，再释放底层急停。软件急停锁存时执行：

```bash
robotctl reset --confirm "RESET ESTOP"
robotctl status
```

软件命令不能替代物理急停、MCU watchdog 或控制器限位。

## 8. 导航和 mission

当前默认 `ROBOT_AGENT_MOTION_BACKEND=disabled`。导航只有在 ROS 2/Nav2 bringup、命名目标、
watchdog 和 HIL 验收全部通过后才能启用；公开接口不接受任意坐标。

启用前先查看命名目录并 dry-run：

```bash
robotctl motion-catalog
robotctl navigate kitchen_table_staging
```

真实导航使用同样的一次性授权：

```bash
robotctl arm-motion navigate kitchen_table_staging \
  --confirm "CONFIRM motion:navigate:kitchen_table_staging"
robotctl navigate kitchen_table_staging --real
```

mission 是版本化的导航、停靠和任务图，不能在运行时由自然语言临时生成：

```bash
robotctl mission-catalog
robotctl plan "整理客厅遥控器"
robotctl mission-dry-run tidy_remote_at_living_room
robotctl arm-mission tidy_remote_at_living_room \
  --confirm "CONFIRM mission:tidy_remote_at_living_room"
robotctl run-mission tidy_remote_at_living_room --real
```

导航配置、bringup 预检和 HIL 流程见 [ros2-bringup.md](ros2-bringup.md) 与
[m1-nav2-hil.md](m1-nav2-hil.md)。不要把 `config/navigation.example.yaml` 的示例坐标直接
安装到真机。

## 9. Episode、证据和评估

真实执行会在 `ROBOT_AGENT_EPISODE_DIR` 下写入不可变 episode manifest。先检查清单：

```bash
robot-episode-check /var/lib/robot-agent/episodes
```

为指定 episode 建立 RGB 视频或状态证据。证据规格必须人工审核，文件路径和时间范围必须落在
episode 边界内：

```bash
robot-episode-evidence index \
  --episode /var/lib/robot-agent/episodes/EPISODE_ID \
  --spec /etc/robot-agent/episode-evidence.yaml

robot-episode-evidence check \
  --episodes /var/lib/robot-agent/episodes \
  --require-kind rgb_video \
  --require-kind robot_state
```

### 9.1 Learned policy 晋级评估

按任务、策略 fingerprint 和环境切片生成 benchmark：

```bash
robot-benchmark-evaluate \
  /etc/robot-agent/benchmark.yaml \
  /var/lib/robot-agent/episodes \
  --output /var/lib/robot-agent/benchmarks/pick-up-bottle.json
```

报告必须来自真实 episode，不能用复制或伪造数据满足样本门槛。失败报告不会被 `robotd` 接受为
生产策略门禁。

### 9.2 RGB/VLA 对照实验

比较 `direct_vla`、`rgb_precondition_vla` 和 `rgb_pre_post_vla`：

```bash
robot-experiment-evaluate \
  config/experiment.example.yaml \
  /var/lib/robot-agent/episodes \
  --output /var/lib/robot-agent/experiments/pick-up-bottle.json
```

使用 `ROBOT_AGENT_EXPERIMENT_MODE` 标记采集臂，默认值是 `rgb_pre_post_vla`。该字段只属于
episode 元数据，不改变授权 token。对照报告要求真实的 `direct_vla` 基线；没有基线时报告明确
失败，不应人工补造。

### 9.3 失败分流

```bash
robot-data-triage \
  /etc/robot-agent/data-triage.yaml \
  /var/lib/robot-agent/episodes \
  /var/lib/robot-agent/triage/report.json
```

相机不可用、图像过期、视角错误和前置条件不明确优先修系统；RGB 观察有效且策略重复失败，
才进入数据集候选队列。

## 10. 可选 RGB 世界模型能力

RGB segmentation 到世界模型的桥接默认关闭，需要显式启动 ROS 节点并提供绝对世界模型路径：

```bash
ros2 run xlerobot_agent_bringup rgb_world_model_bridge --ros-args \
  -p world_model_path:=/var/lib/robot-agent/world.yaml \
  -p segmentation_topic:=/xlerobot/segmentation
```

它只保存标签、置信度、相机帧、二维 ROI、mask digest 和 provenance。`rgb_world_predicate`
是只读执行器，可判断 `object_visible(...)`、`object_on(...)`、`object_in(...)` 等定性事实，
不会驱动机械臂或底盘，也不能替代三维规划和接触检测。

## 11. 新增任务的推荐流程

新增任务应按以下顺序进行：

1. 在 task catalog 增加稳定 ID、别名、RGB 可观察的前置/后置条件和最大运行时间。
2. 保持 `executor: learned_policy`，除非新的执行器已经注册并通过自己的配置校验。
3. 在 policy registry 增加对应模型、运行时、路径摘要和三路相机配置。
4. 先执行 `robotctl capabilities`、`robotctl dry-run TASK_ID` 和离线 replay。
5. 在围挡/受控环境采集真实 episode，建立 RGB 视频和机器人状态证据。
6. 通过 benchmark 和人工复核后，再启用 policy 并开放真实授权。

这套流程使新增任务主要通过目录、模型和证据扩展，不需要为每个任务修改 `robotd` 主流程。

## 12. 常见问题

| 现象 | 处理 |
| --- | --- |
| `robotd` 无法连接 | 检查 `ROBOT_AGENT_SOCKET`、服务进程和 socket 权限；先运行 `robotctl health`。 |
| 观察失败 | 检查三路 snapshot URL、VLM API URL/token、网络和 VLM 超时。 |
| `observation_required` | 重新执行 `robotctl observe ...`，并在 TTL 内重新 arm。 |
| `policy ... disabled` 或摘要不匹配 | 重新核对 registry、模型文件和 `robot-policy-check` 输出。 |
| `benchmark gate failed` | 生成与当前 task/policy/deployment 绑定且未过期的通过报告。 |
| `safety_interlock_blocked` | 让受信任 safety adapter 发布新鲜合法快照；不要手工伪造安全状态。 |
| `motion backend is disabled` | 这是默认行为；完成 ROS 2 bringup/HIL 后才配置 `ros2`。 |
| token 无效或过期 | 重新执行 `arm`；不要复用旧 token。 |
| 急停后不能执行 | 检查底层急停已释放，再执行 `robotctl reset --confirm "RESET ESTOP"`。 |

更详细的边界和验收说明见：

- [架构与权限边界](architecture.md)
- [已确认硬件基线](hardware-profile.md)
- [策略注册与模型摘要](policy-registry.md)
- [数据飞轮和评估](data-flywheel.md)
- [安全 interlock](safety-interlock.md)
- [任务目录](task-catalog.md)
- [ROS 2/Nav2 bringup](ros2-bringup.md)
