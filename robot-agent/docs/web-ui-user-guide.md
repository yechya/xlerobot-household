# 机械臂控制网页使用说明

这是一份给操作者看的网页控制台说明。网页控制台的作用是把常用的
`robotd` 操作放到浏览器里完成，不需要直接输入命令。

网页默认只监听本机地址 `127.0.0.1`，因此只能在运行机器人软件的这台电脑上打开。
它不会直接连接机械臂，也不会绕过 `robotd` 的任务目录、授权、观察和执行门禁。

## 1. 你需要启动什么

网页本身只是操作界面，至少需要同时运行：

1. `robotd`：任务服务，负责状态、规划、dry-run、授权和执行。
2. `robot-web`：网页服务，负责浏览器界面。

如果要使用网页里的实时相机预览，还需要运行 `robot-camera`。相机服务负责持续采集
三路 RGB 画面，网页和 VLM 观察都从它读取，不会各自抢占 `/dev/video*`。数据集录制会由
`robotd` 临时停止相机服务，让 LeRobot 直接读取三路设备，结束后自动恢复。

当前确认的相机角色如下：

| 网页名称 | 设备 |
| --- | --- |
| 头部 | `/dev/video2` |
| 左臂 | `/dev/video4` |
| 右臂 | `/dev/video0` |

## 2. 已部署环境启动

先打开一个终端，加载部署配置并启动任务服务：

```bash
set -a
. "$HOME/.config/robot-agent/robot-agent.env"
set +a

systemctl --user start robotd-software.service
```

如果需要相机预览，再启动相机服务：

```bash
systemctl --user start robot-camera.service
```

检查服务状态：

```bash
robotctl health
robotctl status
```

然后打开第二个终端启动网页：

```bash
$HOME/.local/share/robot-agent/.venv/bin/robot-web --host 127.0.0.1 --port 8765
```

浏览器打开：

<http://127.0.0.1:8765/>

关闭网页服务时，在运行 `robot-web` 的终端按 `Ctrl-C`。这不会停止 `robotd` 或相机服务。

## 3. 未使用 systemd 的开发启动

如果没有安装用户级服务，先在项目目录创建开发环境变量：

```bash
cd /path/to/project/robot-agent
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

终端 B 启动网页：

```bash
python -m robot_agent.webui --host 127.0.0.1 --port 8765
```

如果网页和 `robotd` 使用了非默认 socket，显式指定同一个路径：

```bash
python -m robot_agent.webui \
  --socket "$ROBOT_AGENT_SOCKET" \
  --host 127.0.0.1 \
  --port 8765
```

## 4. 页面各区域怎么用

### 4.1 连接状态和运行状态

页面右上角显示网页是否能连接 `robotd`。右侧的“运行状态”显示：

- 当前状态，例如 `IDLE`、`DRY_RUN` 或执行中状态；
- 当前任务和执行器是否正在运行；
- safety interlock 是否阻断；
- 当前缓存的视觉计划数量；
- 当前 `robotd` 会话的末尾标识。

看到“robotd 不可用”时，先不要点击执行，回到终端检查 `robotctl health`。

### 4.2 目标指令

在“目标指令”框输入自然语言目标，例如：

```text
把瓶子放进筐里
```

有三个入口：

- **生成计划**：解析目标。对于抽象目标，系统可能使用头部相机生成视觉计划，页面会显示拆分后的已注册任务。
- **目标 Dry-run**：走完整任务调度、前置/后置检查和授权流程，但最后只调用 dummy 执行器，不驱动机械臂。
- **观察**：读取当前三路相机并请求 VLM 返回环境观察结果，不执行任务。

目标文本只会匹配已注册任务，不会直接变成任意 shell 命令或新的硬件动作。

### 4.3 已注册任务

“已注册任务”列表来自当前 task catalog。每个任务有三个按钮：

- **Dry-run**：只验证该任务的完整执行链路；
- **授权**：为该任务生成一次性授权 token；
- **执行**：使用刚刚生成的 token 执行同一个任务。

授权和执行必须是同一个任务。刷新网页后，token 不会恢复，需要重新授权。

当前常见任务包括：

| 任务 ID | 含义 |
| --- | --- |
| `fold_simple` | 折叠布或毛巾 |
| `stack_contrast` | 将小碗叠放到盘子上 |
| `pick_up_bottle_towel` | 将瓶子放到毛巾上 |
| `pick_up_bottle` | 将瓶子放入前方筐中 |

### 4.4 视觉计划

对“整理桌面上的物品”这类综合目标，先点击“生成计划”。如果 VLM 能从头部相机画面
判断出可执行的已注册任务，页面会列出步骤、原因、置信度和计划摘要。

使用视觉计划时：

1. 检查步骤是否符合现场目标；
2. 点击“授权计划”；
3. 再点击“执行计划”。

视觉计划只允许使用 task catalog 中已有任务。没有步骤、任务未知或置信度不足时，计划不能授权。

### 4.5 相机预览

点击“打开预览”后，页面显示头部、左臂和右臂三路画面；再次点击可关闭预览。

网页读取的是相机服务提供的本地 HTTP stream，不会在浏览器中再次打开摄像头设备。因此，
打开或关闭预览不会中断 VLM 观察，也不会产生第二个摄像头采集进程。数据集录制是例外：它需要
直接读取 `/dev/video0`、`/dev/video2` 和 `/dev/video4`，因此录制期间预览和 VLM 观察会暂停。

如果显示“相机服务未就绪”：

```bash
systemctl --user status robot-camera.service
curl http://127.0.0.1:8766/health
```

### 4.6 数据集录制

“数据集录制”区域可以新建或续录 LeRobot 数据集。填写数据集名称、任务描述、本次新增回合数、
单回合秒数和复位秒数后，先点击“验证配置”，再点击“开始录制”。

录制开始前要求 `robotd` 处于 `DRY_RUN`，且没有任务或运动执行器正在运行。开始后页面会关闭
相机预览，并显示新增回合、总回合、总帧数、存储路径和录制器日志。录制结束、失败或点击停止后，
相机服务会自动恢复。

新建模式不会覆盖同名目录。向已有数据集追加回合时必须勾选“续录已有数据集”，后端会检查
robot type、FPS 和三路相机尺寸。完整配置与故障处理见
[网页数据集录制说明](dataset-recording.md)。

### 4.7 暂停、停止、急停和复位

- **暂停**：请求当前任务暂停，具体是否支持由执行器决定；
- **停止**：请求当前任务停止并结束执行；
- **急停**：发送急停请求，页面会要求再次确认；
- **复位**：只有在底层急停已经释放后，输入正确的复位确认才可恢复 `robotd` 状态。

网页急停是软件层请求，不能替代机械臂控制器上的物理急停、驱动器限位或底层 watchdog。
发现人员、设备或机械臂状态异常时，应优先使用现场物理急停。

## 5. 推荐操作顺序

### 5.1 只验证软件链路

1. 启动 `robotd` 和 `robot-web`；
2. 确认右上角显示“robotd 已连接”；
3. 点击任务列表中的“Dry-run”；
4. 在“最近响应”和“RESULT DETAIL”中查看结果；
5. 确认结果成功后再测试观察或视觉计划。

### 5.2 使用实时相机和 VLM

1. 启动 `robot-camera.service`；
2. 点击“打开预览”，确认三个角色画面没有错位；
3. 在配置中确认 `VLM_API_URL`、`VLM_API_TOKEN` 和 `VLM_MODEL` 已设置；
4. 输入观察指令并点击“观察”；
5. 观察成功后再生成计划或 dry-run。

### 5.3 录制数据集

1. 确认四个机械臂串口和三路相机角色与配置一致；
2. 打开三路预览，确认画面角色，然后关闭预览；
3. 填写录制参数并点击“验证配置”；
4. 准备好首回合场景和物理急停；
5. 点击“开始录制”，使用双臂主臂示教；
6. 完成后核对新增回合数、总帧数和日志，并确认相机服务已恢复。

### 5.4 真机执行前

1. 先完成一次观察，确认目标和工作区状态；
2. 先执行同一任务的 dry-run；
3. 确认任务、相机角色和策略配置正确；
4. 点击“授权”，检查授权结果；
5. 确认现场无人进入机械臂工作区后，才点击“执行”。

当前软件部署 profile 默认关闭真实策略和真实运动后端。即使网页上出现“执行”按钮，
没有有效的 policy registry、硬件适配器和其他生产门禁时，`robotd` 仍会拒绝真实动作。

## 6. 常见问题

### 页面打不开

确认网页终端仍在运行，并访问 `http://127.0.0.1:8765/`。如果端口被占用，换一个端口：

```bash
$HOME/.local/share/robot-agent/.venv/bin/robot-web --host 127.0.0.1 --port 8767
```

### 页面显示“robotd 不可用”

```bash
robotctl health
systemctl --user status robotd-software.service
```

开发环境还要确认网页和 `robotd` 使用同一个 `ROBOT_AGENT_SOCKET`。

### 观察失败：`VLM_API_TOKEN is not configured`

这是 VLM 配置缺失，不是网页故障。加载正确的环境文件，确认以下变量存在后重启 `robotd`：

```bash
echo "$VLM_API_URL"
echo "${VLM_API_TOKEN:+token is set}"
echo "$VLM_MODEL"
```

### 预览显示“未连接”或画面为空

检查相机服务和设备映射：

```bash
systemctl --user status robot-camera.service
curl http://127.0.0.1:8766/health
ls -l /dev/video0 /dev/video2 /dev/video4
```

不要在另一个程序中同时打开这些设备；网页本身不直接占用设备。

### 数据集录制显示“未配置”

检查 `ROBOT_AGENT_DATASET_RECORDING_CONFIG` 和 `ROBOT_AGENT_POLICY_REGISTRY` 是否指向可读的
YAML 文件，修改后重启 `robotd-software.service`。详细检查见
[网页数据集录制说明](dataset-recording.md)。

### Dry-run 被拒绝

查看页面的“RESULT DETAIL”，或者在终端执行：

```bash
robotctl capabilities
robotctl dry-run TASK_ID
```

常见原因是任务未注册、policy registry 未配置或任务前置观察未满足。Dry-run 的拒绝结果
表示系统没有完成完整执行链路，不应直接跳过并尝试真机执行。

## 7. 相关文档

- [完整 Agent 使用说明](user-guide.md)
- [任务目录说明](task-catalog.md)
- [策略注册与校验](policy-registry.md)
- [网页数据集录制说明](dataset-recording.md)
- [已确认硬件基线](hardware-profile.md)
