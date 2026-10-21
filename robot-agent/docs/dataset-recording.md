# 网页数据集录制说明

网页控制台可以通过 `robotd` 启动和监管 LeRobot 遥操作数据集录制。网页不会直接执行
`lerobot-record`，也不会拼接 shell 字符串；录制参数经过校验后由 `robotd` 以结构化参数列表启动。

## 1. 当前硬件绑定

录制后端从 `ROBOT_AGENT_POLICY_REGISTRY` 读取从臂和三路 RGB 相机，不再维护第二份设备映射。
当前已确认的绑定是：

| 角色 | 设备 |
| --- | --- |
| 左从臂 | `/dev/ttyACM0` |
| 右从臂 | `/dev/ttyACM1` |
| 左主臂 | `/dev/ttyACM2` |
| 右主臂 | `/dev/ttyACM3` |
| 头部 RGB | `/dev/video2` |
| 左臂 RGB | `/dev/video4` |
| 右臂 RGB | `/dev/video0` |

USB 设备重新插拔后，先重新确认角色再修改策略注册表或录制配置。设备节点存在只表示系统发现了
设备，不能证明左右臂或相机角色没有发生交换。

## 2. 配置

参考 [dataset-recording.example.yaml](../config/dataset-recording.example.yaml) 创建本机配置，并在
`robot-agent.env` 中设置：

```bash
ROBOT_AGENT_DATASET_RECORDING_CONFIG=$HOME/.config/robot-agent/dataset-recording.yaml
```

配置分为四部分：

- `executable`、`workspace` 和 `storage_root` 指向当前 LeRobot 运行环境与数据根目录；
- `teleoperator` 只描述双臂主臂，双臂从臂和相机仍由策略注册表提供；
- `defaults` 和 `limits` 控制网页默认值及后端可接受的最大录制量；
- `camera_service` 定义录制前暂停、退出后恢复 `robot-camera.service` 的结构化命令。

修改配置后必须重启 `robotd`。状态面板显示“已就绪”前，不要尝试录制。

## 3. 为什么录制会暂停相机预览

`robot-camera` 平时独占 `/dev/video0`、`/dev/video2` 和 `/dev/video4`，网页预览与 VLM 观察共享
它的输出。LeRobot 录制需要直接读取同一组三路设备，因此录制流程按下面的顺序转移所有权：

1. `robotd` 确认自身处于 `DRY_RUN`，并且没有任务或运动执行器占用硬件；
2. 停止 `robot-camera.service`；
3. 启动 `lerobot-record`，连接双臂主从机械臂和三路 RGB 相机；
4. 录制完成、失败或被停止后，重新启动 `robot-camera.service`。

录制期间网页预览、VLM 观察、dry-run 和真机任务都会被阻断。相机恢复完成后这些入口才重新可用。

## 4. 网页操作

打开 <http://127.0.0.1:8765/>，找到“数据集录制”区域。

1. 输入只包含小写字母、数字、点、短横线或下划线的数据集名称；
2. 输入准确、统一的任务描述；
3. 设置本次新增回合数、单回合录制秒数和回合间复位秒数；
4. 新数据集不要勾选“续录已有数据集”，已有数据集追加回合时必须勾选；
5. 点击“验证配置”，确认设备、目录、参数和续录元数据都通过；
6. 点击“开始录制”并确认，随后通过双臂主臂完成示教；
7. 在回合间复位时间内把物体和环境恢复到下一回合的初始状态；
8. 等待目标回合数完成，或点击“停止录制”。

页面每两秒更新新增回合、数据集总回合、总帧数、存储路径和录制器日志。网页刷新不会停止正在
运行的录制，重新打开页面后会从 `robotd` 恢复当前状态。

## 5. 新建和续录规则

数据默认保存到：

```text
~/.cache/huggingface/lerobot/<dataset_owner>/<dataset_name>
```

新建模式发现同名目录时会拒绝启动，系统不会删除或覆盖旧数据。续录模式要求目录存在，并验证：

- `meta/info.json` 是有效且大小受限的普通 JSON 文件；
- `robot_type` 与当前双臂从臂注册一致；
- 数据集 FPS 与录制配置一致；
- `head_cam`、`left_cam`、`right_cam` 的图像尺寸与当前策略注册表一致。

页面中的“新增回合”是本次要追加的数量，不是续录后的数据集总回合数。

## 6. 停止与日志

“停止录制”会向 LeRobot 录制进程组发送终止信号；超过宽限时间仍未退出时再终止整个进程组。
网页的“暂停”“停止”和“急停”也会先停止数据集录制，再执行对应的 `robotd` 状态转换。

录制日志保存在：

```text
~/.local/state/robot-agent/dataset-recording/
```

页面显示日志末尾，完整日志仍保留在上述目录。正常停止可能只保存已经完成并提交的回合；正在
录制但尚未执行 `save_episode` 的回合是否保留由 LeRobot 决定，因此停止后要核对总回合数。

## 7. 常见问题

### 页面显示“未配置”或“策略注册表不可用”

检查环境变量和两个 YAML 文件，然后重启服务：

```bash
test -r "$ROBOT_AGENT_DATASET_RECORDING_CONFIG"
test -r "$ROBOT_AGENT_POLICY_REGISTRY"
systemctl --user restart robotd-software.service
```

### 验证提示设备不可用

```bash
ls -l /dev/ttyACM0 /dev/ttyACM1 /dev/ttyACM2 /dev/ttyACM3
ls -l /dev/video0 /dev/video2 /dev/video4
```

先确认设备角色，不要只按当前编号直接修改配置。

### 录制结束后相机没有恢复

页面日志会显示相机恢复错误。手动检查并恢复：

```bash
systemctl --user start robot-camera.service
systemctl --user status robot-camera.service
curl http://127.0.0.1:8766/health
```

### 续录被拒绝

先核对页面显示的数据集路径和 `meta/info.json`。不要为了绕过校验而删除元数据；需要保留旧数据
时，改用新的数据集名称，或修复策略注册表与实际硬件的配置偏差。
