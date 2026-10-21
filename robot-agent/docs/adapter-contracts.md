# Hardware Adapter Contracts

生产运行时不从固定工程目录导入控制脚本。所有真实硬件能力必须通过下面的显式命令适配器提供，并由 `robotd` 的受监管任务进程调用。

## Learned Policy Adapter

配置：`ROBOT_AGENT_POLICY_REGISTRY`。

旧的任意 shell task runner 已移除。`robotd` 从版本化 registry 选择与任务绑定的策略，校验模型 SHA-256，并在受监管进程组内以结构化 argv 启动直接可执行文件。

要求：

- policy 必须绑定一个运行时白名单 `task_id`，并显式启用。
- operator token 绑定 policy ID、完整配置 fingerprint 和模型 digest。
- 正常完成返回 `0`，其他返回码表示失败。
- stdout/stderr 不得包含 token、原始图像或用户隐私数据。
- policy 进程必须响应 `SIGTERM`，退出时进入机械臂安全保持状态。
- 模型 hash 只证明身份；离线回放与 manipulation HIL 仍是启用前置条件。

## Skill Executor Contract

任务目录选择的是 executor ID，运行时通过统一的 `SkillInvocation`、
`SkillExecutionContext` 和 `SkillExecutionResult` 调用具体执行器。当前
`learned_policy` 通过兼容适配器接入已有 LeRobot 后端，因此现有四个任务的
行为不变；后续确定性技能可以直接注册同一个 `SkillExecutor` 接口，不需要
修改 `ManipulationExecutor` 或 `robotd` 的任务流程。

执行器必须接收目录绑定的 task context，并返回结构化的成功状态、错误码、
可重试标记和扩展数据。未注册的 executor、context 与任务不匹配、以及无效
执行结果都会被路由层拒绝或转换为失败结果。

执行器还可以实现 `validate_config(config)` 声明自己的 `executor_config` schema。
没有声明校验器的执行器只允许空配置；当前 `learned_policy` 兼容适配器遵循这一
默认拒绝规则。配置会进入任务 fingerprint，并在每次真实执行前重新校验。

当前附带的 `rgb_world_predicate` 只读执行器不会驱动机械或底盘；它只能消费世界模型
中的 RGB 结构化事实，作为任务图中的观察判断节点。

## Chassis Driver

配置：`ROBOT_AGENT_CHASSIS_COMMAND`。

调用：

```text
<command> <direction> <distance-or-angle> <speed>
```

运行时已经限制方向、单步距离和角度，驱动端仍必须再次校验。底盘碰撞安全不能依赖 VLM；至少需要保险杠、深度相机或激光雷达的本地停止链路。

## Panorama Driver

配置：`ROBOT_AGENT_PANORAMA_COMMAND`。

调用：

```text
<command> --output-json <path> [--task <instruction>]
```

JSON 文件应包含：

```json
{
  "angles": [
    {"pan": 0, "tilt": 0}
  ]
}
```

VLM 文本可输出到 stdout。一次调用必须同时完成图像采集和分析，禁止为了元数据重复移动云台。

## Emergency Stop

配置：`ROBOT_AGENT_EMERGENCY_STOP_COMMAND`。

该命令无参数，必须幂等、在五秒内返回，并直接触达低层控制器。它是软件急停补充，不能替代物理急停按钮、驱动器使能线或 MCU watchdog。

## ROS 2 Migration

CLI 适配器是迁移期协议。当前已提供可选的 ROS 2 motion gateway：

- Nav2 `NavigateToPose` 用于命名位置导航，Open Navigation `DockRobot` 用于命名 dock。
- gateway 仅在配置启用时导入 ROS 2；没有 ROS 2 的开发机保持可安装和可测试。
- 任意坐标不进入公开 IPC，目录版本和目标名进入审计与一次性授权绑定。
- 当前不提供 MoveIt、深度几何或力控适配器；新增适配器前必须先确认真实硬件接口。
- `ros2_control` 与 MCU watchdog 负责实时控制和失联停车。
- `robotd` 只调用 action/service，并保持授权、状态机、审计和任务取消职责。
