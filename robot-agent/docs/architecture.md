# Standalone Runtime Architecture

## Ownership

`robotd` 是唯一允许创建真实任务进程的用户态服务。`robotctl`、Web UI、语音入口或其他应用都只能通过权限受限的 Unix Socket 请求操作。

```text
User interfaces
      |
robotctl / local SDK
      |
Unix socket request-response
      |
robotd
  |-- state machine
  |-- task allowlist
  |-- one-time authorization
  |-- observation freshness
  |-- append-only audit
  |-- read-only semantic world queries
  |-- process supervisor
  |-- bounded mission graph
  |-- structured household safety interlock
  `-- named motion gateway
          |
       adapters / ROS 2 actions
          |
 collision monitor / base bridge
          |
 XLeRobot host watchdog / motor drivers
```

## Concurrency

- 规划、状态查询和观察可以通过独立连接进入服务。
- 同一时间最多一个真实动作任务。
- 导航只接受版本化目录中的命名目标，公开接口不接受任意地图坐标。
- 真实任务运行在独立进程组；另一个连接可以执行 `stop` 或 `estop`。
- `stop` 终止任务并回到 `DISABLED`；`estop` 保持锁存，直到显式 `reset`。
- 机械运动不做并发优化。并行只用于无动作感知、模型预热和规划。
- mission 同一时刻只执行一个 action；每个子 action 仍由 Nav2 或任务进程自己的取消/租约边界监管。

## Trust Boundaries

- Unix Socket 权限默认 `0600`，没有 TCP/SSE 控制入口。
- token 不进入日志，不通过环境变量或进程参数传给任务进程。
- 真实硬件适配器未配置时默认拒绝。
- VLM 输出不是安全传感器，只能提供任务语义和辅助验证。
- 人员/宠物/隐私 snapshot 只提供离散上层门禁；连续防撞和物理急停必须独立于 `robotd`。
- 即使 `robotd` 崩溃，低层 watchdog 也必须在限定时间内停车。
- 当前 ZMQ host 无认证，M1 只允许同机 loopback；开放远程连接前必须增加认证传输。

## Remaining Production Work

- 将 JSON token store 升级为 SQLite WAL，并记录状态转换事件。
- 使用 Linux peer credentials 或独立 operator group 做本地身份绑定。
- 在目标 ROS 2 环境验证 lifecycle、diagnostics、Nav2 action cancellation 和 docking。
- 为恢复、断电、摄像头丢失、网络中断和执行器超时增加故障注入测试。
- 建立真实机器人硬件在环测试台和发布门禁。
- 为世界事件日志增加按部署容量设计的归档、checkpoint 压缩和可选异机签名存证。
