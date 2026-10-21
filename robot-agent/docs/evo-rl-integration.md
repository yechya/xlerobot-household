# Evo-RL Offline Integration

本项目采用 Evo-RL 的**离线数据改进**部分，不把 Evo-RL、PyTorch、Transformers 或 GPU 运行时
加载进 `robotd`。当前评估的上游版本为 `MINT-SJTU/Evo-RL` commit
`6f2db449a21e1bac750b996f2e27cac6739aa63f`，其基础版本为 LeRobot 0.4.4。

## 为什么这样接入

Evo-RL 的价值函数、advantage-conditioned policy（ACP）和 human-in-the-loop 闭环适合提升
策略鲁棒性，但 Pi*0.6 价值模型依赖视觉/语言模型和训练设备，不适合成为实时控制进程依赖。
当前 `robotd` 继续保持轻量、默认拒绝和三路 RGB 硬件边界；Evo-RL 只负责离线筛选数据和训练
新 checkpoint。

## 数据边界

桥接器只读取已校验的 `EpisodeManifest`，并导出：

- `episode_success`：成功或失败标签；
- 任务、部署、策略 fingerprint 和失败阶段；
- `dataset_refs`：指向实际 LeRobot 数据集的引用；
- `complementary_info.is_intervention`：当前 manifest 的 episode 级人工介入标记；
- `head_cam`、`left_cam`、`right_cam` 三个普通 RGB 角色。

桥接器不会伪造逐帧 `policy_action` 或逐帧介入 mask。要训练 Evo-RL，必须把索引与引用的
LeRobot 数据集按 `episode_id` 合并；逐帧动作和介入信息应来自原始 HIL 数据。

## 导出一轮数据

在 robot-agent 环境中执行：

```bash
robot-evo-rl-export /var/lib/robot-agent/episodes \
  --task-id fold_simple \
  --output /var/lib/robot-agent/evo-rl/fold-simple-index.json
```

导出文件带有 `episode_set_digest` 和 `export_sha256`，可以作为一次离线训练轮次的输入
清单。`--executor learned_policy` 可用于排除感知、规划和系统失败。

## Evo-RL 训练轮次

1. 用数据 triage 报告排除 `precondition_evidence`、相机、规划和停止链路失败。
2. 把导出索引与 LeRobot HIL 数据集合并，保留 `episode_success`、任务和部署标签。
3. 在独立 Evo-RL 环境中训练价值模型：

   ```bash
   lerobot-value-train --dataset.repo_id=<MERGED_DATASET> \
     --value.type=pistar06 --output_dir=outputs/value_train/<RUN>
   ```

4. 执行 value inference，生成 advantage 和 ACP indicator；再训练支持 task text 的策略：

   ```bash
   lerobot-value-infer --dataset.repo_id=<MERGED_DATASET> \
     --inference.checkpoint_path=outputs/value_train/<RUN> --acp.enable=true
   lerobot-train --dataset.repo_id=<MERGED_DATASET> \
     --acp.enable=true --acp.indicator_field=complementary_info.acp_indicator_<TAG>
   ```

5. 将新 checkpoint、运行时清单和 SHA-256 写入新的 policy registry 条目，保持旧策略不变。
6. 通过 `robotctl dry-run TASK_ID`、RGB 前后置证据和 benchmark 后，才考虑单任务启用。

## 当前限制

- 真实三路相机仍只有普通 RGB；Evo-RL 配置不得添加深度、点云或力觉字段。
- `robot-agent` 不会自动启动训练、不上传数据、不修改 `enabled` 状态。
- 没有逐帧 HIL 数据时，导出索引只能用于 provenance 和 episode-level 筛选，不能冒充完整
  Evo-RL 训练集。
