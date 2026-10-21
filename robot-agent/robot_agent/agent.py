"""Catalog-driven perception and manipulation agent used by robotd workers.

Base motion is intentionally absent from this API. Navigation and docking are
resolved from reviewed named targets and authorized by the standalone robotd.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from . import skill_flow as _skill_flow_module
from .dialogue_history import DialogueHistory
from .environment_scanner import EnvironmentScanner
from .error_diagnoser import ErrorDiagnoser
from .goal_planner import GoalPlanner, PlanningError, TaskPlan
from .hardware.camera import CameraClient
from .hardware.lerobot import LeRobotClient
from .hardware.vlm import VLMClient, parse_observation_report
from .manipulation import ManipulationExecutor
from .rgb_world_predicate import RgbWorldPredicateExecutor
from .skill_flow import SkillRegistry
from .table_docking import TableDockingExecutor
from .task_backends import TaskBackendRouter
from .task_catalog import TaskCatalog
from .world_model import WorldModel

logger = logging.getLogger("robot_agent")


@dataclass
class ExecutionResult:
    """执行结果"""
    success: bool
    message: str
    data: dict = field(default_factory=dict)


class DummyTaskBackend:
    """Final task boundary used by the pipeline dry-run path."""

    def execute_task(self, task_id: str) -> dict[str, Any]:
        return {
            "success": True,
            "message": f"dry-run dummy instruction accepted: {task_id}",
            "task_id": task_id,
            "dry_run": True,
            "dummy_command": True,
        }


class RobotAgent:
    """
    统一的机器人控制接口。

    Product functions:
    - structured observation and world-model updates
    - catalog-constrained goal selection
    - safety-gated manipulation backend routing

    用法：
        agent = RobotAgent()
        result = agent.execute("fold_simple")
        print(result.message)
    """

    # Class-level lookup remains for compatibility with callers that do not construct an
    # agent. Normal instances use their injected TaskCatalog instead.
    _task_keywords: dict[str, list[str]] | None = None

    @classmethod
    def get_task_keywords(cls) -> dict[str, list[str]]:
        if cls._task_keywords is None:
            cls._task_keywords = _skill_flow_module.get_task_keywords()
        return cls._task_keywords

    @classmethod
    def get_task_map(cls) -> dict[str, str]:
        return _skill_flow_module.get_task_map()

    @classmethod
    def _get_all_aliases(cls) -> list[str]:
        """收集所有任务的关键词，用于简单任务判断"""
        keywords = []
        for aliases in cls.get_task_keywords().values():
            keywords.extend(aliases)
        return keywords

    # 从共享注册表获取任务映射
    TASK_MAP = _skill_flow_module.get_task_map()
    SIMPLE_TASK_PATTERNS = ["折叠", "叠", "放瓶子", "抓取"]

    def __init__(
        self,
        max_adjustments: int = 3,
        max_retries: int = 3,
        persistence_path: str | None = None,
        dry_run: bool = False,
        task_catalog: TaskCatalog | None = None,
        task_backend: Any | None = None,
        enforce_authorization_binding: bool = True,
    ):
        """
        Args:
            max_adjustments: 最大位置调整次数
            max_retries: 最大重试次数
            persistence_path: 世界模型持久化路径
        """
        self.dry_run = dry_run
        self.status = "idle"
        self.task_catalog = task_catalog or TaskCatalog.from_environment()
        self.TASK_MAP = {
            task_id: task.description
            for task_id, task in self.task_catalog.tasks.items()
        }
        self._instance_task_keywords = {
            task_id: list(task.aliases)
            for task_id, task in self.task_catalog.tasks.items()
        }

        # 初始化组件
        self.world_model = WorldModel(persistence_path=persistence_path)

        # Phase 3: 新增组件
        self.goal_planner = GoalPlanner(allowed_skills=frozenset(self.task_catalog.tasks))
        self.dialogue_history = DialogueHistory()
        self.error_diagnoser = ErrorDiagnoser()
        # Phase 5: 环境扫描器（依赖 world_model）
        self.environment_scanner = EnvironmentScanner(world_model=self.world_model)

        self.vlm = VLMClient()
        self.lerobot = LeRobotClient()
        self.camera = CameraClient()

        # Product manipulation path: structured preconditions, one backend attempt,
        # and an independent postcondition observation.
        self.task_backends = TaskBackendRouter(
            self.task_catalog,
            {"learned_policy": task_backend or self.lerobot},
        )
        self.task_backends.register_executor(RgbWorldPredicateExecutor())
        self.task_backends.register_executor(TableDockingExecutor())
        self.manipulation_executor = ManipulationExecutor(
            vlm_client=self.vlm,
            task_backend=self.task_backends,
            task_catalog=self.task_catalog,
            enforce_authorization_binding=enforce_authorization_binding,
            world_model=self.world_model,
        )

        # 初始化技能注册表
        self.skill_registry = SkillRegistry()

        logger.info("[RobotAgent] 初始化完成 (Phase 3, dry_run=%s)", dry_run)

    def _is_dry_run(self) -> bool:
        return bool(getattr(self, "dry_run", False))

    def execute_dry_run_pipeline(
        self,
        task_id: str,
        args: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """Run the real task pipeline while replacing final actuation with a dummy."""
        task = self.task_catalog.get(task_id)
        if task is None:
            return ExecutionResult(False, f"未知任务: {task_id}", {"task_id": task_id})
        from .hardware.authorization import (
            _make_robotd_hardware_authorization,
            _robotd_authorized_context,
        )

        binding = {
            "task_fingerprint": task.fingerprint(),
            "task_executor": task.executor,
            "dry_run": True,
        }
        authorization = _make_robotd_hardware_authorization(
            f"dry-run-{uuid.uuid4().hex}",
            task_id,
            binding,
        )
        with _robotd_authorized_context(authorization):
            result = self.execute(task_id, args)
        result.data.setdefault("dry_run", True)
        result.data["execution_mode"] = "pipeline_dummy"
        result.data["execution_binding"] = binding
        return result

    def match_task(self, goal: str) -> str | None:
        goal_lower = goal.lower()
        candidates = []
        keywords = getattr(self, "_instance_task_keywords", None)
        if keywords is None:
            keywords = self.get_task_keywords()
        for task_id, aliases in keywords.items():
            if task_id.lower() in goal_lower:
                candidates.append((task_id, task_id))
            for alias in aliases:
                if alias.lower() in goal_lower:
                    candidates.append((alias, task_id))

        if not candidates:
            return None

        candidates.sort(key=lambda x: len(x[0]), reverse=True)
        matched_alias, matched_task = candidates[0]
        logger.debug("[match_task] matched '%s' -> %s", matched_alias, matched_task)
        return matched_task

    def execute(
        self,
        task_id: str,
        args: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """
        执行任务（智能模式）。

        自动完成：观察 → 位置调整 → 执行 → 验证

        Args:
            task_id: 任务ID

        Returns:
            ExecutionResult
        """
        self.status = "executing"

        if task_id not in self.TASK_MAP:
            self.status = "failed"
            return ExecutionResult(
                success=False,
                message=f"未知任务: {task_id}",
                data={"available_tasks": list(self.TASK_MAP.keys())}
            )

        logger.info(f"[RobotAgent] execute: {task_id}")

        if self._is_dry_run():
            return self._dry_run_execute(task_id)

        # 记录对话
        self.dialogue_history.add("user", task_id, {"task_id": task_id})

        # 调用智能执行器
        if args:
            result = self.manipulation_executor.execute(task_id, params=args)
        else:
            result = self.manipulation_executor.execute(task_id)

        # 记录响应
        self.dialogue_history.add("assistant", result.final_message, {"success": result.success})

        # 如果失败，进行诊断
        diagnosis = None
        if not result.success and result.steps:
            failed_step = result.steps[-1]
            diagnosis = self.error_diagnoser.diagnose(
                failed_step=failed_step,
                world_model=self.world_model,
                vlm_response=failed_step.message
            )

        execution_result = ExecutionResult(
            success=result.success,
            message=result.final_message,
            data={
                "task_id": result.task_id,
                "total_steps": result.total_steps,
                "completed_steps": result.completed_steps,
                "attempts": result.attempts,
                "adjustments_made": result.adjustments_made,
                "steps": [
                    {
                        "name": s.name,
                        "action": s.action,
                        "success": s.success,
                        "message": s.message,
                        "data": dict(s.data),
                    }
                    for s in result.steps
                ],
                "diagnosis": {
                    "error_type": diagnosis.error_type,
                    "possible_causes": diagnosis.possible_causes,
                    "suggestions": diagnosis.suggestions,
                    "confidence": diagnosis.confidence,
                } if diagnosis else None
            }
        )
        self.status = "idle" if execution_result.success else "failed"
        return execution_result

    def run_complex(self, goal: str) -> ExecutionResult:
        """
        执行复杂目标（使用 GoalPlanner 分解）。

        Args:
            goal: 高层目标描述（如 "把桌子收拾干净"）

        Returns:
            ExecutionResult
        """
        logger.info(f"[RobotAgent] run_complex: {goal}")

        matched_task = self.match_task(goal)
        if matched_task and not self._is_composite_goal(goal):
            return self.execute(matched_task)

        if self._is_dry_run():
            return self._dry_run_goal(goal)

        # 复杂目标 → 任务规划
        try:
            plans = self.plan_goal(goal)
        except PlanningError as e:
            return ExecutionResult(
                success=False,
                message=f"任务规划失败: {e}",
                data={"error": str(e), "error_type": "planning_error"}
            )

        # 按依赖顺序执行任务（拓扑排序）
        # 先构建执行计划：找出无依赖的任务，按层级执行
        results = []
        completed_ids: set = set()
        pending_plans = list(plans)

        while pending_plans:
            # 找出所有依赖都已完成的任务
            ready = [
                p for p in pending_plans
                if all(dep_id in completed_ids for dep_id in p.depends_on)
            ]

            if not ready:
                # 有循环依赖或无法解析的依赖
                for p in pending_plans:
                    results.append({
                        "plan_id": p.id,
                        "skill": p.skill,
                        "success": False,
                        "message": f"无法满足依赖: {p.depends_on}"
                    })
                break

            for plan in ready:
                plan.status = "executing"
                logger.info(f"[RobotAgent] 执行子任务: {plan.id} - {plan.skill}")
                result = self.execute(plan.skill)
                plan.status = "completed" if result.success else "failed"
                if result.success:
                    completed_ids.add(plan.id)
                pending_plans.remove(plan)

                results.append({
                    "plan_id": plan.id,
                    "skill": plan.skill,
                    "success": result.success,
                    "message": result.message
                })

                if not result.success:
                    # 任务失败，记录诊断
                    diagnosis = result.data.get("diagnosis")
                    logger.warning(
                        f"[RobotAgent] 子任务 {plan.id} 失败: {result.message}"
                    )
                    if diagnosis:
                        logger.warning(
                            f"[RobotAgent] 诊断: {diagnosis['possible_causes']}"
                        )
                        logger.warning(
                            f"[RobotAgent] 建议: {diagnosis['suggestions']}"
                        )

        # 汇总结果
        all_success = all(r["success"] for r in results)
        return ExecutionResult(
            success=all_success,
            message="所有子任务完成" if all_success else "部分任务失败",
            data={"plans": results}
        )

    def _is_simple_task(self, goal: str) -> bool:
        """判断是否为简单任务：使用最长匹配优先策略，选择包含最长别名的 task。

        策略说明：
        1. 收集所有 (alias, task_id) 对，按 alias 长度降序排列
        2. 遍历时使用子串匹配，找到第一个即返回
        3. 短语别名优先于短裸词，减少跨 skill 误匹配风险
        4. 不使用裸词别名（避免歧义），aliases 定义时已避免裸词模式
        """
        return self.match_task(goal) is not None

    def _is_composite_goal(self, goal: str) -> bool:
        connectors = ["并", "然后", "再", "同时", "以及", "和", "，", ",", "；", ";"]
        return any(connector in goal for connector in connectors) or bool(
            self._identify_missing_capabilities(goal)
        )

    def plan_goal(self, goal: str) -> list[TaskPlan]:
        matched_task = self.match_task(goal)
        if matched_task and not self._is_composite_goal(goal):
            return [TaskPlan(id="task_1", skill=matched_task, goal=goal)]
        if self._is_dry_run():
            return self._heuristic_household_plan(goal)
        return self.goal_planner.decompose(
            goal,
            context={"world_model": self.world_model.summary()}
        )

    def _dry_run_execute(self, task_id: str) -> ExecutionResult:
        steps = [
            {
                "name": "observe",
                "action": "simulate_observe",
                "success": True,
                "message": "模拟观察桌面与目标物体",
            },
            {
                "name": "check_preconditions",
                "action": "simulate_precondition_check",
                "success": True,
                "message": "模拟确认机器人已贴近桌面且目标可见",
            },
            {
                "name": "adjust_position",
                "action": "simulate_chassis_adjustment",
                "success": True,
                "message": "模拟位置微调，不调用底盘",
            },
            {
                "name": "execute_task",
                "action": "simulate_lerobot_task",
                "success": True,
                "message": f"模拟执行 {task_id}，不调用机械臂脚本",
            },
            {
                "name": "verify",
                "action": "simulate_verification",
                "success": True,
                "message": "模拟验证任务完成",
            },
        ]
        self.dialogue_history.add("user", task_id, {"task_id": task_id, "dry_run": True})
        self.dialogue_history.add(
            "assistant",
            f"dry-run 模拟完成: {task_id}",
            {"success": True, "dry_run": True},
        )
        self.status = "idle"
        return ExecutionResult(
            success=True,
            message=f"dry-run 模拟完成: {task_id}",
            data={
                "dry_run": True,
                "task_id": task_id,
                "total_steps": len(steps),
                "completed_steps": len(steps),
                "attempts": 1,
                "adjustments_made": ["simulated"],
                "steps": steps,
            }
        )

    def _dry_run_goal(self, goal: str) -> ExecutionResult:
        plans = self.plan_goal(goal)
        unsupported = self._identify_missing_capabilities(goal)
        results = []

        for plan in plans:
            if plan.skill not in self.TASK_MAP:
                results.append({
                    "plan_id": plan.id,
                    "skill": plan.skill,
                    "success": False,
                    "message": "任务未注册，dry-run 跳过",
                })
                continue
            result = self._dry_run_execute(plan.skill)
            results.append({
                "plan_id": plan.id,
                "skill": plan.skill,
                "success": result.success,
                "message": result.message,
                "steps": result.data.get("steps", []),
            })

        if not results:
            self.status = "failed"
            return ExecutionResult(
                success=False,
                message="dry-run 未找到可执行子任务",
                data={"dry_run": True, "plans": [], "unsupported": unsupported}
            )

        success = all(r["success"] for r in results) and not unsupported
        self.status = "idle" if success else "partial"
        return ExecutionResult(
            success=success,
            message="dry-run 规划模拟完成" if success else "dry-run 部分可执行，存在缺失能力",
            data={
                "dry_run": True,
                "goal": goal,
                "plans": results,
                "unsupported": unsupported,
            }
        )

    def _heuristic_household_plan(self, goal: str) -> list[TaskPlan]:
        plan_skills: list[str] = []
        household_keywords = [
            "收拾",
            "整理",
            "家务",
            "桌面",
            "清理桌面",
            "桌子收拾",
            "桌子整理",
            "收拾干净",
        ]

        for task_id, aliases in self.get_task_keywords().items():
            if task_id in goal or any(alias in goal for alias in aliases):
                plan_skills.append(task_id)

        if any(keyword in goal for keyword in household_keywords):
            plan_skills.extend(["stack_contrast", "pick_up_bottle", "fold_simple"])

        unique_skills = []
        for skill in plan_skills:
            if skill not in unique_skills:
                unique_skills.append(skill)

        return [
            TaskPlan(id=f"task_{index}", skill=skill, goal=f"dry-run simulate {skill}")
            for index, skill in enumerate(unique_skills, start=1)
        ]

    def _identify_missing_capabilities(self, goal: str) -> list[dict]:
        capability_rules = [
            (
                "semantic_navigation_and_table_docking",
                "P0",
                ["去", "移动到", "走到", "靠近", "桌子旁", "桌边"],
            ),
            ("generic_pick_and_place", "P0", ["收拾", "整理", "清理", "东西", "物品", "杂物"]),
            ("trash_detection_and_disposal", "P0", ["垃圾", "扔掉", "丢掉", "清理"]),
            ("wipe_table", "P1", ["擦", "擦桌", "擦干净", "污渍", "灰尘"]),
            ("multi_object_sorting", "P1", ["分类", "归类", "摆好", "放回原位", "整理"]),
            ("persistent_completion_monitoring", "P1", ["检查", "确认", "完成", "继续"]),
        ]
        missing = []
        for capability, priority, keywords in capability_rules:
            matched = [keyword for keyword in keywords if keyword in goal]
            if matched:
                missing.append({
                    "capability": capability,
                    "priority": priority,
                    "matched_keywords": matched,
                })
        return missing

    def observe(self, instruction: str = "描述当前环境") -> ExecutionResult:
        """视觉观察。"""
        logger.info(f"[RobotAgent] observe: {instruction}")

        if self._is_dry_run():
            return ExecutionResult(
                success=True,
                message=f"dry-run 模拟观察: {instruction}。假设桌面上有小碗、瓶子、抹布和筐。",
                data={"dry_run": True, "instruction": instruction}
            )

        response = self.vlm.observe(
            instruction,
            registered_task_ids=tuple(self.task_catalog.tasks),
        )
        if response.startswith(("观察失败", "观察超时", "观察异常")):
            return ExecutionResult(
                success=False,
                message=response,
                data={"instruction": instruction, "observe_only": True},
            )
        try:
            report = parse_observation_report(
                response,
                registered_task_ids=tuple(self.task_catalog.tasks),
            )
        except ValueError as exc:
            return ExecutionResult(
                success=False,
                message=f"观察结果格式无效: {exc}",
                data={
                    "instruction": instruction,
                    "observe_only": True,
                    "raw_response": response[:4000],
                    "error_type": "observation_schema_error",
                },
            )

        return ExecutionResult(
            success=True,
            message=(
                f"观察完成: next_action={report['next_action']}; "
                f"{report['scene_summary']}"
            ),
            data={
                "instruction": instruction,
                "observe_only": True,
                "observation": report,
                "structured": True,
                "response_json": json.dumps(report, ensure_ascii=False, separators=(",", ":")),
            },
        )

    def panorama(self, task: str | None = None) -> ExecutionResult:
        """全景扫描。"""
        logger.info(f"[RobotAgent] panorama: {task}")

        if self._is_dry_run():
            return ExecutionResult(
                success=True,
                message=f"dry-run 模拟全景扫描: {task or '全面扫描'}",
                data={
                    "dry_run": True,
                    "task": task,
                    "views": ["front", "left", "right", "up", "down"],
                },
            )

        result = self.camera.panorama_scan(task)

        return ExecutionResult(
            success=result.get("success", False),
            message=result.get("stdout", result.get("error", "")),
            data=result
        )

    def get_world_state(self) -> dict:
        """获取世界状态"""
        return self.world_model.to_dict()

    def get_world_events(self, *, since_sequence: int = 0, limit: int = 100) -> dict:
        """Return a bounded page of replayable world-model events."""
        events = self.world_model.get_events(since_sequence=since_sequence)[:limit]
        return {
            "revision": self.world_model.revision,
            "since_sequence": since_sequence,
            "limit": limit,
            "events": [event.to_dict() for event in events],
        }

    def get_dialogue_summary(self) -> str:
        """获取对话摘要"""
        return self.dialogue_history.summarize()

    def get_capabilities(self) -> dict:
        """获取机器人能力"""
        from .skill_flow import get_task
        task_descriptions = {}
        for task_id in self.TASK_MAP.keys():
            info = get_task(task_id)
            if info:
                aliases = ", ".join(info.aliases[:3])
                task_descriptions[task_id] = f"{info.english_description} ({aliases}...)"
            else:
                task_descriptions[task_id] = task_id
        return {
            "tasks": list(self.TASK_MAP.keys()),
            "hardware": {
                "vlm": True,
                "lerobot": True,
                "camera": True,
            },
            "features": {
                "smart_execution": True,
                "goal_planning": True,
                "dialogue_history": True,
                "error_diagnosis": True,
                "dry_run": True,
                "base_motion_via_robotd_named_targets": True,
            },
            "dry_run_active": self._is_dry_run(),
            "status": getattr(self, "status", "unknown"),
            "execution_pipeline": type(self.manipulation_executor).__name__,
            "max_backend_attempts": 1,
        }

    def reset_world(self) -> None:
        """重置世界模型和对话历史"""
        self.world_model.reset()
        self.dialogue_history = DialogueHistory()
        logger.info("[RobotAgent] 世界模型和对话历史已重置")

    def full_scan(self, task: str | None = None) -> ExecutionResult:
        """
        执行完整的环境扫描（Phase 5）。

        调用摄像头云台进行全景扫描，解析 VLM 响应，
        更新世界模型中的物品位置和置信度。

        Args:
            task: 可选的搜索任务（如 "找右边的筐子"）

        Returns:
            ExecutionResult
        """
        logger.info(f"[RobotAgent] full_scan: task={task}")

        if self._is_dry_run():
            return ExecutionResult(
                success=True,
                message=f"dry-run 模拟完整环境扫描: {task or '全面扫描'}",
                data={
                    "dry_run": True,
                    "objects": [
                        {"label": "bowl", "direction": "front", "distance": "reachable"},
                        {"label": "bottle", "direction": "front", "distance": "reachable"},
                        {"label": "rag", "direction": "front", "distance": "reachable"},
                    ],
                    "vlm_response": "dry-run simulated scan",
                    "updated_objects": 0,
                    "new_objects": 3,
                }
            )

        # 记录对话
        self.dialogue_history.add("user", f"环境扫描: {task or '全面扫描'}", {})

        # 执行扫描
        scan_result = self.environment_scanner.full_scan(task=task)

        # 记录响应
        self.dialogue_history.add("assistant", scan_result.message, {
            "objects_found": len(scan_result.objects),
            "updated": scan_result.updated_objects,
            "new": scan_result.new_objects
        })

        return ExecutionResult(
            success=scan_result.success,
            message=scan_result.message,
            data={
                "objects": [obj.to_dict() for obj in scan_result.objects],
                "vlm_response": scan_result.vlm_response,
                "updated_objects": scan_result.updated_objects,
                "new_objects": scan_result.new_objects,
            }
        )

    def decay_confidence(self) -> ExecutionResult:
        """
        对世界模型中的物品应用置信度衰减。

        Returns:
            ExecutionResult
        """
        result = self.world_model.decay_confidence()

        return ExecutionResult(
            success=True,
            message=f"置信度衰减完成: {len(result['decayed'])} 衰减, {len(result['removed'])} 移除",
            data=result
        )
