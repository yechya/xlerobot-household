"""Catalog-constrained goal planner.

The language model may only select and order registered task IDs. It cannot
invent skills, dependencies, preconditions, coordinates, or executable text.
Reusable dependency graphs belong in the reviewed mission catalog.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("robot_agent.goal_planner")


# ============================================================================
# Exceptions
# ============================================================================

class PlanningError(Exception):
    """任务规划错误"""
    pass


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TaskPlan:
    """任务计划"""
    id: str
    skill: str
    goal: str
    depends_on: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    status: str = "pending"  # pending | executing | completed | failed


# ============================================================================
# GoalPlanner
# ============================================================================

class GoalPlanner:
    """Select an ordered subset of registered tasks for dry-run planning."""

    ALLOWED_SKILLS = frozenset()
    _allowed_skills: frozenset[str] | None = None

    @classmethod
    def get_allowed_skills(cls) -> frozenset[str]:
        if cls._allowed_skills is None:
            from .skill_flow import get_allowed_skills
            cls._allowed_skills = get_allowed_skills()
            logger.info(f"[GoalPlanner] 白名单已同步: {sorted(cls._allowed_skills)}")
        return cls._allowed_skills

    def __init__(
        self,
        llm_client: Optional["LLMClient"] = None,
        allowed_skills: frozenset[str] | None = None,
    ):
        """
        Args:
            llm_client: LLM 客户端（可选，默认使用环境变量配置）
        """
        # 延迟导入避免循环依赖
        self._llm_client = llm_client
        # 实例级白名单引用（懒加载）
        self._instance_skills = allowed_skills

    @property
    def allowed_skills(self) -> frozenset[str]:
        """实例级白名单引用（延迟加载）"""
        if self._instance_skills is None:
            self._instance_skills = self.get_allowed_skills()
        return self._instance_skills

    @property
    def llm(self):
        """懒加载 LLM 客户端"""
        if self._llm_client is None:
            from .hardware.vlm import VLMClient
            self._llm_client = VLMClient()
        return self._llm_client

    def decompose(
        self,
        goal: str,
        context: Optional[dict] = None
    ) -> list[TaskPlan]:
        """
        分解高层目标。

        Args:
            goal: 高层目标描述（如 "把桌子收拾干净"）
            context: 额外上下文（如 world_model 摘要）

        Returns:
            TaskPlan 列表

        Raises:
            PlanningError: LLM 调用失败或返回格式无效
        """
        prompt = self._build_planning_prompt(goal, context)

        try:
            response = self.llm.generate(prompt, timeout=30)
        except TimeoutError as e:
            raise PlanningError(f"LLM 调用超时（30秒）: {e}")
        except Exception as e:
            raise PlanningError(f"LLM 调用失败: {e}")

        return self._parse_and_validate(response, goal=goal)

    def _build_planning_prompt(self, goal: str, context: Optional[dict]) -> str:
        """构建规划 prompt"""
        context_str = ""
        if context:
            context_str = f"\n\n额外上下文：\n{json.dumps(context, ensure_ascii=False, indent=2)}"

        return f"""你是一个机器人任务选择器。只能从已注册任务中选择零个或多个任务 ID。

高层目标：{goal}{context_str}

已注册任务 ID（只能逐字使用这些值）：
{', '.join(sorted(self.allowed_skills))}

只返回以下 JSON，不要返回其他字段：
{{
  "task_ids": ["registered_task_id"]
}}

规则：
1. 只能选择已注册任务 ID，不得生成新任务、参数、坐标或动作
2. 按建议执行顺序排列，不得重复
3. 复杂依赖流程必须由已审核 mission catalog 表达，不能在这里生成
4. 没有匹配能力时返回空数组
"""

    def _parse_and_validate(self, response: str, *, goal: str = "") -> list[TaskPlan]:
        """Parse one exact selection object and construct canonical plans."""
        try:
            data = json.loads(response.strip())
        except json.JSONDecodeError as e:
            raise PlanningError(f"JSON 解析失败: {e}")
        if not isinstance(data, dict) or set(data) != {"task_ids"}:
            raise PlanningError("规划结果只能包含 'task_ids' 字段")
        task_ids = data["task_ids"]
        if not isinstance(task_ids, list) or len(task_ids) > 32:
            raise PlanningError("'task_ids' 必须是最多 32 项的数组")
        if not all(isinstance(task_id, str) for task_id in task_ids):
            raise PlanningError("'task_ids' 只能包含字符串")
        if len(task_ids) != len(set(task_ids)):
            raise PlanningError("'task_ids' 不得重复")
        unknown = sorted(set(task_ids) - self.allowed_skills)
        if unknown:
            raise PlanningError(f"任务 ID 不在允许列表中: {unknown}")
        plans = [
            TaskPlan(
                id=f"task_{index}",
                skill=task_id,
                goal=goal,
            )
            for index, task_id in enumerate(task_ids, start=1)
        ]
        logger.info(f"[GoalPlanner] 成功分解为 {len(plans)} 个任务")
        return plans
