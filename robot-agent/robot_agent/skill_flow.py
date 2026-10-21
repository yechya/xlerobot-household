"""SkillFlow — 声明式 Skill 路由协议。

基于 vector-os-nano 的 skill.py 设计，适配 lerobot 硬件。

同时也作为任务配置的单一数据源（Single Source of Truth）。
所有 skill/task 的注册、白名单、TASK_MAP 都集中在这里定义，
避免多处硬编码导致不同步。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from .task_catalog import TaskCatalog, TaskDefinition

logger = logging.getLogger("skill_flow")

# ============================================================================
# Task Registry — 任务配置单一数据源
# ============================================================================


@dataclass(frozen=True)
class TaskSafetyMetadata:
    """Default-deny allowlist metadata for one hardware task."""

    task_id: str
    name: str
    aliases: tuple[str, ...]
    args: dict[str, Any]
    requires_observation: bool
    requires_token: bool
    max_runtime_seconds: int
    allowed_modes: tuple[str, ...]
    stop_conditions: tuple[str, ...]

@dataclass(frozen=True)
class TaskInfo:
    task_id: str
    english_description: str
    aliases: tuple[str, ...]
    timeout: int
    planner_skill: str = ""
    name: str = ""

    def __post_init__(self) -> None:
        if not self.planner_skill:
            object.__setattr__(self, "planner_skill", self.task_id)
        if not self.name:
            object.__setattr__(self, "name", self.english_description)

    @classmethod
    def from_definition(cls, definition: TaskDefinition) -> "TaskInfo":
        return cls(
            task_id=definition.task_id,
            english_description=definition.description,
            aliases=definition.aliases,
            timeout=definition.max_runtime_seconds,
            planner_skill=definition.task_id,
            name=definition.name,
        )


def get_all_task_ids() -> list[str]:
    """返回所有已注册的任务 ID"""
    return list(_task_catalog().tasks)


def get_task(task_id: str) -> Optional[TaskInfo]:
    """通过 task_id 获取任务元信息"""
    definition = _task_catalog().get(task_id)
    return TaskInfo.from_definition(definition) if definition else None


def get_allowed_skills() -> frozenset[str]:
    """返回 GoalPlanner 可用的 skill 白名单（用于 LLM 规划验证）"""
    return frozenset(_task_catalog().tasks)


def get_task_map() -> dict[str, str]:
    """返回 task_id → english_description 的映射（供 LeRobot 执行用）"""
    return {
        task_id: definition.description
        for task_id, definition in _task_catalog().tasks.items()
    }


def get_task_keywords() -> dict[str, list[str]]:
    """返回 task_id → aliases 的映射（供 agent 意图识别用）"""
    return {
        task_id: list(definition.aliases)
        for task_id, definition in _task_catalog().tasks.items()
    }


def get_task_safety_metadata(task_id: str) -> Optional[TaskSafetyMetadata]:
    """Return safety metadata for one allowlisted task, or None if unknown."""
    definition = _task_catalog().get(task_id)
    if definition is None:
        return None
    return _safety_metadata(definition)


def get_all_task_safety_metadata() -> dict[str, TaskSafetyMetadata]:
    """Return safety metadata for every canonical hardware task."""
    return {
        task_id: _safety_metadata(definition)
        for task_id, definition in _task_catalog().tasks.items()
    }


_TASK_CATALOG: TaskCatalog | None = None


def _task_catalog() -> TaskCatalog:
    global _TASK_CATALOG
    if _TASK_CATALOG is None:
        _TASK_CATALOG = TaskCatalog.from_environment()
    return _TASK_CATALOG


def _safety_metadata(definition: TaskDefinition) -> TaskSafetyMetadata:
    return TaskSafetyMetadata(
        task_id=definition.task_id,
        name=definition.name,
        aliases=definition.aliases,
        args=dict(definition.args),
        requires_observation=definition.requires_observation,
        requires_token=definition.requires_token,
        max_runtime_seconds=definition.max_runtime_seconds,
        allowed_modes=definition.allowed_modes,
        stop_conditions=definition.stop_conditions,
    )


# ============================================================================
# @skill Decorator
# ============================================================================

def skill(
    cls=None,
    *,
    aliases: list[str] | None = None,
    direct: bool = False,
    auto_steps: list[str] | None = None,
):
    """
    装饰器：将类标记为 Skill 并声明路由元数据。

    Args:
        aliases: 触发此 skill 的关键词（中英文）
        direct: True=立即执行无需 LLM 规划
        auto_steps: 默认 skill 链，如 ["scan", "detect", "pick"]

    Example:
        @skill(aliases=["抓", "grab"], auto_steps=["observe", "pick"])
        class PickSkill:
            ...
    """
    def wrapper(cls):
        cls.__skill_aliases__ = aliases or []
        cls.__skill_direct__ = direct
        cls.__skill_auto_steps__ = auto_steps or []
        return cls

    if cls is not None:
        cls.__skill_aliases__ = []
        cls.__skill_direct__ = False
        cls.__skill_auto_steps__ = []
        return cls

    return wrapper


# ============================================================================
# Protocols
# ============================================================================

class SkillContext:
    """Skill 执行所需的上下文"""
    def __init__(
        self,
        world_model: Any = None,
        arm: Any = None,
        gripper: Any = None,
        vlm_observer: Any = None,
        config: dict | None = None,
    ):
        self.world_model = world_model
        self.arm = arm
        self.gripper = gripper
        self.vlm_observer = vlm_observer
        self.config = config or {}


@dataclass
class SkillResult:
    """Skill 执行结果"""
    success: bool
    message: str = ""
    data: dict = field(default_factory=dict)
    error: str = ""


class Skill(Protocol):
    """Skill 接口"""
    name: str
    description: str
    parameters: dict
    preconditions: list[str]
    postconditions: list[str]
    effects: dict

    def execute(self, params: dict, context: SkillContext) -> SkillResult: ...


    def validate(self, params: dict) -> tuple[bool, str | None]:
        """验证参数是否有效"""
        ...


@dataclass(frozen=True)
class SkillInvocation:
    """One catalog-bound request sent to a concrete skill executor."""

    skill_id: str
    params: Mapping[str, Any] = field(default_factory=dict)
    trace_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.skill_id, str) or not self.skill_id.strip():
            raise ValueError("skill_id must be a non-empty string")
        if not isinstance(self.params, Mapping):
            raise TypeError("skill invocation params must be a mapping")
        if not isinstance(self.trace_id, str):
            raise TypeError("skill invocation trace_id must be a string")
        object.__setattr__(self, "skill_id", self.skill_id.strip())
        object.__setattr__(self, "params", dict(self.params))


@dataclass(frozen=True)
class SkillExecutionContext:
    """Reviewed runtime context made available to a skill executor."""

    task: TaskDefinition
    world_model: Any = None
    observer: Any = None
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.config, Mapping):
            raise TypeError("skill execution config must be a mapping")
        object.__setattr__(self, "config", dict(self.config))


@dataclass(frozen=True)
class SkillExecutionResult:
    """Normalized result returned by every concrete skill executor."""

    success: bool
    message: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)
    error: str = ""
    retryable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.data, Mapping):
            raise TypeError("skill execution data must be a mapping")
        object.__setattr__(self, "data", dict(self.data))

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.data)
        payload.setdefault("success", bool(self.success))
        if self.message:
            payload.setdefault("message", self.message)
        if self.error:
            payload.setdefault("error", self.error)
        payload.setdefault("retryable", bool(self.retryable))
        return payload


class SkillExecutor(Protocol):
    """Stable execution port for learned and deterministic skills."""

    executor_id: str

    def validate_config(self, config: Mapping[str, Any]) -> tuple[bool, str | None]: ...

    def execute(
        self,
        invocation: SkillInvocation,
        context: SkillExecutionContext,
    ) -> SkillExecutionResult: ...


# ============================================================================
# SkillMatch
# ============================================================================

@dataclass
class SkillMatch:
    """匹配结果"""
    skill_name: str
    direct: bool
    auto_steps: list[str]
    extracted_arg: str = ""  # 匹配后剩余的文本


# ============================================================================
# SkillRegistry
# ============================================================================

class SkillRegistry:
    """Skill 注册表 — 通过别名匹配进行路由"""

    def __init__(self):
        self._skills: dict[str, Any] = {}
        self._alias_map: dict[str, tuple[str, bool, list[str]]] = {}

    def register(self, skill_instance: Any) -> None:
        """注册 skill 实例"""
        name = skill_instance.name
        self._skills[name] = skill_instance

        aliases = getattr(skill_instance, "__skill_aliases__", [])
        direct = getattr(skill_instance, "__skill_direct__", False)
        auto_steps = getattr(skill_instance, "__skill_auto_steps__", [])

        # 注册 skill 名本身
        self._alias_map[name.lower()] = (name, direct, auto_steps)

        # 注册所有别名
        for alias in aliases:
            self._alias_map[alias.lower()] = (name, direct, auto_steps)

        logger.debug(f"[SkillRegistry] 注册 skill: {name}, 别名: {len(aliases)}")

    def get(self, name: str) -> Any | None:
        """通过名称获取 skill"""
        return self._skills.get(name)

    def match(self, user_input: str) -> SkillMatch | None:
        """
        匹配用户输入到 skill。

        1. 精确匹配
        2. 前缀匹配（最长匹配优先）

        Returns:
            SkillMatch 或 None（无匹配）
        """
        text = user_input.strip().lower()

        # 1. 精确匹配
        if text in self._alias_map:
            name, direct, auto = self._alias_map[text]
            return SkillMatch(name, direct, auto, "")

        # 2. 前缀匹配
        best_match: tuple[str, bool, list[str]] | None = None
        best_len = 0
        best_arg = ""

        for alias, (name, direct, auto) in self._alias_map.items():
            if text.startswith(alias) and len(alias) > best_len:
                best_match = (name, direct, auto)
                best_len = len(alias)
                best_arg = text[len(alias):].strip()

        if best_match and best_len > 0:
            name, direct, auto = best_match
            return SkillMatch(name, direct, auto, best_arg)

        return None

    def list_skills(self) -> list[str]:
        """列出所有注册的 skill 名称"""
        return list(self._skills.keys())

    def to_schemas(self) -> list[dict]:
        """序列化所有 skill 的 schema（供 LLM 使用）"""
        schemas = []
        for s in self._skills.values():
            schema = {
                "name": s.name,
                "description": s.description,
                "parameters": getattr(s, "parameters", {}),
                "preconditions": list(getattr(s, "preconditions", [])),
                "postconditions": list(getattr(s, "postconditions", [])),
                "effects": dict(getattr(s, "effects", {})),
            }
            aliases = getattr(s, "__skill_aliases__", [])
            auto_steps = getattr(s, "__skill_auto_steps__", [])
            if aliases:
                schema["aliases"] = aliases
            if auto_steps:
                schema["auto_steps"] = auto_steps
            schemas.append(schema)
        return schemas


# ============================================================================
# Skill Base Classes
# ============================================================================

class BaseSkill:
    """Skill 基类"""

    name: str = ""
    description: str = ""
    parameters: dict = {}
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {}
    failure_modes: list[str] = []

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        raise NotImplementedError

    def validate(self, params: dict) -> tuple[bool, str | None]:
        """验证参数"""
        required = []
        for pname, pdef in self.parameters.items():
            if "required" in pdef and pdef["required"] and pname not in params:
                required.append(pname)
        if required:
            return False, f"缺少必需参数: {required}"
        return True, None
