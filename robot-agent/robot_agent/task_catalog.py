from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from .contract import RUNTIME_STATES


class TaskCatalogError(ValueError):
    pass


@dataclass(frozen=True)
class EvidenceRequirement:
    requirement_id: str
    question: str
    min_confidence: float


@dataclass(frozen=True)
class ManipulationContract:
    preconditions: tuple[EvidenceRequirement, ...]
    postconditions: tuple[EvidenceRequirement, ...]
    require_new_snapshot: bool
    max_policy_attempts: int


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    name: str
    description: str
    executor: str
    executor_config: dict[str, Any]
    aliases: tuple[str, ...]
    args: dict[str, Any]
    requires_observation: bool
    requires_token: bool
    max_runtime_seconds: int
    allowed_modes: tuple[str, ...]
    stop_conditions: tuple[str, ...]
    manipulation: ManipulationContract

    def fingerprint(self) -> str:
        payload = {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "executor": self.executor,
            "executor_config": self.executor_config,
            "aliases": list(self.aliases),
            "args": self.args,
            "requires_observation": self.requires_observation,
            "requires_token": self.requires_token,
            "max_runtime_seconds": self.max_runtime_seconds,
            "allowed_modes": list(self.allowed_modes),
            "stop_conditions": list(self.stop_conditions),
            "manipulation": {
                "preconditions": [
                    requirement.__dict__ for requirement in self.manipulation.preconditions
                ],
                "postconditions": [
                    requirement.__dict__ for requirement in self.manipulation.postconditions
                ],
                "require_new_snapshot": self.manipulation.require_new_snapshot,
                "max_policy_attempts": self.manipulation.max_policy_attempts,
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class TaskCatalog:
    def __init__(self, version: int, tasks: dict[str, TaskDefinition]):
        if version != 1:
            raise TaskCatalogError("task catalog version must be 1")
        if not tasks:
            raise TaskCatalogError("task catalog must define at least one task")
        self.version = version
        self.tasks = dict(tasks)

    @classmethod
    def load(cls, path: str | Path) -> TaskCatalog:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise TaskCatalogError("task catalog must be a YAML object")
        tasks_raw = raw.get("tasks")
        if not isinstance(tasks_raw, dict):
            raise TaskCatalogError("task catalog tasks must be an object")
        tasks = {
            task_id: _parse_task(task_id, value)
            for task_id, value in tasks_raw.items()
        }
        return cls(_catalog_version(raw), tasks)

    @classmethod
    def default(cls) -> TaskCatalog:
        resource = files("robot_agent").joinpath("data/tasks.yaml")
        with resource.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        tasks = {
            task_id: _parse_task(task_id, value)
            for task_id, value in raw["tasks"].items()
        }
        return cls(_catalog_version(raw), tasks)

    @classmethod
    def from_environment(cls) -> TaskCatalog:
        configured = os.environ.get("ROBOT_AGENT_TASK_CATALOG", "").strip()
        return cls.load(configured) if configured else cls.default()

    def get(self, task_id: str) -> TaskDefinition | None:
        return self.tasks.get(task_id)

    def public_view(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tasks": {
                task_id: {
                    "name": task.name,
                    "description": task.description,
                    "executor": task.executor,
                    "max_runtime_seconds": task.max_runtime_seconds,
                    "fingerprint": task.fingerprint(),
                    "preconditions": [
                        requirement.requirement_id
                        for requirement in task.manipulation.preconditions
                    ],
                    "postconditions": [
                        requirement.requirement_id
                        for requirement in task.manipulation.postconditions
                    ],
                }
                for task_id, task in sorted(self.tasks.items())
            },
        }


def _parse_task(task_id: Any, raw: Any) -> TaskDefinition:
    if not isinstance(task_id, str) or not task_id or not isinstance(raw, dict):
        raise TaskCatalogError("task entries require an ID and object")
    aliases = raw.get("aliases")
    if not isinstance(aliases, list) or not aliases or not all(
        isinstance(alias, str) and alias.strip() for alias in aliases
    ):
        raise TaskCatalogError(f"task {task_id!r} aliases must be a non-empty string list")
    safety = raw.get("safety")
    manipulation = raw.get("manipulation")
    if not isinstance(safety, dict) or not isinstance(manipulation, dict):
        raise TaskCatalogError(f"task {task_id!r} requires safety and manipulation objects")
    allowed_modes = _string_tuple(safety, "allowed_modes", task_id)
    unknown_modes = set(allowed_modes) - set(RUNTIME_STATES)
    if unknown_modes:
        raise TaskCatalogError(f"task {task_id!r} has unknown modes: {sorted(unknown_modes)}")
    args = safety.get("args", {})
    if not isinstance(args, dict):
        raise TaskCatalogError(f"task {task_id!r} safety.args must be an object")
    executor = _required_string(raw, "executor", task_id)
    executor_config = raw.get("executor_config", {})
    if not isinstance(executor_config, dict):
        raise TaskCatalogError(f"task {task_id!r} executor_config must be an object")
    try:
        json.dumps(executor_config, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise TaskCatalogError(f"task {task_id!r} executor_config must be JSON-safe") from exc
    return TaskDefinition(
        task_id=task_id,
        name=_required_string(raw, "name", task_id),
        description=_required_string(raw, "description", task_id),
        executor=executor,
        executor_config=dict(executor_config),
        aliases=tuple(dict.fromkeys(alias.strip() for alias in aliases)),
        args=dict(args),
        requires_observation=_required_bool(safety, "requires_observation", task_id),
        requires_token=_required_bool(safety, "requires_token", task_id),
        max_runtime_seconds=_positive_int(safety, "max_runtime_seconds", task_id),
        allowed_modes=allowed_modes,
        stop_conditions=_string_tuple(safety, "stop_conditions", task_id),
        manipulation=ManipulationContract(
            preconditions=_parse_requirements(manipulation, "preconditions", task_id),
            postconditions=_parse_requirements(manipulation, "postconditions", task_id),
            require_new_snapshot=_required_bool(
                manipulation,
                "require_new_snapshot",
                task_id,
            ),
            max_policy_attempts=_positive_int(
                manipulation,
                "max_policy_attempts",
                task_id,
            ),
        ),
    )


def _catalog_version(raw: dict[str, Any]) -> int:
    version = raw.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise TaskCatalogError("task catalog version must be an integer")
    return version


def _parse_requirements(
    raw: dict[str, Any],
    key: str,
    task_id: str,
) -> tuple[EvidenceRequirement, ...]:
    values = raw.get(key)
    if not isinstance(values, list) or not values:
        raise TaskCatalogError(f"task {task_id!r} {key} must be a non-empty list")
    requirements: list[EvidenceRequirement] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, dict):
            raise TaskCatalogError(f"task {task_id!r} {key} entries must be objects")
        requirement_id = _required_string(value, "id", task_id)
        confidence = value.get("min_confidence")
        if requirement_id in seen:
            raise TaskCatalogError(f"task {task_id!r} has duplicate evidence ID {requirement_id!r}")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise TaskCatalogError(f"task {task_id!r} evidence confidence must be in [0, 1]")
        seen.add(requirement_id)
        requirements.append(
            EvidenceRequirement(
                requirement_id=requirement_id,
                question=_required_string(value, "question", task_id),
                min_confidence=float(confidence),
            )
        )
    return tuple(requirements)



def _required_string(raw: dict[str, Any], key: str, task_id: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TaskCatalogError(f"task {task_id!r} requires non-empty {key}")
    return value.strip()


def _required_bool(raw: dict[str, Any], key: str, task_id: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise TaskCatalogError(f"task {task_id!r} requires boolean {key}")
    return value


def _positive_int(raw: dict[str, Any], key: str, task_id: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TaskCatalogError(f"task {task_id!r} requires positive integer {key}")
    return value


def _string_tuple(raw: dict[str, Any], key: str, task_id: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise TaskCatalogError(f"task {task_id!r} requires non-empty string list {key}")
    return tuple(value)


__all__ = [
    "EvidenceRequirement",
    "ManipulationContract",
    "TaskCatalog",
    "TaskCatalogError",
    "TaskDefinition",
]
