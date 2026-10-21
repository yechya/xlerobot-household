from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable, Protocol

import yaml


class MissionError(ValueError):
    pass


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_NODE_TYPES = frozenset({"sequence", "selector", "retry", "action"})
_ACTIONS = frozenset({"navigate", "dock", "dock_table", "task"})
_RECOVERY_ACTIONS = frozenset({"observe", "task"})


@dataclass(frozen=True)
class MissionAction:
    node_id: str
    action: str
    target: str
    timeout_seconds: float


@dataclass(frozen=True)
class MissionRecovery:
    retry_node_id: str
    action: MissionAction


@dataclass(frozen=True)
class MissionDefinition:
    mission_id: str
    name: str
    description: str
    aliases: tuple[str, ...]
    requires_observation: bool
    max_runtime_seconds: int
    max_node_visits: int
    graph: dict[str, Any]
    actions: tuple[MissionAction, ...]
    recoveries: tuple[MissionRecovery, ...] = ()

    def fingerprint(self) -> str:
        payload = {
            "mission_id": self.mission_id,
            "name": self.name,
            "description": self.description,
            "aliases": list(self.aliases),
            "requires_observation": self.requires_observation,
            "max_runtime_seconds": self.max_runtime_seconds,
            "max_node_visits": self.max_node_visits,
            "graph": self.graph,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class MissionCatalog:
    def __init__(self, version: int, missions: dict[str, MissionDefinition]):
        if version != 1:
            raise MissionError("mission catalog version must be 1")
        self.version = version
        self.missions = dict(missions)
        owners: dict[str, str] = {}
        for mission in self.missions.values():
            for phrase in (mission.mission_id, *mission.aliases):
                normalized = phrase.casefold().strip()
                owner = owners.setdefault(normalized, mission.mission_id)
                if owner != mission.mission_id:
                    raise MissionError(
                        f"mission phrase {phrase!r} is shared by {owner!r} and "
                        f"{mission.mission_id!r}"
                    )

    @classmethod
    def load(cls, path: str | Path) -> "MissionCatalog":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict) or set(raw) != {"version", "missions"}:
            raise MissionError("mission catalog has missing or unexpected fields")
        missions_raw = raw.get("missions")
        if not isinstance(missions_raw, dict):
            raise MissionError("mission catalog missions must be an object")
        return cls(
            raw.get("version"),
            {
                mission_id: _parse_mission(mission_id, value)
                for mission_id, value in missions_raw.items()
            },
        )

    @classmethod
    def empty(cls) -> "MissionCatalog":
        return cls(1, {})

    @classmethod
    def default(cls) -> "MissionCatalog":
        resource = files("robot_agent").joinpath("data/missions.yaml")
        with resource.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls(
            raw.get("version"),
            {
                mission_id: _parse_mission(mission_id, value)
                for mission_id, value in raw.get("missions", {}).items()
            },
        )

    @classmethod
    def from_environment(cls) -> "MissionCatalog":
        configured = os.environ.get("ROBOT_AGENT_MISSION_CATALOG", "").strip()
        return cls.load(configured) if configured else cls.default()

    def get(self, mission_id: str) -> MissionDefinition | None:
        return self.missions.get(mission_id)

    def match(self, text: str) -> MissionDefinition | None:
        normalized = text.casefold()
        candidates: list[tuple[int, str, MissionDefinition]] = []
        for mission in self.missions.values():
            for phrase in (mission.mission_id, *mission.aliases):
                if phrase.casefold() in normalized:
                    candidates.append((len(phrase), mission.mission_id, mission))
        if not candidates:
            return None
        return max(candidates, key=lambda value: (value[0], value[1]))[2]

    def public_view(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "missions": {
                mission_id: {
                    "name": mission.name,
                    "description": mission.description,
                    "fingerprint": mission.fingerprint(),
                    "max_runtime_seconds": mission.max_runtime_seconds,
                    "actions": [
                        {
                            "node_id": action.node_id,
                            "action": action.action,
                            "target": action.target,
                        }
                        for action in mission.actions
                    ],
                }
                for mission_id, mission in sorted(self.missions.items())
            },
        }


@dataclass(frozen=True)
class MissionActionResult:
    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)


class MissionActionProvider(Protocol):
    def execute(self, action: MissionAction, deadline: float) -> MissionActionResult: ...


@dataclass(frozen=True)
class MissionResult:
    success: bool
    message: str
    trace: tuple[dict[str, Any], ...]
    node_visits: int


@dataclass(frozen=True)
class _NodeResult:
    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)


class MissionExecutor:
    def __init__(
        self,
        *,
        provider: MissionActionProvider,
        cancelled: Callable[[], bool] = lambda: False,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.provider = provider
        self.cancelled = cancelled
        self.clock = clock

    def execute(self, mission: MissionDefinition) -> MissionResult:
        self._mission = mission
        self._deadline = self.clock() + mission.max_runtime_seconds
        self._visits = 0
        self._trace: list[dict[str, Any]] = []
        result = self._run_node(mission.graph)
        return MissionResult(
            success=result.success,
            message=result.message,
            trace=tuple(self._trace),
            node_visits=self._visits,
        )

    def _run_node(self, node: dict[str, Any]) -> _NodeResult:
        self._visits += 1
        node_id = str(node["id"])
        node_type = str(node["type"])
        started = self.clock()
        if self.cancelled():
            return self._record(node_id, node_type, started, _NodeResult(False, "mission canceled"))
        if self._visits > self._mission.max_node_visits:
            return self._record(
                node_id,
                node_type,
                started,
                _NodeResult(False, "mission node visit limit exceeded"),
            )
        if started >= self._deadline:
            return self._record(
                node_id,
                node_type,
                started,
                _NodeResult(False, "mission deadline exceeded"),
            )
        if node_type == "sequence":
            result = _NodeResult(True, "sequence completed")
            for child in node["children"]:
                result = self._run_node(child)
                if not result.success:
                    break
        elif node_type == "selector":
            failures: list[str] = []
            result = _NodeResult(False, "selector exhausted")
            for child in node["children"]:
                result = self._run_node(child)
                if result.success:
                    break
                failures.append(result.message)
            if not result.success:
                result = _NodeResult(False, "; ".join(failures) or "selector exhausted")
        elif node_type == "retry":
            result = _NodeResult(False, "retry exhausted")
            recovery = next(
                (
                    entry.action
                    for entry in self._mission.recoveries
                    if entry.retry_node_id == node_id
                ),
                None,
            )
            attempts = int(node["max_attempts"])
            for attempt in range(1, attempts + 1):
                result = self._run_node(node["child"])
                if result.success or self.cancelled():
                    break
                if attempt >= attempts:
                    break
                if recovery is not None:
                    recovery_started = self.clock()
                    recovery_result = self._run_action(recovery)
                    self._trace.append(
                        {
                            "node_id": recovery.node_id,
                            "type": "recovery",
                            "action": recovery.action,
                            "target": recovery.target,
                            "success": recovery_result.success,
                            "message": recovery_result.message,
                            "duration_seconds": max(
                                0.0, self.clock() - recovery_started
                            ),
                            "data": dict(recovery_result.data),
                        }
                    )
                    if not recovery_result.success:
                        result = _NodeResult(
                            False,
                            "recovery action failed before retry: "
                            + recovery_result.message,
                            recovery_result.data,
                        )
                        break
        else:
            action = MissionAction(
                node_id=node_id,
                action=str(node["action"]),
                target=str(node["target"]),
                timeout_seconds=float(node["timeout_seconds"]),
            )
            result = self._run_action(action)
        return self._record(node_id, node_type, started, result)

    def _run_action(self, action: MissionAction) -> _NodeResult:
        deadline = min(self._deadline, self.clock() + action.timeout_seconds)
        try:
            executed = self.provider.execute(action, deadline)
            data = dict(executed.data)
            encoded = json.dumps(data, sort_keys=True, separators=(",", ":"))
            if len(encoded.encode()) > 256 * 1024:
                raise MissionError("mission action result exceeds 256 KiB")
            result = _NodeResult(
                bool(executed.success),
                str(executed.message),
                data,
            )
        except Exception as exc:
            result = _NodeResult(False, f"mission action failed: {type(exc).__name__}: {exc}")
        if self.clock() > deadline:
            result = _NodeResult(False, "mission action exceeded its deadline", result.data)
        return result

    def _record(
        self,
        node_id: str,
        node_type: str,
        started: float,
        result: _NodeResult,
    ) -> _NodeResult:
        self._trace.append(
            {
                "node_id": node_id,
                "type": node_type,
                "success": result.success,
                "message": result.message,
                "duration_seconds": max(0.0, self.clock() - started),
                "data": dict(result.data),
            }
        )
        return result


def _parse_mission(mission_id: Any, raw: Any) -> MissionDefinition:
    _identifier(mission_id, "mission_id")
    expected = {
        "name",
        "description",
        "aliases",
        "requires_observation",
        "max_runtime_seconds",
        "max_node_visits",
        "graph",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise MissionError(f"mission {mission_id!r} has missing or unexpected fields")
    aliases = raw.get("aliases")
    if not isinstance(aliases, list) or not aliases or not all(
        isinstance(alias, str) and 0 < len(alias.strip()) <= 128 for alias in aliases
    ):
        raise MissionError(f"mission {mission_id!r} aliases are invalid")
    if len(aliases) != len(set(alias.casefold().strip() for alias in aliases)):
        raise MissionError(f"mission {mission_id!r} aliases must be unique")
    requires_observation = raw.get("requires_observation")
    if not isinstance(requires_observation, bool):
        raise MissionError(f"mission {mission_id!r} requires_observation must be boolean")
    max_runtime = _bounded_int(raw.get("max_runtime_seconds"), "max_runtime_seconds", 3600)
    max_visits = _bounded_int(raw.get("max_node_visits"), "max_node_visits", 256)
    graph = raw.get("graph")
    actions, recoveries = _inspect_graph(graph)
    return MissionDefinition(
        mission_id=mission_id,
        name=_text(raw.get("name"), "name"),
        description=_text(raw.get("description"), "description"),
        aliases=tuple(alias.strip() for alias in aliases),
        requires_observation=requires_observation,
        max_runtime_seconds=max_runtime,
        max_node_visits=max_visits,
        graph=graph,
        actions=actions,
        recoveries=recoveries,
    )


def _inspect_graph(
    root: Any,
) -> tuple[tuple[MissionAction, ...], tuple[MissionRecovery, ...]]:
    identifiers: set[str] = set()
    actions: list[MissionAction] = []
    recoveries: list[MissionRecovery] = []

    def inspect(
        node: Any,
        depth: int,
        *,
        inside_retry: bool = False,
        recovery_declared: bool = False,
    ) -> None:
        if depth > 12:
            raise MissionError("mission graph depth exceeds 12")
        if not isinstance(node, dict):
            raise MissionError("mission graph nodes must be objects")
        node_id = node.get("id")
        node_type = node.get("type")
        _identifier(node_id, "mission node id")
        if node_id in identifiers:
            raise MissionError(f"mission node ID is duplicated: {node_id!r}")
        identifiers.add(node_id)
        if len(identifiers) > 64:
            raise MissionError("mission graph contains more than 64 nodes")
        if node_type not in _NODE_TYPES:
            raise MissionError(f"unsupported mission node type: {node_type!r}")
        if node_type in {"sequence", "selector"}:
            if set(node) != {"id", "type", "children"}:
                raise MissionError(f"mission {node_type} node has invalid fields")
            children = node.get("children")
            if not isinstance(children, list) or not 1 <= len(children) <= 16:
                raise MissionError(f"mission {node_type} requires 1..16 children")
            for child in children:
                inspect(
                    child,
                    depth + 1,
                    inside_retry=inside_retry,
                    recovery_declared=recovery_declared,
                )
            return
        if node_type == "retry":
            allowed_fields = {"id", "type", "max_attempts", "child", "recovery"}
            if not {"id", "type", "max_attempts", "child"} <= set(node) or not set(
                node
            ) <= allowed_fields:
                raise MissionError("mission retry node has invalid fields")
            attempts = node.get("max_attempts")
            if (
                isinstance(attempts, bool)
                or not isinstance(attempts, int)
                or not 1 <= attempts <= 3
            ):
                raise MissionError("mission retry max_attempts must be in [1, 3]")
            recovery_action = _parse_recovery(node.get("recovery"), node_id)
            if recovery_action is not None:
                recovery_id = f"{node_id}.recovery"
                if recovery_id in identifiers:
                    raise MissionError(f"mission node ID is duplicated: {recovery_id!r}")
                identifiers.add(recovery_id)
            inspect(
                node.get("child"),
                depth + 1,
                inside_retry=True,
                recovery_declared=recovery_action is not None,
            )
            if recovery_action is not None:
                actions.append(recovery_action)
                recoveries.append(MissionRecovery(node_id, recovery_action))
            return
        if set(node) != {"id", "type", "action", "target", "timeout_seconds"}:
            raise MissionError("mission action node has invalid fields")
        action = node.get("action")
        target = node.get("target")
        if action not in _ACTIONS:
            raise MissionError(f"unsupported mission action: {action!r}")
        _identifier(target, "mission action target")
        if inside_retry and action == "task" and not recovery_declared:
            raise MissionError(
                "task actions cannot be descendants of retry nodes without a recovery action"
            )
        timeout = node.get("timeout_seconds")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or not 0 < float(timeout) <= 600
        ):
            raise MissionError("mission action timeout must be in (0, 600]")
        actions.append(MissionAction(node_id, action, target, float(timeout)))

    inspect(root, 1)
    return tuple(actions), tuple(recoveries)


def _parse_recovery(raw: Any, retry_node_id: str) -> MissionAction | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise MissionError("mission retry recovery must be an object")
    if set(raw) != {"action", "target", "timeout_seconds"}:
        raise MissionError("mission retry recovery has invalid fields")
    action = raw.get("action")
    if action not in _RECOVERY_ACTIONS:
        raise MissionError(f"unsupported recovery action: {action!r}")
    target = raw.get("target")
    _identifier(target, "mission recovery target")
    timeout = raw.get("timeout_seconds")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or not 0 < float(timeout) <= 600
    ):
        raise MissionError("mission recovery timeout must be in (0, 600]")
    return MissionAction(f"{retry_node_id}.recovery", action, target, float(timeout))


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise MissionError(f"{label} is not a safe identifier")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not 0 < len(value.strip()) <= 500:
        raise MissionError(f"mission {label} is invalid")
    return value.strip()


def _bounded_int(value: Any, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise MissionError(f"mission {label} must be in [1, {maximum}]")
    return value


__all__ = [
    "MissionAction",
    "MissionActionProvider",
    "MissionActionResult",
    "MissionCatalog",
    "MissionDefinition",
    "MissionError",
    "MissionExecutor",
    "MissionRecovery",
    "MissionResult",
]
