"""Closed-loop RGB table-edge docking for the holonomic service base.

The controller never moves on a low-confidence observation. Every step is
bounded, every observation is recorded, and convergence requires a second
observation after the base has settled.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import yaml

from .skill_flow import SkillExecutionContext, SkillExecutionResult, SkillInvocation

TABLE_DOCKING_EXECUTOR = "table_docking"

DEFAULT_TABLE_DOCKING_CONFIG: dict[str, float | int] = {
    "standoff_m": 0.45,
    "distance_tolerance_m": 0.06,
    "lateral_tolerance_m": 0.04,
    "yaw_tolerance_deg": 3.0,
    "min_confidence": 0.6,
    "max_step_m": 0.12,
    "max_step_yaw_deg": 6.0,
    "max_steps": 12,
    "deadline_seconds": 60.0,
    "settle_seconds": 0.6,
}


class TableDockingError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Observation / actuation contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TableObservation:
    """One visual estimate of the table edge in the robot base frame.

    distance_m: forward distance to the nearest table/cabinet edge.
    lateral_offset_m: edge midpoint offset to the robot's right (positive right).
    yaw_error_deg: heading error; positive means the robot must rotate CCW.
    """

    distance_m: float | None
    lateral_offset_m: float | None
    yaw_error_deg: float | None
    confidence: float
    reason: str = ""
    snapshot_sha256: str = ""
    captured_at: float = 0.0

    def finite(self) -> bool:
        values = (self.distance_m, self.lateral_offset_m, self.yaw_error_deg)
        return all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in values
        )

    def usable(self, min_confidence: float) -> bool:
        return (
            self.finite()
            and math.isfinite(float(self.confidence))
            and float(self.confidence) >= float(min_confidence)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "distance_m": self.distance_m,
            "lateral_offset_m": self.lateral_offset_m,
            "yaw_error_deg": self.yaw_error_deg,
            "confidence": self.confidence,
            "reason": self.reason,
            "snapshot_sha256": self.snapshot_sha256,
            "captured_at": self.captured_at,
        }


@dataclass(frozen=True)
class DockStep:
    forward_m: float = 0.0
    right_m: float = 0.0
    yaw_deg: float = 0.0

    def is_zero(self) -> bool:
        return self.forward_m == 0.0 and self.right_m == 0.0 and self.yaw_deg == 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "forward_m": self.forward_m,
            "right_m": self.right_m,
            "yaw_deg": self.yaw_deg,
        }


@dataclass(frozen=True)
class DockStepResult:
    success: bool
    message: str = ""


class TableObserver(Protocol):
    def observe(self) -> TableObservation: ...


class DockActuator(Protocol):
    def step(self, command: DockStep) -> DockStepResult: ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TableDockingConfig:
    standoff_m: float = 0.45
    distance_tolerance_m: float = 0.06
    lateral_tolerance_m: float = 0.04
    yaw_tolerance_deg: float = 3.0
    min_confidence: float = 0.6
    max_step_m: float = 0.12
    max_step_yaw_deg: float = 6.0
    max_steps: int = 12
    deadline_seconds: float = 60.0
    settle_seconds: float = 0.6

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> TableDockingConfig:
        values: dict[str, Any] = dict(DEFAULT_TABLE_DOCKING_CONFIG)
        if raw is not None:
            if not isinstance(raw, Mapping):
                raise TableDockingError("table docking config must be a mapping")
            unknown = sorted(set(raw) - set(DEFAULT_TABLE_DOCKING_CONFIG))
            if unknown:
                raise TableDockingError(
                    f"unknown table docking config keys: {', '.join(unknown)}"
                )
            values.update(raw)
        float_keys = (
            "standoff_m",
            "distance_tolerance_m",
            "lateral_tolerance_m",
            "yaw_tolerance_deg",
            "min_confidence",
            "max_step_m",
            "max_step_yaw_deg",
            "deadline_seconds",
            "settle_seconds",
        )
        parsed: dict[str, float] = {}
        for key in float_keys:
            value = values[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise TableDockingError(f"table docking {key} must be a finite number")
            parsed[key] = float(value)
        steps = values["max_steps"]
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 64:
            raise TableDockingError("table docking max_steps must be in [1, 64]")

        config = cls(max_steps=steps, **parsed)  # type: ignore[arg-type]
        config.validate()
        return config

    def validate(self) -> None:
        if not 0 < self.standoff_m <= 2.0:
            raise TableDockingError("standoff_m must be in (0, 2.0]")
        if not 0 < self.distance_tolerance_m <= 0.5:
            raise TableDockingError("distance_tolerance_m must be in (0, 0.5]")
        if not 0 < self.lateral_tolerance_m <= 0.5:
            raise TableDockingError("lateral_tolerance_m must be in (0, 0.5]")
        if not 0 < self.yaw_tolerance_deg <= 30:
            raise TableDockingError("yaw_tolerance_deg must be in (0, 30]")
        if not 0 <= self.min_confidence <= 1:
            raise TableDockingError("min_confidence must be in [0, 1]")
        if not 0 < self.max_step_m <= 0.3:
            raise TableDockingError("max_step_m must be in (0, 0.3]")
        if not 0 < self.max_step_yaw_deg <= 45:
            raise TableDockingError("max_step_yaw_deg must be in (0, 45]")
        if not 1 <= self.max_steps <= 64:
            raise TableDockingError("max_steps must be in [1, 64]")
        if not 0 < self.deadline_seconds <= 600:
            raise TableDockingError("deadline_seconds must be in (0, 600]")
        if not 0 <= self.settle_seconds <= 10:
            raise TableDockingError("settle_seconds must be in [0, 10]")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DockingResult:
    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

_EPSILON_M = 1e-6
_EPSILON_DEG = 1e-6


def _clamp(value: float, limit: float) -> float:
    if limit <= 0:
        return 0.0
    return max(-limit, min(limit, value))


class TableDockingController:
    """Bounded, fail-closed visual approach to a table edge."""

    def __init__(
        self,
        *,
        observer: TableObserver,
        actuator: DockActuator,
        config: TableDockingConfig | None = None,
        cancelled: Callable[[], bool] = lambda: False,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.observer = observer
        self.actuator = actuator
        self.config = config or TableDockingConfig()
        self.config.validate()
        self.cancelled = cancelled
        self.clock = clock
        self.sleep = sleep

    def run(self) -> DockingResult:
        started = self.clock()
        deadline = started + self.config.deadline_seconds
        observations: list[dict[str, Any]] = []
        commands: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        unusable_streak = 0
        last_errors: tuple[float, float, float] | None = None
        no_progress_streak = 0
        final: TableObservation | None = None

        for step_index in range(1, self.config.max_steps + 1):
            if self.cancelled():
                return self._result(
                    False, "docking canceled", observations, commands, trace, final
                )
            if self.clock() >= deadline:
                return self._result(
                    False,
                    "docking deadline exceeded",
                    observations,
                    commands,
                    trace,
                    final,
                )
            try:
                observation = self.observer.observe()
            except Exception as exc:  # noqa: BLE001 - observation must fail closed
                return self._result(
                    False,
                    f"table observation failed: {type(exc).__name__}: {exc}",
                    observations,
                    commands,
                    trace,
                    final,
                )
            observations.append(observation.to_dict())
            final = observation

            if not observation.usable(self.config.min_confidence):
                unusable_streak += 1
                trace.append(
                    {
                        "step": step_index,
                        "phase": "observe",
                        "observation": observation.to_dict(),
                        "success": False,
                        "message": observation.reason or "observation below confidence floor",
                    }
                )
                if unusable_streak >= 2:
                    return self._result(
                        False,
                        "table edge is not reliably visible; no motion was commanded",
                        observations,
                        commands,
                        trace,
                        final,
                    )
                continue
            unusable_streak = 0

            assert observation.distance_m is not None
            assert observation.lateral_offset_m is not None
            assert observation.yaw_error_deg is not None
            forward_error = observation.distance_m - self.config.standoff_m
            lateral_error = observation.lateral_offset_m
            yaw_error = observation.yaw_error_deg
            within_tolerance = (
                abs(forward_error) <= self.config.distance_tolerance_m
                and abs(lateral_error) <= self.config.lateral_tolerance_m
                and abs(yaw_error) <= self.config.yaw_tolerance_deg
            )
            if within_tolerance:
                if self.config.settle_seconds > 0:
                    self.sleep(self.config.settle_seconds)
                if self.clock() >= deadline:
                    return self._result(
                        False,
                        "docking deadline exceeded during settle",
                        observations,
                        commands,
                        trace,
                        final,
                    )
                try:
                    confirmation = self.observer.observe()
                except Exception as exc:  # noqa: BLE001
                    return self._result(
                        False,
                        f"confirming observation failed: {type(exc).__name__}: {exc}",
                        observations,
                        commands,
                        trace,
                        final,
                    )
                observations.append(confirmation.to_dict())
                final = confirmation
                confirmed = (
                    confirmation.usable(self.config.min_confidence)
                    and confirmation.distance_m is not None
                    and confirmation.lateral_offset_m is not None
                    and confirmation.yaw_error_deg is not None
                    and abs(confirmation.distance_m - self.config.standoff_m)
                    <= self.config.distance_tolerance_m
                    and abs(confirmation.lateral_offset_m)
                    <= self.config.lateral_tolerance_m
                    and abs(confirmation.yaw_error_deg)
                    <= self.config.yaw_tolerance_deg
                )
                trace.append(
                    {
                        "step": step_index,
                        "phase": "confirm",
                        "observation": confirmation.to_dict(),
                        "success": confirmed,
                        "message": "docking confirmed" if confirmed else "confirmation out of tolerance",
                    }
                )
                if confirmed:
                    return self._result(
                        True,
                        "table docking converged",
                        observations,
                        commands,
                        trace,
                        final,
                    )
                continue

            error_vector = (
                abs(forward_error),
                abs(lateral_error),
                abs(yaw_error),
            )
            if last_errors is not None:
                improved = (
                    error_vector[0] < last_errors[0] - 0.005
                    or error_vector[1] < last_errors[1] - 0.005
                    or error_vector[2] < last_errors[2] - 0.5
                )
                no_progress_streak = 0 if improved else no_progress_streak + 1
            last_errors = error_vector
            if no_progress_streak >= 3:
                trace.append(
                    {
                        "step": step_index,
                        "phase": "stall",
                        "success": False,
                        "message": "base is not making progress",
                        "observation": observation.to_dict(),
                    }
                )
                return self._result(
                    False,
                    "docking stalled: repeated observation without progress",
                    observations,
                    commands,
                    trace,
                    final,
                )

            command = DockStep(
                forward_m=_clamp(forward_error, self.config.max_step_m),
                right_m=_clamp(lateral_error, self.config.max_step_m),
                yaw_deg=_clamp(yaw_error, self.config.max_step_yaw_deg),
            )

            try:
                outcome = self.actuator.step(command)
            except Exception as exc:  # noqa: BLE001 - actuation must fail closed
                return self._result(
                    False,
                    f"base step failed: {type(exc).__name__}: {exc}",
                    observations,
                    commands,
                    trace,
                    final,
                )
            commands.append(command.to_dict())
            trace.append(
                {
                    "step": step_index,
                    "phase": "step",
                    "command": command.to_dict(),
                    "success": bool(outcome.success),
                    "message": outcome.message,
                }
            )
            if not outcome.success:
                return self._result(
                    False,
                    "base step was refused: " + (outcome.message or "unknown reason"),
                    observations,
                    commands,
                    trace,
                    final,
                )
        return self._result(
            False,
            f"docking did not converge within {self.config.max_steps} steps",
            observations,
            commands,
            trace,
            final,
        )

    @staticmethod
    def _result(
        success: bool,
        message: str,
        observations: list[dict[str, Any]],
        commands: list[dict[str, Any]],
        trace: list[dict[str, Any]],
        final: TableObservation | None,
    ) -> DockingResult:
        return DockingResult(
            success=success,
            message=message,
            data={
                "observations": observations,
                "commands": commands,
                "trace": trace,
                "final_observation": final.to_dict() if final is not None else None,
            },
        )


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class VlmTableObserver:
    """Estimates table-edge geometry from the three-camera RGB rig."""

    PROMPT = (
        "你是移动机器人底盘对接的几何测量模块。根据头部相机（head_cam）画面以及两路腕部相机，"
        "只输出一个 JSON 对象，不要输出解释。\n"
        "字段：\n"
        '  "distance_m": 机器人与最近桌沿的垂直距离（米，正数）\n'
        '  "lateral_offset_m": 桌沿中点相对机器人前进轴线的横向偏移（米，右侧为正）\n'
        '  "yaw_error_deg": 为垂直对准桌沿，机器人需要逆时针旋转的角度（度，逆时针为正）\n'
        '  "confidence": 0 到 1 的置信度\n'
        '  "reason": 一句话说明测量依据\n'
        "如果桌沿不可见或无法判断，返回 confidence 为 0 并说明原因。"
    )

    def __init__(self, vlm_client: Any | None = None):
        if vlm_client is None:
            from .hardware.vlm import VLMClient

            vlm_client = VLMClient()
        self.vlm = vlm_client

    def observe(self) -> TableObservation:
        evidence = self.vlm.observe_evidence(self.PROMPT)
        if not evidence.success:
            return TableObservation(
                None,
                None,
                None,
                0.0,
                reason=evidence.error or "camera rig observation failed",
            )
        payload = _extract_json_object(evidence.response)
        if payload is None:
            return TableObservation(
                None,
                None,
                None,
                0.0,
                reason="VLM response was not a JSON object",
                snapshot_sha256=evidence.snapshot_sha256,
                captured_at=evidence.captured_at,
            )
        distance = _number(payload.get("distance_m"))
        lateral = _number(payload.get("lateral_offset_m"))
        yaw = _number(payload.get("yaw_error_deg"))
        confidence = _number(payload.get("confidence"))
        reason = payload.get("reason")
        return TableObservation(
            distance,
            lateral,
            yaw,
            confidence if confidence is not None else 0.0,
            reason=reason if isinstance(reason, str) else "",
            snapshot_sha256=evidence.snapshot_sha256,
            captured_at=evidence.captured_at,
        )


class ChassisDockActuator:
    """Executes bounded holonomic steps through the authorized chassis client."""

    def __init__(self, chassis_client: Any | None = None):
        if chassis_client is None:
            from .hardware.chassis import ChassisClient

            chassis_client = ChassisClient()
        self.chassis = chassis_client

    def step(self, command: DockStep) -> DockStepResult:
        steps: list[tuple[str, float | None, float | None]] = []
        if abs(command.right_m) > _EPSILON_M:
            steps.append(
                (
                    "right" if command.right_m > 0 else "left",
                    abs(command.right_m),
                    None,
                )
            )
        if abs(command.forward_m) > _EPSILON_M:
            steps.append(
                (
                    "forward" if command.forward_m > 0 else "backward",
                    abs(command.forward_m),
                    None,
                )
            )
        if abs(command.yaw_deg) > _EPSILON_DEG:
            steps.append(
                (
                    "rotate_left" if command.yaw_deg > 0 else "rotate_right",
                    None,
                    abs(command.yaw_deg),
                )
            )
        for direction, distance, angle in steps:
            result = self.chassis.move(
                direction,
                distance=distance,
                angle=angle,
                speed="slow",
            )
            if not result.get("success"):
                return DockStepResult(
                    False,
                    str(result.get("error") or result.get("stderr") or "chassis step failed"),
                )
        return DockStepResult(True, "base step executed")


class UnavailableTableObserver:
    def __init__(self, reason: str):
        self.reason = reason

    def observe(self) -> TableObservation:
        return TableObservation(None, None, None, 0.0, reason=self.reason)


class UnavailableDockActuator:
    def __init__(self, reason: str):
        self.reason = reason

    def step(self, command: DockStep) -> DockStepResult:
        return DockStepResult(False, self.reason)


# ---------------------------------------------------------------------------
# Table profile catalog
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TableProfile:
    name: str
    docking: TableDockingConfig

    def public_view(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "standoff_m": self.docking.standoff_m,
            "distance_tolerance_m": self.docking.distance_tolerance_m,
            "lateral_tolerance_m": self.docking.lateral_tolerance_m,
            "yaw_tolerance_deg": self.docking.yaw_tolerance_deg,
            "min_confidence": self.docking.min_confidence,
            "max_step_m": self.docking.max_step_m,
            "max_step_yaw_deg": self.docking.max_step_yaw_deg,
            "max_steps": self.docking.max_steps,
            "deadline_seconds": self.docking.deadline_seconds,
            "settle_seconds": self.docking.settle_seconds,
        }


class TableCatalog:
    def __init__(self, version: int, tables: dict[str, TableProfile]):
        if version != 1:
            raise TableDockingError("table catalog version must be 1")
        self.version = version
        self.tables = dict(tables)

    @classmethod
    def empty(cls) -> TableCatalog:
        return cls(1, {})

    @classmethod
    def load(cls, path: str | Path) -> TableCatalog:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise TableDockingError("table catalog must be a YAML object")
        version = raw.get("version")
        if version != 1:
            raise TableDockingError("table catalog version must be 1")
        tables_raw = raw.get("tables")
        if not isinstance(tables_raw, dict):
            raise TableDockingError("table catalog tables must be an object")
        tables: dict[str, TableProfile] = {}
        for name, value in tables_raw.items():
            if not isinstance(name, str) or not name.strip():
                raise TableDockingError("table profile names must be non-empty strings")
            if not isinstance(value, dict):
                raise TableDockingError(f"table profile {name!r} must be an object")
            tables[name.strip()] = TableProfile(
                name=name.strip(),
                docking=TableDockingConfig.from_mapping(value),
            )
        return cls(version, tables)

    def get(self, name: str) -> TableProfile | None:
        return self.tables.get(name)

    def public_view(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tables": {
                name: profile.public_view()
                for name, profile in sorted(self.tables.items())
            },
        }


# ---------------------------------------------------------------------------
# Task executor
# ---------------------------------------------------------------------------


class TableDockingExecutor:
    """Catalog-bound executor for ``dock_table``."""

    executor_id = TABLE_DOCKING_EXECUTOR

    def __init__(
        self,
        *,
        observer_factory: Callable[[], TableObserver] | None = None,
        actuator_factory: Callable[[], DockActuator] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._observer_factory = observer_factory
        self._actuator_factory = actuator_factory
        self._clock = clock
        self._sleep = sleep

    def validate_config(self, config: Mapping[str, Any]) -> tuple[bool, str | None]:
        try:
            TableDockingConfig.from_mapping(config)
        except TableDockingError as exc:
            return False, str(exc)
        return True, None

    def execute(
        self,
        invocation: SkillInvocation,
        context: SkillExecutionContext,
    ) -> SkillExecutionResult:
        try:
            config = TableDockingConfig.from_mapping(context.config)
        except TableDockingError as exc:
            return SkillExecutionResult(False, str(exc), error=str(exc))

        binding = _dry_run_binding()
        if binding:
            return SkillExecutionResult(
                True,
                "table docking simulated; no hardware command was sent",
                data={
                    "dry_run": True,
                    "execution_mode": "simulated",
                    "config": asdict(config),
                },
            )

        from .hardware.authorization import validate_actuation_authorized

        refusal = validate_actuation_authorized(
            "table_docking.execute",
            invocation.skill_id,
        )
        if refusal is not None:
            return SkillExecutionResult(
                False,
                refusal.message,
                data=refusal.to_result(),
                error=refusal.reason_code,
            )

        observer = (
            self._observer_factory()
            if self._observer_factory is not None
            else VlmTableObserver()
        )
        actuator = (
            self._actuator_factory()
            if self._actuator_factory is not None
            else ChassisDockActuator()
        )
        result = TableDockingController(
            observer=observer,
            actuator=actuator,
            config=config,
            clock=self._clock,
            sleep=self._sleep,
        ).run()
        return SkillExecutionResult(
            result.success,
            result.message,
            data=result.data,
            retryable=not result.success,
        )


def _dry_run_binding() -> dict[str, Any]:
    from .hardware.authorization import _authorized_execution_binding

    binding = _authorized_execution_binding()
    return binding if binding.get("dry_run") is True else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


__all__ = [
    "ChassisDockActuator",
    "DockActuator",
    "DockStep",
    "DockStepResult",
    "DockingResult",
    "TABLE_DOCKING_EXECUTOR",
    "TableCatalog",
    "TableDockingConfig",
    "TableDockingController",
    "TableDockingError",
    "TableDockingExecutor",
    "TableObservation",
    "TableObserver",
    "TableProfile",
    "UnavailableDockActuator",
    "UnavailableTableObserver",
    "VlmTableObserver",
]
