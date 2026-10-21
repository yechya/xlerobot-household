"""Public API for the standalone robot agent runtime."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "RobotAgent": ("agent", "RobotAgent"),
    "ExecutionResult": ("agent", "ExecutionResult"),
    "TaskExecutionResult": ("execution_types", "TaskExecutionResult"),
    "ExecutionStep": ("execution_types", "ExecutionStep"),
    "GoalPlanner": ("goal_planner", "GoalPlanner"),
    "TaskPlan": ("goal_planner", "TaskPlan"),
    "PlanningError": ("goal_planner", "PlanningError"),
    "DialogueHistory": ("dialogue_history", "DialogueHistory"),
    "DialogueEntry": ("dialogue_history", "DialogueEntry"),
    "ErrorDiagnoser": ("error_diagnoser", "ErrorDiagnoser"),
    "DiagnosisReport": ("error_diagnoser", "DiagnosisReport"),
    "EnvironmentScanner": ("environment_scanner", "EnvironmentScanner"),
    "ScanResult": ("environment_scanner", "ScanResult"),
    "DetectedObject": ("environment_scanner", "DetectedObject"),
    "EvoRLExport": ("evo_rl", "EvoRLExport"),
    "EvoRLEpisodeRecord": ("evo_rl", "EvoRLEpisodeRecord"),
    "EvoRLBridgeError": ("evo_rl", "EvoRLBridgeError"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
