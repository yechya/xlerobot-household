from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionStep:
    name: str
    action: str
    success: bool = False
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskExecutionResult:
    success: bool
    task_id: str
    total_steps: int
    completed_steps: int
    steps: list[ExecutionStep] = field(default_factory=list)
    final_message: str = ""
    attempts: int = 1
    adjustments_made: list[str] = field(default_factory=list)


__all__ = ["ExecutionStep", "TaskExecutionResult"]
