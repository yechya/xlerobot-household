from __future__ import annotations

from typing import Any

from .manipulation import TaskBackend
from .skill_flow import (
    SkillExecutionContext,
    SkillExecutionResult,
    SkillExecutor,
    SkillInvocation,
)
from .task_catalog import TaskCatalog


class _LegacyTaskBackendExecutor:
    def __init__(self, executor_id: str, backend: TaskBackend):
        self.executor_id = executor_id
        self.backend = backend

    def validate_config(self, config: dict[str, Any]) -> tuple[bool, str | None]:
        if config:
            return False, f"executor {self.executor_id!r} does not accept executor_config"
        return True, None

    def execute(
        self,
        invocation: SkillInvocation,
        context: SkillExecutionContext,
    ) -> SkillExecutionResult:
        del context
        raw = dict(self.backend.execute_task(invocation.skill_id))
        error = str(raw.get("error", "") or "")
        message = str(raw.get("stdout") or raw.get("message") or error or "")
        return SkillExecutionResult(
            success=bool(raw.get("success", False)),
            message=message,
            data=raw,
            error=error,
            retryable=bool(raw.get("retryable", False)),
        )


class TaskBackendRouter:
    def __init__(
        self,
        task_catalog: TaskCatalog,
        backends: dict[str, TaskBackend] | None = None,
    ):
        self.task_catalog = task_catalog
        self.backends = dict(backends or {})
        self.executors: dict[str, SkillExecutor] = {}
        for executor_id, backend in self.backends.items():
            self.executors[executor_id] = _LegacyTaskBackendExecutor(executor_id, backend)

    def register(self, executor: str, backend: TaskBackend) -> None:
        if not executor or executor in self.executors:
            raise ValueError(f"task executor {executor!r} is empty or already registered")
        self.backends[executor] = backend
        self.executors[executor] = _LegacyTaskBackendExecutor(executor, backend)

    def register_executor(self, executor: SkillExecutor) -> None:
        executor_id = str(getattr(executor, "executor_id", "")).strip()
        if not executor_id or executor_id in self.executors:
            raise ValueError(
                f"skill executor {executor_id!r} is empty or already registered"
            )
        self.executors[executor_id] = executor

    def execute_skill(
        self,
        invocation: SkillInvocation,
        *,
        context: SkillExecutionContext | None = None,
    ) -> SkillExecutionResult:
        task = self.task_catalog.get(invocation.skill_id)
        if task is None:
            return SkillExecutionResult(
                success=False,
                message=f"unknown task: {invocation.skill_id}",
                error="unknown_task",
                data={"task_id": invocation.skill_id},
            )

        executor = self.executors.get(task.executor)
        if executor is None:
            return SkillExecutionResult(
                success=False,
                message=f"task executor is not installed: {task.executor}",
                error="executor_not_installed",
                data={"task_id": invocation.skill_id, "executor": task.executor},
            )

        config_error = self._validate_executor_config(executor, task.executor_config)
        if config_error:
            return SkillExecutionResult(
                success=False,
                message=config_error,
                error="invalid_executor_config",
                data={
                    "task_id": invocation.skill_id,
                    "executor": task.executor,
                },
            )

        execution_context = context or SkillExecutionContext(
            task=task,
            config=task.executor_config,
        )
        if execution_context.task.task_id != task.task_id:
            return SkillExecutionResult(
                success=False,
                message="skill execution context is bound to a different task",
                error="context_task_mismatch",
                data={"task_id": invocation.skill_id, "executor": task.executor},
            )
        if dict(execution_context.config) != task.executor_config:
            return SkillExecutionResult(
                success=False,
                message="skill execution context has a different executor_config",
                error="context_config_mismatch",
                data={"task_id": invocation.skill_id, "executor": task.executor},
            )

        try:
            result = executor.execute(invocation, execution_context)
        except Exception as exc:
            return SkillExecutionResult(
                success=False,
                message=f"skill executor failed: {type(exc).__name__}: {exc}",
                error="executor_exception",
                data={"task_id": invocation.skill_id, "executor": task.executor},
            )
        if not isinstance(result, SkillExecutionResult):
            return SkillExecutionResult(
                success=False,
                message="skill executor returned an invalid result",
                error="invalid_executor_result",
                data={"task_id": invocation.skill_id, "executor": task.executor},
            )

        data = dict(result.data)
        data.setdefault("task_id", invocation.skill_id)
        data.setdefault("executor", task.executor)
        return SkillExecutionResult(
            success=result.success,
            message=result.message,
            data=data,
            error=result.error,
            retryable=result.retryable,
        )

    @staticmethod
    def _validate_executor_config(
        executor: SkillExecutor,
        config: dict[str, Any],
    ) -> str:
        validator = getattr(executor, "validate_config", None)
        if validator is None:
            return "executor does not declare executor_config validation" if config else ""
        try:
            valid, message = validator(dict(config))
        except Exception as exc:
            return f"executor_config validation failed: {type(exc).__name__}: {exc}"
        if not isinstance(valid, bool) or (message is not None and not isinstance(message, str)):
            return "executor returned an invalid executor_config validation result"
        return "" if valid else (message or "executor rejected executor_config")

    def execute_task(
        self,
        task_id: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = self.execute_skill(SkillInvocation(task_id, params or {}))
        return result.to_dict()

    def public_view(self) -> dict[str, Any]:
        return {
            "installed_executors": sorted(self.executors),
            "tasks": {
                task_id: {
                    "executor": task.executor,
                    "available": task.executor in self.executors,
                }
                for task_id, task in sorted(self.task_catalog.tasks.items())
            },
        }


__all__ = ["TaskBackendRouter"]
