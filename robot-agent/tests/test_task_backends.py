from __future__ import annotations

from dataclasses import replace

from robot_agent.skill_flow import (
    SkillExecutionContext,
    SkillExecutionResult,
    SkillInvocation,
)
from robot_agent.task_backends import TaskBackendRouter
from robot_agent.task_catalog import TaskCatalog


class FakeBackend:
    def __init__(self, result: dict):
        self.result = result
        self.calls: list[str] = []

    def execute_task(self, task_id: str) -> dict:
        self.calls.append(task_id)
        return dict(self.result)


class RecordingExecutor:
    executor_id = "learned_policy"

    def __init__(self):
        self.invocations: list[SkillInvocation] = []

    def execute(self, invocation, context):
        self.invocations.append(invocation)
        return SkillExecutionResult(
            success=True,
            message="recorded",
            data={"context_task": context.task.task_id},
        )


class ConfiguredExecutor:
    executor_id = "rgb_primitive"

    def __init__(self):
        self.configs = []

    def validate_config(self, config):
        if config.get("strategy") != "closed_loop":
            return False, "strategy must be closed_loop"
        return True, None

    def execute(self, invocation, context):
        self.configs.append(dict(context.config))
        return SkillExecutionResult(success=True, message="configured")


def test_legacy_backend_is_exposed_through_skill_contract():
    backend = FakeBackend({"success": True, "stdout": "executed"})
    router = TaskBackendRouter(TaskCatalog.default(), {"learned_policy": backend})

    result = router.execute_skill(
        SkillInvocation("fold_simple", {"operator": "test"}, trace_id="trace-1")
    )

    assert result.success is True
    assert result.data["task_id"] == "fold_simple"
    assert result.data["executor"] == "learned_policy"
    assert backend.calls == ["fold_simple"]


def test_registered_skill_executor_receives_catalog_bound_context():
    executor = RecordingExecutor()
    router = TaskBackendRouter(TaskCatalog.default())
    router.register_executor(executor)

    result = router.execute_skill(SkillInvocation("pick_up_bottle", {"mode": "test"}))

    assert result.success is True
    assert result.data["context_task"] == "pick_up_bottle"
    assert executor.invocations[0].params == {"mode": "test"}


def test_executor_declared_config_is_passed_and_validated():
    base = TaskCatalog.default()
    task = replace(
        base.get("fold_simple"),
        executor="rgb_primitive",
        executor_config={"strategy": "closed_loop"},
    )
    catalog = TaskCatalog(version=1, tasks={**base.tasks, task.task_id: task})
    executor = ConfiguredExecutor()
    router = TaskBackendRouter(catalog)
    router.register_executor(executor)

    result = router.execute_skill(SkillInvocation(task.task_id))

    assert result.success is True
    assert executor.configs == [{"strategy": "closed_loop"}]


def test_executor_rejects_invalid_declared_config():
    base = TaskCatalog.default()
    task = replace(
        base.get("fold_simple"),
        executor="rgb_primitive",
        executor_config={"strategy": "open_loop"},
    )
    catalog = TaskCatalog(version=1, tasks={**base.tasks, task.task_id: task})
    router = TaskBackendRouter(catalog)
    router.register_executor(ConfiguredExecutor())

    result = router.execute_skill(SkillInvocation(task.task_id))

    assert result.success is False
    assert result.error == "invalid_executor_config"
    assert "closed_loop" in result.message


def test_missing_executor_returns_typed_failure():
    result = TaskBackendRouter(TaskCatalog.default()).execute_skill(
        SkillInvocation("fold_simple")
    )

    assert result.success is False
    assert result.error == "executor_not_installed"
    assert result.data == {"task_id": "fold_simple", "executor": "learned_policy"}


def test_context_for_another_task_is_rejected_before_execution():
    catalog = TaskCatalog.default()
    executor = RecordingExecutor()
    router = TaskBackendRouter(catalog)
    router.register_executor(executor)

    result = router.execute_skill(
        SkillInvocation("fold_simple"),
        context=SkillExecutionContext(task=catalog.get("stack_contrast")),
    )

    assert result.success is False
    assert result.error == "context_task_mismatch"
    assert executor.invocations == []


def test_invocation_copies_parameters_and_rejects_empty_skill_id():
    params = {"attempt": 1}
    invocation = SkillInvocation(" fold_simple ", params)
    params["attempt"] = 2

    assert invocation.skill_id == "fold_simple"
    assert invocation.params == {"attempt": 1}

    try:
        SkillInvocation(" ")
    except ValueError as exc:
        assert "skill_id" in str(exc)
    else:
        raise AssertionError("empty skill IDs must be rejected")
