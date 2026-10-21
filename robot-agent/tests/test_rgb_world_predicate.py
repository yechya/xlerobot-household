from __future__ import annotations

from dataclasses import replace

from robot_agent.rgb_world_predicate import RgbWorldPredicateExecutor
from robot_agent.skill_flow import SkillExecutionContext, SkillInvocation
from robot_agent.task_backends import TaskBackendRouter
from robot_agent.task_catalog import TaskCatalog
from robot_agent.world_model import ObjectState, WorldModel


def _catalog(predicate: str) -> tuple[TaskCatalog, object]:
    base = TaskCatalog.default()
    task = replace(
        base.get("fold_simple"),
        executor="rgb_world_predicate",
        executor_config={"predicate": predicate},
    )
    return TaskCatalog(version=1, tasks={**base.tasks, task.task_id: task}), task


def test_rgb_world_predicate_reads_structured_object_visibility():
    catalog, task = _catalog("object_visible(bottle)")
    world_model = WorldModel()
    world_model.add_object(
        ObjectState(
            object_id="bottle_1",
            label="bottle",
            confidence=0.9,
            source="rgb_segmentation",
        )
    )
    router = TaskBackendRouter(catalog)
    router.register_executor(RgbWorldPredicateExecutor())

    result = router.execute_skill(
        SkillInvocation(task.task_id),
        context=SkillExecutionContext(
            task=task,
            world_model=world_model,
            config=task.executor_config,
        ),
    )

    assert result.success is True
    assert result.data["state"] == "pass"
    assert result.data["source"] == "rgb_world_model"


def test_rgb_world_predicate_is_conservative_when_object_is_missing():
    catalog, task = _catalog("object_visible(bottle)")
    router = TaskBackendRouter(catalog)
    router.register_executor(RgbWorldPredicateExecutor())

    result = router.execute_skill(
        SkillInvocation(task.task_id),
        context=SkillExecutionContext(
            task=task,
            world_model=WorldModel(),
            config=task.executor_config,
        ),
    )

    assert result.success is False
    assert result.error == "predicate_not_proven"
    assert result.data["state"] == "unknown"


def test_rgb_world_predicate_rejects_unreviewed_predicates():
    executor = RgbWorldPredicateExecutor()

    valid, message = executor.validate_config({"predicate": "object_xyz(bottle)"})

    assert valid is False
    assert "unsupported" in message

    valid, message = executor.validate_config({"predicate": "object_reachable(bottle)"})

    assert valid is False
    assert "unsupported" in message
