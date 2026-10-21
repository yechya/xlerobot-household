from __future__ import annotations

import yaml

from robot_agent.task_catalog import TaskCatalog, TaskCatalogError


def _task(executor: str, executor_config: dict) -> dict:
    return {
        "name": "Configured task",
        "description": "run a configured task",
        "executor": executor,
        "executor_config": executor_config,
        "aliases": ["configured task"],
        "safety": {
            "args": {},
            "requires_observation": True,
            "requires_token": True,
            "max_runtime_seconds": 30,
            "allowed_modes": ["DRY_RUN", "ARMED_REAL"],
            "stop_conditions": ["task_completed"],
        },
        "manipulation": {
            "require_new_snapshot": True,
            "max_policy_attempts": 1,
            "preconditions": [
                {"id": "ready", "question": "Is the task ready?", "min_confidence": 0.8}
            ],
            "postconditions": [
                {"id": "done", "question": "Is the task done?", "min_confidence": 0.8}
            ],
        },
    }


def test_catalog_preserves_future_executor_and_json_config(tmp_path):
    path = tmp_path / "tasks.yaml"
    raw = {"version": 1, "tasks": {"configured": _task("rgb_primitive", {"mode": "closed_loop"})}}
    path.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    task = TaskCatalog.load(path).get("configured")

    assert task.executor == "rgb_primitive"
    assert task.executor_config == {"mode": "closed_loop"}


def test_catalog_rejects_non_json_executor_config(tmp_path):
    path = tmp_path / "tasks.yaml"
    path.write_text(
        """version: 1
tasks:
  configured:
    name: Configured task
    description: run a configured task
    executor: rgb_primitive
    executor_config:
      mode: 2026-07-16
    aliases: [configured task]
    safety:
      args: {}
      requires_observation: true
      requires_token: true
      max_runtime_seconds: 30
      allowed_modes: [DRY_RUN, ARMED_REAL]
      stop_conditions: [task_completed]
    manipulation:
      require_new_snapshot: true
      max_policy_attempts: 1
      preconditions:
        - {id: ready, question: Is the task ready?, min_confidence: 0.8}
      postconditions:
        - {id: done, question: Is the task done?, min_confidence: 0.8}
""",
        encoding="utf-8",
    )

    try:
        TaskCatalog.load(path)
    except (TaskCatalogError, yaml.YAMLError):
        pass
    else:
        raise AssertionError("non-JSON executor_config must be rejected")
