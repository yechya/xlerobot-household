from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_agent.missions import (
    MissionActionResult,
    MissionCatalog,
    MissionError,
    MissionExecutor,
)


def _graph() -> dict:
    return {
        "id": "root",
        "type": "sequence",
        "children": [
            {
                "id": "navigate_retry",
                "type": "retry",
                "max_attempts": 2,
                "child": {
                    "id": "go_table",
                    "type": "action",
                    "action": "navigate",
                    "target": "kitchen_table",
                    "timeout_seconds": 120,
                },
            },
            {
                "id": "dock_table",
                "type": "action",
                "action": "dock",
                "target": "kitchen_table_dock",
                "timeout_seconds": 60,
            },
            {
                "id": "pick_remote",
                "type": "action",
                "action": "task",
                "target": "pick_up_remote",
                "timeout_seconds": 180,
            },
        ],
    }


def _catalog(tmp_path: Path, graph: dict | None = None) -> MissionCatalog:
    path = tmp_path / "missions.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "missions": {
                    "fetch_remote": {
                        "name": "Fetch remote",
                        "description": "Navigate to the table and pick up the remote",
                        "aliases": ["拿遥控器", "fetch remote"],
                        "requires_observation": True,
                        "max_runtime_seconds": 600,
                        "max_node_visits": 32,
                        "graph": graph or _graph(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return MissionCatalog.load(path)


class FakeProvider:
    def __init__(self, outcomes: list[bool]):
        self.outcomes = list(outcomes)
        self.calls = []

    def execute(self, action, deadline):
        self.calls.append((action.node_id, action.action, action.target, deadline))
        success = self.outcomes.pop(0)
        return MissionActionResult(success, "ok" if success else "failed")


def test_catalog_fingerprints_bounded_named_actions(tmp_path: Path):
    catalog = _catalog(tmp_path)
    mission = catalog.get("fetch_remote")
    assert [action.action for action in mission.actions] == ["navigate", "dock", "task"]
    assert len(mission.fingerprint()) == 64
    public = catalog.public_view()["missions"]["fetch_remote"]
    assert public["actions"][0]["target"] == "kitchen_table"
    assert catalog.match("请帮我拿遥控器") == mission
    assert catalog.match("unregistered request") is None


def test_executor_retries_motion_then_runs_task_once(tmp_path: Path):
    mission = _catalog(tmp_path).get("fetch_remote")
    provider = FakeProvider([False, True, True, True])
    result = MissionExecutor(provider=provider).execute(mission)
    assert result.success is True
    assert [call[1] for call in provider.calls] == ["navigate", "navigate", "dock", "task"]
    assert sum(call[1] == "task" for call in provider.calls) == 1
    assert result.node_visits == 6
    assert result.trace[-1]["node_id"] == "root"


def test_selector_runs_fallback_and_cancellation_stops_next_action(tmp_path: Path):
    graph = {
        "id": "root",
        "type": "sequence",
        "children": [
            {
                "id": "route",
                "type": "selector",
                "children": [
                    {
                        "id": "primary",
                        "type": "action",
                        "action": "navigate",
                        "target": "kitchen_table",
                        "timeout_seconds": 30,
                    },
                    {
                        "id": "fallback",
                        "type": "action",
                        "action": "navigate",
                        "target": "alternate_route",
                        "timeout_seconds": 30,
                    },
                ],
            },
            {
                "id": "dock",
                "type": "action",
                "action": "dock",
                "target": "kitchen_table_dock",
                "timeout_seconds": 30,
            },
        ],
    }
    provider = FakeProvider([False, True])
    canceled = lambda: len(provider.calls) >= 2
    result = MissionExecutor(provider=provider, cancelled=canceled).execute(
        _catalog(tmp_path, graph).get("fetch_remote")
    )
    assert result.success is False
    assert [call[2] for call in provider.calls] == ["kitchen_table", "alternate_route"]
    assert "canceled" in result.message


def test_catalog_rejects_task_retry_and_duplicate_nodes(tmp_path: Path):
    task_retry = {
        "id": "retry",
        "type": "retry",
        "max_attempts": 2,
        "child": {
            "id": "task",
            "type": "action",
            "action": "task",
            "target": "pick_up_remote",
            "timeout_seconds": 30,
        },
    }
    with pytest.raises(MissionError, match="cannot be descendants"):
        _catalog(tmp_path, task_retry)
    duplicate = {
        "id": "root",
        "type": "sequence",
        "children": [
            {
                "id": "same",
                "type": "action",
                "action": "navigate",
                "target": "one",
                "timeout_seconds": 10,
            },
            {
                "id": "same",
                "type": "action",
                "action": "dock",
                "target": "two",
                "timeout_seconds": 10,
            },
        ],
    }
    with pytest.raises(MissionError, match="duplicated"):
        _catalog(tmp_path, duplicate)


def test_action_deadline_overrun_fails_mission(tmp_path: Path):
    now = [10.0]

    class SlowProvider:
        def execute(self, action, deadline):
            now[0] = deadline + 0.1
            return MissionActionResult(True, "late")

    result = MissionExecutor(provider=SlowProvider(), clock=lambda: now[0]).execute(
        _catalog(tmp_path).get("fetch_remote")
    )
    assert result.success is False
    assert "deadline" in result.message


def test_catalog_rejects_alias_shared_by_two_missions(tmp_path: Path):
    path = tmp_path / "duplicate-alias.yaml"
    mission = {
        "name": "Mission",
        "description": "One bounded motion",
        "aliases": ["same alias"],
        "requires_observation": False,
        "max_runtime_seconds": 60,
        "max_node_visits": 4,
        "graph": {
            "id": "go",
            "type": "action",
            "action": "navigate",
            "target": "kitchen",
            "timeout_seconds": 30,
        },
    }
    path.write_text(
        yaml.safe_dump(
            {"version": 1, "missions": {"first": mission, "second": mission}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(MissionError, match="shared"):
        MissionCatalog.load(path)


def _recovery_graph(recovery: dict | None = None) -> dict:
    retry: dict = {
        "id": "dock_retry",
        "type": "retry",
        "max_attempts": 2,
        "child": {
            "id": "approach",
            "type": "action",
            "action": "dock_table",
            "target": "cabinet_edge",
            "timeout_seconds": 30,
        },
    }
    if recovery is not None:
        retry["recovery"] = recovery
    return {
        "id": "root",
        "type": "sequence",
        "children": [
            retry,
            {
                "id": "place",
                "type": "action",
                "action": "task",
                "target": "operate_drawer",
                "timeout_seconds": 60,
            },
        ],
    }


def test_dock_table_action_and_observe_recovery_are_bound(tmp_path: Path):
    catalog = _catalog(
        tmp_path,
        _recovery_graph(
            {
                "action": "observe",
                "target": "workspace_check",
                "timeout_seconds": 30,
            }
        ),
    )
    mission = catalog.get("fetch_remote")
    assert [action.action for action in mission.actions] == [
        "dock_table",
        "observe",
        "task",
    ]
    assert [entry.retry_node_id for entry in mission.recoveries] == ["dock_retry"]
    assert mission.recoveries[0].action.node_id == "dock_retry.recovery"
    public = catalog.public_view()["missions"]["fetch_remote"]["actions"]
    assert public[1]["action"] == "observe"


def test_task_retry_requires_declared_recovery(tmp_path: Path):
    graph = _recovery_graph()
    graph["children"][0] = {
        "id": "task_retry",
        "type": "retry",
        "max_attempts": 2,
        "child": {
            "id": "place",
            "type": "action",
            "action": "task",
            "target": "operate_drawer",
            "timeout_seconds": 30,
        },
    }
    with pytest.raises(MissionError, match="without a recovery action"):
        _catalog(tmp_path, graph)


def test_task_retry_with_recovery_is_allowed(tmp_path: Path):
    graph = _recovery_graph(
        {
            "action": "observe",
            "target": "workspace_check",
            "timeout_seconds": 30,
        }
    )
    graph["children"] = [graph["children"][0]]
    graph["children"][0]["child"] = {
        "id": "operate",
        "type": "action",
        "action": "task",
        "target": "operate_drawer",
        "timeout_seconds": 60,
    }
    mission = _catalog(tmp_path, graph).get("fetch_remote")
    assert [action.action for action in mission.actions] == ["task", "observe"]


def test_recovery_runs_between_retries_and_then_continues(tmp_path: Path):
    graph = _recovery_graph(
        {
            "action": "observe",
            "target": "workspace_check",
            "timeout_seconds": 30,
        }
    )
    provider = FakeProvider([False, True, True, True])
    result = MissionExecutor(provider=provider).execute(
        _catalog(tmp_path, graph).get("fetch_remote")
    )
    assert result.success is True
    assert [(call[1], call[2]) for call in provider.calls] == [
        ("dock_table", "cabinet_edge"),
        ("observe", "workspace_check"),
        ("dock_table", "cabinet_edge"),
        ("task", "operate_drawer"),
    ]
    assert [entry["type"] for entry in result.trace] == [
        "action",
        "recovery",
        "action",
        "retry",
        "action",
        "sequence",
    ]
    assert result.node_visits == 5


def test_failed_recovery_stops_the_retry_and_the_sequence(tmp_path: Path):
    graph = _recovery_graph(
        {
            "action": "observe",
            "target": "workspace_check",
            "timeout_seconds": 30,
        }
    )
    provider = FakeProvider([False, False])
    result = MissionExecutor(provider=provider).execute(
        _catalog(tmp_path, graph).get("fetch_remote")
    )
    assert result.success is False
    assert "recovery action failed" in result.message
    assert [(call[1], call[2]) for call in provider.calls] == [
        ("dock_table", "cabinet_edge"),
        ("observe", "workspace_check"),
    ]


def test_recovery_rejects_unsupported_action(tmp_path: Path):
    graph = _recovery_graph(
        {"action": "navigate", "target": "somewhere", "timeout_seconds": 30}
    )
    with pytest.raises(MissionError, match="unsupported recovery action"):
        _catalog(tmp_path, graph)
