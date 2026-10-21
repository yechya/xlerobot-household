from __future__ import annotations

from pathlib import Path

import pytest

from robot_agent.episodes import EpisodeRecorder
from robot_agent.execution import WorkerResult
from robot_agent.hardware.vlm import VLMObservation
from robot_agent.robotd import RobotDaemon
from robot_agent.task_catalog import TaskCatalog
from robot_agent.visual_task_planner import VisualPlanError, VisualTaskPlanner


class FakeHeadPlanner:
    def __init__(self, response: str):
        self.response = response
        self.instructions: list[str] = []

    def generate_head_camera(self, instruction: str, *, timeout: int | None = None):
        self.instructions.append(instruction)
        return VLMObservation(
            response=self.response,
            snapshot_sha256="c" * 64,
            captured_at=1000.0,
            model="test-vlm",
            camera_sha256=(("head_cam", "c" * 64),),
        )


def test_visual_planner_uses_head_camera_and_binds_task_fingerprint():
    catalog = TaskCatalog.default()
    llm = FakeHeadPlanner(
        '{"steps":[{"task_id":"pick_up_bottle","confidence":0.91,"reason":"看到瓶子和筐"}]}'
    )
    plan = VisualTaskPlanner(catalog, llm).create("整理桌面", world_context={"room": "kitchen"})

    assert plan.task_ids == ("pick_up_bottle",)
    assert plan.task_fingerprints == (
        ("pick_up_bottle", catalog.tasks["pick_up_bottle"].fingerprint()),
    )
    assert plan.plan_sha256 and len(plan.plan_sha256) == 64
    assert "head_cam" in llm.instructions[0]
    plan.validate()


def test_visual_planner_accepts_complete_json_fence():
    response = """```json
{"steps":[]}
```"""
    plan = VisualTaskPlanner(TaskCatalog.default(), FakeHeadPlanner(response)).create("整理桌面")
    assert plan.steps == ()


@pytest.mark.parametrize(
    "response, message",
    [
        (
            '{"steps":[{"task_id":"unknown","confidence":0.9,"reason":"x"}]}',
            "not registered",
        ),
        (
            '{"steps":[{"task_id":"pick_up_bottle","confidence":0.9,"reason":"x"},'
            '{"task_id":"pick_up_bottle","confidence":0.9,"reason":"x"}]}',
            "duplicated",
        ),
        (
            '{"steps":[{"task_id":"pick_up_bottle","confidence":0.4,"reason":"x"}]}',
            "confidence",
        ),
        (
            '{"steps":[{"task_id":"pick_up_bottle","confidence":0.9,"reason":"x"}],'
            '"extra":1}',
            "only steps",
        ),
    ],
)
def test_visual_planner_rejects_untrusted_model_output(response: str, message: str):
    planner = VisualTaskPlanner(TaskCatalog.default(), FakeHeadPlanner(response))
    with pytest.raises(VisualPlanError, match=message):
        planner.create("整理桌面")


class ImmediateExecutor:
    active = False

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def execute(
        self,
        task_id: str,
        args: dict,
        token: str,
        timeout_seconds: int,
    ) -> WorkerResult:
        self.calls.append((task_id, dict(args)))
        return WorkerResult(True, "executed", {"task_id": task_id})

    def cancel(self, *, reason: str, emergency: bool = False) -> None:
        return None


class VisualAgent:
    def __init__(self, catalog: TaskCatalog, vlm: FakeHeadPlanner):
        self.task_catalog = catalog
        self.vlm = vlm
        self.world_model = type("World", (), {"summary": lambda _self: {"objects": []}})()

    def execute_dry_run_pipeline(self, task_id: str):
        return type(
            "Result",
            (),
            {
                "success": True,
                "message": f"dummy pipeline {task_id}",
                "data": {"execution_mode": "pipeline_dummy", "task_id": task_id},
            },
        )()


def test_robotd_visual_plan_arm_and_execute_runs_registered_tasks(tmp_path: Path):
    catalog = TaskCatalog.default()
    vlm = FakeHeadPlanner(
        '{"steps":[{"task_id":"pick_up_bottle","confidence":0.91,"reason":"看到瓶子和筐"}]}'
    )
    executor = ImmediateExecutor()

    def agent_factory(*, dry_run: bool):
        return VisualAgent(catalog, vlm)

    def binding(task_id: str) -> dict:
        return {
            "task_fingerprint": catalog.tasks[task_id].fingerprint(),
            "task_executor": "learned_policy",
            "policy_fingerprint": "b" * 64,
        }

    daemon = RobotDaemon(
        agent_factory=agent_factory,
        token_store_path=tmp_path / "tokens.json",
        real_executor=executor,
        task_binding_resolver=binding,
        episode_recorder=EpisodeRecorder(tmp_path / "episodes"),
        require_fresh_observation=False,
        require_safety_interlock=False,
        task_catalog=catalog,
        session_id="visual-plan-session",
        clock=lambda: 1000.0,
    )
    inspected = daemon.inspect_visual_plan("整理桌面")
    assert inspected.success is True
    plan = inspected.data["visual_plan"]
    plan_id = plan["plan_id"]

    armed = daemon.request_visual_plan_arm(
        plan_id,
        operator_confirmation=f"CONFIRM plan:{plan_id}",
    )
    assert armed.success is True
    assert armed.data["task_id"] == f"plan:{plan_id}"
    result = daemon.execute_visual_plan(plan_id, token=armed.data["token_secret"])

    assert result.success is True
    assert result.data["task_ids"] == ["pick_up_bottle"]
    assert executor.calls[0][0] == "pick_up_bottle"
    assert daemon.status()["active_task"] is None


def test_empty_visual_plan_cannot_be_armed(tmp_path: Path):
    catalog = TaskCatalog.default()
    daemon = RobotDaemon(
        agent_factory=lambda *, dry_run: VisualAgent(catalog, FakeHeadPlanner('{"steps":[]}')),
        token_store_path=tmp_path / "tokens.json",
        require_safety_interlock=False,
        task_catalog=catalog,
        session_id="visual-plan-empty",
    )

    inspected = daemon.inspect_visual_plan("整理桌面")
    plan_id = inspected.data["visual_plan"]["plan_id"]
    armed = daemon.request_visual_plan_arm(
        plan_id,
        operator_confirmation=f"CONFIRM plan:{plan_id}",
    )

    assert inspected.success is True
    assert inspected.data["requires_confirmation"] is False
    assert armed.success is False
    assert armed.refusal is not None
    assert armed.refusal.reason_code == "task_not_allowlisted"


def test_plan_routes_abstract_goal_through_visual_planner(tmp_path: Path):
    catalog = TaskCatalog.default()
    vlm = FakeHeadPlanner(
        '{"steps":[{"task_id":"pick_up_bottle","confidence":0.91,"reason":"看到瓶子和筐"}]}'
    )
    daemon = RobotDaemon(
        agent_factory=lambda *, dry_run: VisualAgent(catalog, vlm),
        token_store_path=tmp_path / "tokens.json",
        require_safety_interlock=False,
        task_catalog=catalog,
        session_id="visual-plan-auto-route",
    )

    result = daemon.plan("整理桌面")

    assert result.success is True
    assert result.data["visual_plan"]["task_ids"] == ["pick_up_bottle"]
    assert "head_cam" in vlm.instructions[0]


def test_dry_run_visual_plan_executes_each_step_through_dummy_pipeline(tmp_path: Path):
    catalog = TaskCatalog.default()
    vlm = FakeHeadPlanner(
        '{"steps":[{"task_id":"pick_up_bottle","confidence":0.91,"reason":"看到瓶子和筐"}]}'
    )
    agent = VisualAgent(catalog, vlm)
    daemon = RobotDaemon(
        dry_run_agent_factory=lambda: agent,
        token_store_path=tmp_path / "tokens.json",
        require_fresh_observation=False,
        require_safety_interlock=False,
        task_catalog=catalog,
    )

    result = daemon.dry_run("整理桌面")

    assert result.success is True
    assert result.data["execution_mode"] == "pipeline_dummy"
    assert result.data["task_ids"] == ["pick_up_bottle"]
    assert result.data["steps"][0]["data"]["agent_data"]["execution_mode"] == "pipeline_dummy"


def test_visual_plan_token_is_invalidated_when_task_binding_changes(tmp_path: Path):
    catalog = TaskCatalog.default()
    vlm = FakeHeadPlanner(
        '{"steps":[{"task_id":"pick_up_bottle","confidence":0.91,"reason":"看到瓶子和筐"}]}'
    )
    changed = {"value": False}

    def binding(task_id: str) -> dict:
        return {
            "task_fingerprint": catalog.tasks[task_id].fingerprint(),
            "task_executor": "learned_policy",
            "policy_fingerprint": ("c" if changed["value"] else "b") * 64,
        }

    daemon = RobotDaemon(
        agent_factory=lambda *, dry_run: VisualAgent(catalog, vlm),
        token_store_path=tmp_path / "tokens.json",
        real_executor=ImmediateExecutor(),
        task_binding_resolver=binding,
        require_safety_interlock=False,
        task_catalog=catalog,
        session_id="visual-plan-invalidated",
    )
    inspected = daemon.inspect_visual_plan("整理桌面")
    plan_id = inspected.data["visual_plan"]["plan_id"]
    armed = daemon.request_visual_plan_arm(
        plan_id,
        operator_confirmation=f"CONFIRM plan:{plan_id}",
    )
    assert armed.success is True
    changed["value"] = True
    result = daemon.execute_visual_plan(plan_id, token=armed.data["token_secret"])
    assert result.success is False
    assert result.refusal is not None
    assert result.refusal.reason_code == "invalid_token"
