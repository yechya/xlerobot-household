from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_agent.table_docking import (
    DockStep,
    DockStepResult,
    TableCatalog,
    TableDockingConfig,
    TableDockingController,
    TableDockingError,
    TableDockingExecutor,
    TableObservation,
    VlmTableObserver,
)
from robot_agent.hardware.authorization import (
    _make_robotd_hardware_authorization,
    _robotd_authorized_context,
)
from robot_agent.skill_flow import SkillExecutionContext, SkillInvocation
from robot_agent.task_catalog import TaskCatalog


class ScriptedObserver:
    def __init__(self, observations):
        self.observations = list(observations)
        self.calls = 0

    def observe(self):
        self.calls += 1
        item = self.observations.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class RecordingActuator:
    def __init__(self, outcomes=None):
        self.commands: list[DockStep] = []
        self.outcomes = list(outcomes or [])

    def step(self, command):
        self.commands.append(command)
        outcome = self.outcomes.pop(0) if self.outcomes else DockStepResult(True)
        return outcome


def _observation(distance=0.45, lateral=0.0, yaw=0.0, confidence=0.9):
    return TableObservation(distance, lateral, yaw, confidence)


def test_controller_converges_with_bounded_steps():
    observer = ScriptedObserver(
        [
            _observation(distance=0.80, lateral=0.10, yaw=5.0),
            _observation(distance=0.55, lateral=0.06, yaw=2.0),
            _observation(distance=0.44, lateral=0.01, yaw=0.5),
            _observation(distance=0.44, lateral=0.00, yaw=0.2),
        ]
    )
    actuator = RecordingActuator()
    result = TableDockingController(
        observer=observer,
        actuator=actuator,
        config=TableDockingConfig.from_mapping({}),
        sleep=lambda _seconds: None,
    ).run()
    assert result.success is True
    assert len(actuator.commands) == 2
    assert actuator.commands[0].to_dict() == {
        "forward_m": pytest.approx(0.12),
        "right_m": pytest.approx(0.10),
        "yaw_deg": pytest.approx(5.0),
    }
    assert actuator.commands[1].to_dict() == {
        "forward_m": pytest.approx(0.10),
        "right_m": pytest.approx(0.06),
        "yaw_deg": pytest.approx(2.0),
    }
    assert result.data["final_observation"]["distance_m"] == pytest.approx(0.44)
    assert result.data["trace"][-1]["phase"] == "confirm"


def test_low_confidence_never_commands_motion():
    observer = ScriptedObserver(
        [
            TableObservation(0.5, 0.0, 0.0, 0.1, reason="blurred"),
            TableObservation(0.5, 0.0, 0.0, 0.2, reason="still blurred"),
        ]
    )
    actuator = RecordingActuator()
    result = TableDockingController(
        observer=observer,
        actuator=actuator,
        sleep=lambda _seconds: None,
    ).run()
    assert result.success is False
    assert "not reliably visible" in result.message
    assert actuator.commands == []


def test_observer_exception_fails_closed():
    actuator = RecordingActuator()
    result = TableDockingController(
        observer=ScriptedObserver([RuntimeError("camera gone")]),
        actuator=actuator,
    ).run()
    assert result.success is False
    assert "camera gone" in result.message
    assert actuator.commands == []


def test_actuator_refusal_stops_docking():
    observer = ScriptedObserver([_observation(distance=0.9)])
    actuator = RecordingActuator([DockStepResult(False, "controller offline")])
    result = TableDockingController(
        observer=observer,
        actuator=actuator,
    ).run()
    assert result.success is False
    assert "controller offline" in result.message
    assert len(actuator.commands) == 1


def test_repeated_observation_without_progress_stalls():
    observer = ScriptedObserver([_observation(distance=0.9)] * 4)
    actuator = RecordingActuator()
    result = TableDockingController(
        observer=observer,
        actuator=actuator,
        config=TableDockingConfig(max_steps=6),
    ).run()
    assert result.success is False
    assert "stalled" in result.message
    assert len(actuator.commands) == 3


def test_cancellation_prevents_motion():
    actuator = RecordingActuator()
    result = TableDockingController(
        observer=ScriptedObserver([_observation(distance=0.9)]),
        actuator=actuator,
        cancelled=lambda: True,
    ).run()
    assert result.success is False
    assert "canceled" in result.message
    assert actuator.commands == []


def test_deadline_prevents_motion():
    ticks = iter([0.0, 2.0])
    actuator = RecordingActuator()
    controller = TableDockingController(
        observer=ScriptedObserver([_observation(distance=0.9)]),
        actuator=actuator,
        config=TableDockingConfig(deadline_seconds=1.0),
        clock=lambda: next(ticks, 2.0),
    )
    result = controller.run()
    assert result.success is False
    assert "deadline" in result.message
    assert actuator.commands == []


@pytest.mark.parametrize(
    "config, message",
    [
        ({"unknown": 1}, "unknown table docking config keys"),
        ({"max_step_m": 0.5}, "max_step_m"),
        ({"min_confidence": 2}, "min_confidence"),
        ({"max_steps": 0}, "max_steps"),
        ({"deadline_seconds": 0}, "deadline_seconds"),
        ({"settle_seconds": -1}, "settle_seconds"),
    ],
)
def test_config_validation_rejects_unsafe_values(config, message):
    with pytest.raises(TableDockingError, match=message):
        TableDockingConfig.from_mapping(config)


def test_table_catalog_rejects_bad_profile(tmp_path: Path):
    path = tmp_path / "tables.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "tables": {"edge": {"standoff_m": -1}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TableDockingError, match="standoff_m"):
        TableCatalog.load(path)


def test_table_catalog_round_trip(tmp_path: Path):
    path = tmp_path / "tables.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "tables": {
                    "edge": {
                        "standoff_m": 0.4,
                        "distance_tolerance_m": 0.05,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    catalog = TableCatalog.load(path)
    profile = catalog.get("edge")
    assert profile is not None
    assert profile.docking.standoff_m == pytest.approx(0.4)
    assert catalog.public_view()["tables"]["edge"]["max_steps"] == 12


class FakeEvidence:
    def __init__(self, response, success=True):
        self.response = response
        self.success = success
        self.error = "" if success else "no camera"
        self.snapshot_sha256 = "a" * 64
        self.captured_at = 123.0


class FakeVlm:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def observe_evidence(self, prompt):
        self.prompts.append(prompt)
        return FakeEvidence(self.response)


def test_vlm_observer_parses_geometry():
    vlm = FakeVlm(
        '{"distance_m": 0.71, "lateral_offset_m": 0.05, '
        '"yaw_error_deg": -2.5, "confidence": 0.83, "reason": "edge line"}'
    )
    observation = VlmTableObserver(vlm).observe()
    assert observation.usable(0.6)
    assert observation.distance_m == pytest.approx(0.71)
    assert observation.lateral_offset_m == pytest.approx(0.05)
    assert observation.yaw_error_deg == pytest.approx(-2.5)
    assert observation.confidence == pytest.approx(0.83)
    assert observation.snapshot_sha256 == "a" * 64


def test_vlm_observer_rejects_non_json():
    observation = VlmTableObserver(FakeVlm("not json")).observe()
    assert observation.usable(0.0) is False
    assert "JSON" in observation.reason


def test_executor_simulates_dry_run_without_hardware():
    def forbidden_factory():
        raise AssertionError("dry-run must not build hardware adapters")

    executor = TableDockingExecutor(
        observer_factory=forbidden_factory,
        actuator_factory=forbidden_factory,
    )
    task = TaskCatalog.default().get("dock_table")
    assert task is not None
    invocation = SkillInvocation("dock_table")
    context = SkillExecutionContext(task=task, config=task.executor_config)
    authorization = _make_robotd_hardware_authorization(
        "dry-run-token",
        "dock_table",
        {"dry_run": True, "task_executor": task.executor},
    )
    with _robotd_authorized_context(authorization):
        result = executor.execute(invocation, context)
    assert result.success is True
    assert result.data["dry_run"] is True


def test_executor_refuses_without_authorization():
    executor = TableDockingExecutor(
        observer_factory=lambda: ScriptedObserver([_observation()]),
        actuator_factory=RecordingActuator,
    )
    task = TaskCatalog.default().get("dock_table")
    assert task is not None
    result = executor.execute(
        SkillInvocation("dock_table"),
        SkillExecutionContext(task=task, config={}),
    )
    assert result.success is False
    assert result.error


def test_executor_runs_controller_under_authorization():
    observer = ScriptedObserver(
        [_observation(distance=0.45), _observation(distance=0.45)]
    )
    actuator = RecordingActuator()
    executor = TableDockingExecutor(
        observer_factory=lambda: observer,
        actuator_factory=lambda: actuator,
        sleep=lambda _seconds: None,
    )
    task = TaskCatalog.default().get("dock_table")
    assert task is not None
    authorization = _make_robotd_hardware_authorization(
        "real-token",
        "dock_table",
        {"task_executor": task.executor},
    )
    with _robotd_authorized_context(authorization):
        result = executor.execute(
            SkillInvocation("dock_table"),
            SkillExecutionContext(task=task, config={}),
        )
    assert result.success is True
    assert result.data["final_observation"]["distance_m"] == pytest.approx(0.45)
    assert actuator.commands == []


class FakeChassis:
    def __init__(self, outcomes=None):
        self.calls = []
        self.outcomes = list(outcomes or [])

    def move(self, direction, distance=None, angle=None, speed="medium"):
        self.calls.append((direction, distance, angle, speed))
        outcome = self.outcomes.pop(0) if self.outcomes else True
        return {"success": outcome, "error": "" if outcome else "serial busy"}


def test_chassis_actuator_decomposes_steps_in_fixed_order():
    chassis = FakeChassis()
    from robot_agent.table_docking import ChassisDockActuator

    result = ChassisDockActuator(chassis).step(
        DockStep(forward_m=0.10, right_m=-0.05, yaw_deg=5.0)
    )
    assert result.success is True
    assert chassis.calls == [
        ("left", 0.05, None, "slow"),
        ("forward", 0.10, None, "slow"),
        ("rotate_left", None, 5.0, "slow"),
    ]


def test_chassis_actuator_stops_on_first_refusal():
    chassis = FakeChassis([False])
    from robot_agent.table_docking import ChassisDockActuator

    result = ChassisDockActuator(chassis).step(
        DockStep(forward_m=0.10, right_m=0.05, yaw_deg=5.0)
    )
    assert result.success is False
    assert "serial busy" in result.message
    assert chassis.calls == [("right", 0.05, None, "slow")]
