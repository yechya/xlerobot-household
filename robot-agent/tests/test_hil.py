from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_agent.hil import HilEvent, HilPolicy, evaluate_events, load_events
from robot_agent.hil_recorder import JsonlEventWriter


def event(
    scenario_id: str,
    scenario: str,
    kind: str,
    timestamp: float,
    **data,
) -> HilEvent:
    return HilEvent(scenario_id, scenario, kind, timestamp, data)


def passing_scenario(scenario_id: str, scenario: str, start: float) -> list[HilEvent]:
    return [
        event(scenario_id, scenario, "scenario_start", start),
        event(
            scenario_id,
            scenario,
            "velocity",
            start + 0.01,
            linear_speed=0.12,
            angular_speed=0.1,
        ),
        event(scenario_id, scenario, "fault", start + 0.1),
        event(
            scenario_id,
            scenario,
            "velocity",
            start + 0.2,
            linear_speed=0.08,
            angular_speed=0.06,
        ),
        event(
            scenario_id,
            scenario,
            "velocity",
            start + 0.4,
            linear_speed=0.01,
            angular_speed=0.01,
        ),
        event(
            scenario_id,
            scenario,
            "velocity",
            start + 0.5,
            linear_speed=0.0,
            angular_speed=0.0,
        ),
        event(
            scenario_id,
            scenario,
            "velocity",
            start + 0.6,
            linear_speed=0.0,
            angular_speed=0.0,
        ),
        event(
            scenario_id,
            scenario,
            "action_terminal",
            start + 0.7,
            status="canceled",
        ),
        event(scenario_id, scenario, "scenario_end", start + 0.8),
    ]


def test_hil_report_requires_all_scenarios_and_passes_valid_evidence():
    scenarios = ("heartbeat_loss", "robotd_killed", "nav2_killed", "sensor_loss")
    events = []
    for index, scenario in enumerate(scenarios):
        events.extend(passing_scenario(f"run-{index}", scenario, index * 10.0))
    policy = HilPolicy(required_scenarios=scenarios)
    report = evaluate_events(events, policy)
    assert report.passed is True
    assert report.missing_scenarios == ()
    assert all(result.stop_latency_seconds == pytest.approx(0.3) for result in report.scenarios)


def test_hil_report_fails_collision_late_stop_and_missing_scenario():
    events = passing_scenario("run-1", "heartbeat_loss", 0.0)
    events.append(event("run-1", "heartbeat_loss", "collision", 0.3, detected=True))
    policy = HilPolicy(
        max_stop_latency_seconds=0.2,
        required_scenarios=("heartbeat_loss", "sensor_loss"),
    )
    report = evaluate_events(events, policy)
    assert report.passed is False
    assert report.missing_scenarios == ("sensor_loss",)
    assert "collision was detected" in report.scenarios[0].failures
    assert any("stop latency" in failure for failure in report.scenarios[0].failures)


def test_sparse_velocity_evidence_does_not_prove_stable_stop():
    events = [
        event("run", "nav2_killed", "fault", 1.0),
        event("run", "nav2_killed", "velocity", 1.1, linear_speed=0.0, angular_speed=0.0),
        event("run", "nav2_killed", "velocity", 1.5, linear_speed=0.0, angular_speed=0.0),
        event("run", "nav2_killed", "action_terminal", 1.2, status="aborted"),
    ]
    report = evaluate_events(events)
    assert report.passed is False
    assert "stable zero velocity" in report.scenarios[0].failures[0]


def test_event_writer_creates_private_jsonl(tmp_path: Path):
    path = tmp_path / "evidence" / "events.jsonl"
    writer = JsonlEventWriter(path, "run-1", "heartbeat_loss")
    writer.write("fault", {"marker": "fault"})
    writer.close()
    loaded = load_events(path)
    assert len(loaded) == 1
    assert loaded[0].scenario == "heartbeat_loss"
    assert json.loads(path.read_text(encoding="utf-8"))["kind"] == "fault"
    assert path.stat().st_mode & 0o077 == 0
