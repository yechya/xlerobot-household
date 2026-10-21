from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_agent.safety import (
    FileSafetyInterlock,
    SafetyPolicy,
    SafetySnapshot,
    write_safety_snapshot,
)


def _policy(tmp_path: Path) -> SafetyPolicy:
    path = tmp_path / "safety.yaml"
    path.write_text(
        """
version: 1
max_age_seconds: 0.5
allowed_source_prefixes: [safety/approved/]
require_physical_estop: true
deny_when_privacy_active: true
motion:
  min_person_distance_m: 0.8
  min_pet_distance_m: 0.5
manipulation:
  min_person_distance_m: 1.2
  min_pet_distance_m: 0.8
  require_clear_workspace: true
""".strip(),
        encoding="utf-8",
    )
    return SafetyPolicy.load(path)


def _snapshot(sequence: int = 1, **overrides) -> SafetySnapshot:
    values = {
        "sequence": sequence,
        "observed_at": 100.0,
        "source": "safety/approved/person-pet-v1",
        "sensor_health": "ok",
        "physical_estop_available": True,
        "emergency_stop_pressed": False,
        "person_count": 0,
        "nearest_person_m": None,
        "pet_count": 0,
        "nearest_pet_m": None,
        "person_in_workspace": False,
        "pet_in_workspace": False,
        "privacy_zone_active": False,
    }
    values.update(overrides)
    return SafetySnapshot.create(**values)


def test_fresh_clear_snapshot_allows_motion_and_manipulation(tmp_path: Path):
    path = tmp_path / "safety.json"
    write_safety_snapshot(path, _snapshot())
    interlock = FileSafetyInterlock(path, _policy(tmp_path))
    motion = interlock.check("motion", now=100.2)
    manipulation = interlock.check("manipulation", now=100.2)
    assert motion.allowed is True
    assert manipulation.allowed is True
    assert motion.snapshot_sequence == 1
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_distance_and_workspace_rules_are_operation_specific(tmp_path: Path):
    path = tmp_path / "safety.json"
    write_safety_snapshot(
        path,
        _snapshot(
            person_count=1,
            nearest_person_m=1.0,
            pet_count=1,
            nearest_pet_m=0.6,
            person_in_workspace=True,
        ),
    )
    interlock = FileSafetyInterlock(path, _policy(tmp_path))
    assert interlock.check("motion", now=100.1).allowed is True
    manipulation = interlock.check("manipulation", now=100.1)
    assert manipulation.allowed is False
    assert any("nearest person" in reason for reason in manipulation.reasons)
    assert any("workspace" in reason for reason in manipulation.reasons)


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"sensor_health": "degraded"}, "sensor health"),
        ({"physical_estop_available": False}, "unavailable"),
        ({"emergency_stop_pressed": True}, "pressed"),
        ({"privacy_zone_active": True}, "privacy"),
        ({"source": "unreviewed/model"}, "source"),
    ],
)
def test_health_estop_privacy_and_source_fail_closed(tmp_path: Path, overrides, reason):
    path = tmp_path / "safety.json"
    write_safety_snapshot(path, _snapshot(**overrides))
    decision = FileSafetyInterlock(path, _policy(tmp_path)).check("motion", now=100.1)
    assert decision.allowed is False
    assert any(reason in value for value in decision.reasons)


def test_stale_future_tampered_and_public_files_are_rejected(tmp_path: Path):
    path = tmp_path / "safety.json"
    write_safety_snapshot(path, _snapshot())
    interlock = FileSafetyInterlock(path, _policy(tmp_path))
    assert interlock.check("motion", now=100.6).allowed is False
    assert interlock.check("motion", now=99.9).allowed is False

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["privacy_zone_active"] = True
    path.write_text(json.dumps(raw), encoding="utf-8")
    tampered = FileSafetyInterlock(path, _policy(tmp_path)).check("motion", now=100.1)
    assert tampered.allowed is False
    assert "hash" in tampered.reasons[0]

    write_safety_snapshot(path, _snapshot(sequence=2))
    path.chmod(0o644)
    public = FileSafetyInterlock(path, _policy(tmp_path)).check("motion", now=100.1)
    assert public.allowed is False
    assert "group" in public.reasons[0]


def test_sequence_rollback_and_same_sequence_rewrite_are_rejected(tmp_path: Path):
    path = tmp_path / "safety.json"
    interlock = FileSafetyInterlock(path, _policy(tmp_path))
    write_safety_snapshot(path, _snapshot(sequence=2))
    assert interlock.check("motion", now=100.1).allowed is True

    write_safety_snapshot(path, _snapshot(sequence=1))
    rollback = interlock.check("motion", now=100.1)
    assert rollback.allowed is False
    assert any("rolled back" in reason for reason in rollback.reasons)

    write_safety_snapshot(path, _snapshot(sequence=2, privacy_zone_active=True))
    rewritten = interlock.check("motion", now=100.1)
    assert rewritten.allowed is False
    assert any("without a new sequence" in reason for reason in rewritten.reasons)


def test_invalid_count_distance_pair_is_rejected():
    with pytest.raises(ValueError, match="must be null"):
        _snapshot(person_count=0, nearest_person_m=1.0)
