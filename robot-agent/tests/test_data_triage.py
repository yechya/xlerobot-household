from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from robot_agent.data_triage import (
    DataTriageError,
    DataTriageSpec,
    evaluate_data_triage,
    load_triage_report,
    write_triage_report,
)
from robot_agent.episodes import EpisodeContext, EpisodeManifest


def _spec(tmp_path: Path, **gate_overrides) -> DataTriageSpec:
    gate = {
        "task_ids": ["fold_simple"],
        "executors": ["learned_policy", "primitive_graph"],
        "failure_stages": ["task_backend_execute", "contact_task"],
        "min_attempts": 4,
        "min_learning_failures": 2,
        "min_learning_failure_rate": 0.4,
        "max_system_failure_rate": 0.0,
        "max_unclassified_failure_rate": 0.0,
        "max_intervention_rate": 0.1,
        "coverage": {"lighting": 2, "layout": 2},
    }
    gate.update(gate_overrides)
    path = tmp_path / "triage.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "approved": True,
                "triage_id": "home-triage-v1",
                "system_routes": {
                    "perception_fix": ["precondition_evidence", "scene_resolve"],
                    "planning_fix": ["moveit_task"],
                },
                "learning_gate": gate,
            }
        ),
        encoding="utf-8",
    )
    return DataTriageSpec.load(path)


def _episode(
    episode_id: str,
    *,
    success: bool,
    stage: str = "task_backend_execute",
    lighting: str = "bright",
    layout: str = "layout-a",
    task_id: str = "fold_simple",
    task_fingerprint: str = "a" * 64,
    executor: str = "learned_policy",
    intervention: bool = False,
    deployment_fingerprint: str = "d" * 64,
) -> EpisodeManifest:
    context = EpisodeContext(
        deployment_id="home-1",
        tags={"lighting": lighting, "layout": layout},
        fingerprint="b" * 64,
    )
    result = {"steps": []}
    if not success:
        result["steps"] = [{"action": stage, "success": False, "data": {}}]
    return EpisodeManifest.create(
        task_id=task_id,
        execution_binding={
            "task_fingerprint": task_fingerprint,
            "task_executor": executor,
            "deployment_fingerprint": deployment_fingerprint,
        },
        started_at=10.0,
        ended_at=11.0,
        success=success,
        terminal_state="DRY_RUN",
        result=result,
        context=context,
        human_intervention=intervention,
        episode_id=episode_id,
    )


def _eligible_episodes() -> list[EpisodeManifest]:
    return [
        _episode("episode-1", success=False, lighting="bright", layout="layout-a"),
        _episode("episode-2", success=False, lighting="dim", layout="layout-b"),
        _episode("episode-3", success=True, lighting="bright", layout="layout-b"),
        _episode("episode-4", success=True, lighting="dim", layout="layout-a"),
    ]


def test_triage_recommends_only_repeated_cross_context_learning_failures(tmp_path: Path):
    report = evaluate_data_triage(_spec(tmp_path), _eligible_episodes(), generated_at=20.0)

    candidate = report.candidates[0]
    assert candidate.eligible is True
    assert candidate.learning_failures == 2
    assert candidate.coverage_values == {
        "layout": ("layout-a", "layout-b"),
        "lighting": ("bright", "dim"),
    }
    assert candidate.recommended_episode_ids == ("episode-1", "episode-2")
    assert all(report.system_routes[route] == () for route in report.system_routes)


def test_triage_routes_system_failures_and_blocks_learning_collection(tmp_path: Path):
    episodes = _eligible_episodes() + [
        _episode("episode-system", success=False, stage="precondition_evidence")
    ]
    report = evaluate_data_triage(_spec(tmp_path), episodes, generated_at=20.0)
    candidate = report.candidates[0]

    assert candidate.eligible is False
    assert candidate.recommended_episode_ids == ()
    assert candidate.system_failures == 1
    assert any("system failure rate" in reason for reason in candidate.reasons)
    assert report.system_routes["perception_fix"] == ("episode-system",)


def test_triage_blocks_unknown_failures_intervention_and_weak_coverage(tmp_path: Path):
    episodes = [
        _episode("episode-1", success=False, stage="unknown_stage", intervention=True),
        _episode("episode-2", success=False),
        _episode("episode-3", success=True),
    ]
    report = evaluate_data_triage(_spec(tmp_path), episodes, generated_at=20.0)
    candidate = report.candidates[0]

    assert candidate.eligible is False
    assert report.unclassified_episode_ids == ("episode-1",)
    assert any("attempts" in reason for reason in candidate.reasons)
    assert any("unclassified failure rate" in reason for reason in candidate.reasons)
    assert any("intervention rate" in reason for reason in candidate.reasons)
    assert any("coverage axis" in reason for reason in candidate.reasons)


def test_triage_never_pools_task_fingerprints_or_unapproved_tasks(tmp_path: Path):
    episodes = _eligible_episodes() + [
        _episode(
            "episode-other-contract",
            success=False,
            task_fingerprint="c" * 64,
        ),
        _episode("episode-other-task", success=False, task_id="wipe_table"),
        _episode(
            "episode-other-deployment",
            success=False,
            deployment_fingerprint="e" * 64,
        ),
    ]
    report = evaluate_data_triage(_spec(tmp_path), episodes, generated_at=20.0)

    assert len(report.candidates) == 4
    assert sum(candidate.eligible for candidate in report.candidates) == 1
    other_task = next(
        candidate for candidate in report.candidates if candidate.task_id == "wipe_table"
    )
    assert any("not approved" in reason for reason in other_task.reasons)


def test_triage_spec_rejects_overlapping_system_and_learning_stages(tmp_path: Path):
    with pytest.raises(DataTriageError, match="overlap"):
        _spec(tmp_path, failure_stages=["moveit_task"])


def test_triage_report_is_private_hashed_and_detects_tampering(tmp_path: Path):
    report = evaluate_data_triage(_spec(tmp_path), _eligible_episodes(), generated_at=20.0)
    path = tmp_path / "reports" / "triage.json"
    write_triage_report(path, report)

    assert load_triage_report(path) == report
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["candidates"][0]["eligible"] = False
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DataTriageError, match="hash does not match"):
        load_triage_report(path)


def test_triage_episode_set_digest_changes_with_source_manifests(tmp_path: Path):
    spec = _spec(tmp_path)
    first = evaluate_data_triage(spec, _eligible_episodes(), generated_at=20.0)
    changed = _eligible_episodes() + [_episode("episode-5", success=True)]
    second = evaluate_data_triage(spec, changed, generated_at=20.0)
    assert first.episode_set_digest != second.episode_set_digest
