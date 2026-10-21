from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_agent.episodes import (
    EpisodeContext,
    EpisodeError,
    EpisodeManifest,
    EpisodeRecorder,
    load_episode,
    load_episodes,
)


def _binding(executor: str = "primitive_graph") -> dict:
    return {
        "task_fingerprint": "a" * 64,
        "task_executor": executor,
        "authorized_primitives": ["moveit_task"],
    }


def _failed_result() -> dict:
    return {
        "task_id": "place_remote",
        "steps": [
            {
                "action": "precondition_evidence",
                "success": True,
                "data": {"snapshot_sha256": "b" * 64},
            },
            {
                "action": "task_backend_execute",
                "success": False,
                "message": "lease expired robotd-tok-secret",
                "data": {
                    "outcome_code": "LEASE_EXPIRED",
                    "dataset_repo_id": "local/eval_place_remote_run-1",
                    "trace": [{"node_id": "pick", "success": False}],
                    "token_secret": "do-not-store",
                },
            },
        ],
    }


def test_context_is_versioned_and_fingerprinted(tmp_path: Path):
    path = tmp_path / "context.yaml"
    path.write_text(
        "version: 1\ndeployment_id: lab-kitchen-1\ntags:\n  lighting: bright\n  layout: layout-a\n",
        encoding="utf-8",
    )
    context = EpisodeContext.load(path)
    assert context.tags == {"layout": "layout-a", "lighting": "bright"}
    assert len(context.fingerprint) == 64


def test_recorder_atomically_persists_trace_failure_and_dataset_reference(tmp_path: Path):
    context_path = tmp_path / "context.yaml"
    context_path.write_text(
        "version: 1\ndeployment_id: home-1\ntags:\n  lighting: dim\n",
        encoding="utf-8",
    )
    recorder = EpisodeRecorder(
        tmp_path / "episodes",
        context=EpisodeContext.load(context_path),
    )
    manifest = recorder.record(
        task_id="place_remote",
        execution_binding=_binding(),
        started_at=100.0,
        ended_at=105.0,
        success=False,
        terminal_state="DRY_RUN",
        result=_failed_result(),
        human_intervention=True,
        episode_id="episode-test-1",
    )
    loaded = load_episode(tmp_path / "episodes" / "episode-test-1")
    assert loaded == manifest
    assert loaded.failure_stage == "task_backend_execute"
    assert loaded.outcome_code == "LEASE_EXPIRED"
    assert loaded.dataset_refs == ("local/eval_place_remote_run-1",)
    assert loaded.human_intervention is True
    serialized = json.dumps(loaded.to_dict())
    assert "robotd-tok-secret" not in serialized
    assert "do-not-store" not in serialized
    assert "[REDACTED]" in serialized
    episode_dir = tmp_path / "episodes" / "episode-test-1"
    assert oct(episode_dir.stat().st_mode & 0o777) == "0o700"
    assert oct((episode_dir / "manifest.json").stat().st_mode & 0o777) == "0o600"
    assert list((tmp_path / "episodes").glob(".*.tmp")) == []


def test_manifest_detects_tampering(tmp_path: Path):
    recorder = EpisodeRecorder(tmp_path / "episodes")
    recorder.record(
        task_id="fold_simple",
        execution_binding=_binding("learned_policy"),
        started_at=10.0,
        ended_at=11.0,
        success=True,
        terminal_state="DRY_RUN",
        result={"steps": [], "outcome_code": "SUCCESS"},
        episode_id="episode-tamper",
    )
    path = tmp_path / "episodes" / "episode-tamper" / "manifest.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["success"] = False
    raw["failure_stage"] = "forged"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(EpisodeError, match="hash does not match"):
        load_episode(path)


def test_failed_manifest_gets_deterministic_fallback_stage():
    manifest = EpisodeManifest.create(
        task_id="task",
        execution_binding=_binding(),
        started_at=1.0,
        ended_at=2.0,
        success=False,
        terminal_state="ERROR",
        result={},
    )
    assert manifest.failure_stage == "execution"
    assert manifest.outcome_code == "FAILED"


def test_failure_stage_is_found_inside_robotd_agent_data():
    manifest = EpisodeManifest.create(
        task_id="task",
        execution_binding=_binding(),
        started_at=1.0,
        ended_at=2.0,
        success=False,
        terminal_state="ERROR",
        result={
            "agent_data": {
                "steps": [
                    {"action": "postcondition_evidence", "success": False}
                ]
            }
        },
    )
    assert manifest.failure_stage == "postcondition_evidence"


def test_episode_ids_are_immutable_and_root_loader_verifies_all(tmp_path: Path):
    recorder = EpisodeRecorder(tmp_path / "episodes")
    values = {
        "task_id": "fold_simple",
        "execution_binding": _binding("learned_policy"),
        "started_at": 10.0,
        "ended_at": 11.0,
        "success": True,
        "terminal_state": "DRY_RUN",
        "result": {"steps": []},
        "episode_id": "episode-one",
    }
    recorder.record(**values)
    with pytest.raises(EpisodeError, match="already exists"):
        recorder.record(**values)
    assert [episode.episode_id for episode in load_episodes(tmp_path / "episodes")] == [
        "episode-one"
    ]
