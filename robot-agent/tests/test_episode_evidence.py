from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest
import yaml

from robot_agent.episode_evidence import (
    EpisodeEvidenceError,
    EvidenceIndexSpec,
    index_episode_evidence,
    load_episode_evidence,
    load_evidence_store,
)
from robot_agent.episode_replay import (
    EpisodeReplayError,
    ReplaySpec,
    build_replay_plan,
    load_replay_plan,
    write_replay_plan,
)
from robot_agent.episodes import EpisodeError, EpisodeRecorder


def _binding(executor="primitive_graph"):
    return {
        "task_fingerprint": "a" * 64,
        "task_executor": executor,
    }


def _record(
    root: Path,
    episode_id: str,
    *,
    executor="primitive_graph",
    success=False,
    stage="scene_resolve",
):
    result = (
        {"steps": [], "outcome_code": "SUCCESS"}
        if success
        else {
            "steps": [{"action": stage, "success": False}],
            "outcome_code": "FAILED_STAGE",
        }
    )
    return EpisodeRecorder(root).record(
        task_id="tidy_table",
        execution_binding=_binding(executor),
        started_at=10.0,
        ended_at=12.0,
        success=success,
        terminal_state="SUCCEEDED" if success else "ERROR",
        result=result,
        episode_id=episode_id,
    )


def _write_evidence_spec(
    tmp_path: Path,
    episode_id: str,
    *,
    started=10.0,
    ended=12.0,
    rgb_source: Path | None = None,
):
    rgb = rgb_source or (tmp_path / f"{episode_id}-head.mp4")
    if rgb_source is None:
        rgb.write_bytes(b"fake rgb video")
    state = tmp_path / f"{episode_id}-state.mcap"
    state.write_bytes(b"fake robot state")
    path = tmp_path / f"{episode_id}-evidence.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "approved": True,
                "episode_id": episode_id,
                "max_total_bytes": 1024 * 1024,
                "max_time_margin_seconds": 0.5,
                "artifacts": [
                    {
                        "artifact_id": "head_rgb",
                        "kind": "rgb_video",
                        "source": str(rgb),
                        "mime_type": "video/mp4",
                        "captured_at_start": started,
                        "captured_at_end": ended,
                    },
                    {
                        "artifact_id": "robot_state",
                        "kind": "robot_state",
                        "source": str(state),
                        "mime_type": "application/mcap",
                        "captured_at_start": started,
                        "captured_at_end": ended,
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _index(tmp_path: Path, episodes: Path, episode_id: str):
    return index_episode_evidence(
        episodes / episode_id,
        EvidenceIndexSpec.load(_write_evidence_spec(tmp_path, episode_id)),
        created_at=20.0,
    )


def _write_replay_spec(tmp_path: Path, **changes):
    raw = {
        "version": 1,
        "replay_id": "triage-v1",
        "filters": {
            "task_ids": ["tidy_table"],
            "executors": [],
            "success": "failure",
            "failure_stages": [],
            "outcome_codes": [],
            "tags": {},
        },
        "required_evidence_kinds": ["rgb_video", "robot_state"],
        "routes": [
            {
                "route_id": "perception_fix",
                "executors": [],
                "failure_stages": ["scene_resolve"],
                "outcome_codes": [],
            },
            {
                "route_id": "learned_review",
                "executors": ["learned_policy"],
                "failure_stages": ["task_backend_execute"],
                "outcome_codes": [],
            },
        ],
        "default_route": "system_review",
    }
    raw.update(changes)
    path = tmp_path / "replay.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_evidence_index_is_atomic_private_and_binds_copied_files(tmp_path: Path):
    episodes = tmp_path / "episodes"
    episode = _record(episodes, "episode-one")
    manifest = _index(tmp_path, episodes, episode.episode_id)
    loaded = load_episode_evidence(
        episodes / episode.episode_id,
        required_kinds=("rgb_video", "robot_state"),
    )
    assert loaded == manifest
    assert loaded.episode_manifest_sha256 == episode.manifest_sha256
    evidence_dir = episodes / episode.episode_id / "evidence"
    assert stat.S_IMODE(evidence_dir.stat().st_mode) == 0o700
    for artifact in loaded.artifacts:
        path = evidence_dir / artifact.relative_path
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact.sha256
    assert not list((episodes / episode.episode_id).glob(".evidence.*.tmp"))
    with pytest.raises(EpisodeEvidenceError, match="already exists"):
        _index(tmp_path, episodes, episode.episode_id)


def test_evidence_loader_rejects_media_tampering_and_missing_kind(tmp_path: Path):
    episodes = tmp_path / "episodes"
    _record(episodes, "episode-one")
    manifest = _index(tmp_path, episodes, "episode-one")
    with pytest.raises(EpisodeEvidenceError, match="missing required kinds"):
        load_episode_evidence(
            episodes / "episode-one", required_kinds=("wrench",)
        )
    artifact = manifest.artifacts[0]
    path = episodes / "episode-one" / "evidence" / artifact.relative_path
    path.write_bytes(b"tampered")
    with pytest.raises(EpisodeEvidenceError, match="was modified"):
        load_episode_evidence(episodes / "episode-one")


def test_evidence_index_rejects_symlinks_and_time_outside_episode(tmp_path: Path):
    episodes = tmp_path / "episodes"
    _record(episodes, "episode-one")
    real = tmp_path / "real.mp4"
    real.write_bytes(b"video")
    linked = tmp_path / "linked.mp4"
    linked.symlink_to(real)
    spec = EvidenceIndexSpec.load(
        _write_evidence_spec(tmp_path, "episode-one", rgb_source=linked)
    )
    with pytest.raises(EpisodeEvidenceError, match="regular file"):
        index_episode_evidence(episodes / "episode-one", spec)

    spec = EvidenceIndexSpec.load(
        _write_evidence_spec(tmp_path, "episode-one", started=1.0, ended=2.0)
    )
    with pytest.raises(EpisodeEvidenceError, match="outside episode bounds"):
        index_episode_evidence(episodes / "episode-one", spec)


def test_store_validation_requires_evidence_for_every_episode(tmp_path: Path):
    episodes = tmp_path / "episodes"
    _record(episodes, "episode-one")
    _record(episodes, "episode-two")
    _index(tmp_path, episodes, "episode-one")
    with pytest.raises(EpisodeEvidenceError, match="cannot load"):
        load_evidence_store(episodes, required_kinds=("rgb_video",))


def test_replay_plan_filters_before_evidence_and_routes_failures(tmp_path: Path):
    episodes = tmp_path / "episodes"
    _record(episodes, "episode-perception", stage="scene_resolve")
    _record(
        episodes,
        "episode-learned",
        executor="learned_policy",
        stage="task_backend_execute",
    )
    _record(episodes, "episode-success", success=True)
    _index(tmp_path, episodes, "episode-perception")
    _index(tmp_path, episodes, "episode-learned")
    spec = ReplaySpec.load(_write_replay_spec(tmp_path))
    plan = build_replay_plan(spec, episodes)
    assert plan["episode_count"] == 2
    assert plan["route_counts"] == {"learned_review": 1, "perception_fix": 1}
    assert [entry["episode_id"] for entry in plan["entries"]] == [
        "episode-learned",
        "episode-perception",
    ]
    assert all(len(entry["artifacts"]) == 2 for entry in plan["entries"])
    assert len(plan["source_set_sha256"]) == 64
    assert len(plan["report_sha256"]) == 64


def test_replay_plan_rejects_ambiguous_routes_and_existing_output(tmp_path: Path):
    episodes = tmp_path / "episodes"
    _record(
        episodes,
        "episode-learned",
        executor="learned_policy",
        stage="task_backend_execute",
    )
    _index(tmp_path, episodes, "episode-learned")
    raw = yaml.safe_load(_write_replay_spec(tmp_path).read_text(encoding="utf-8"))
    raw["routes"].append(
        {
            "route_id": "also_learned",
            "executors": ["learned_policy"],
            "failure_stages": [],
            "outcome_codes": [],
        }
    )
    ambiguous = tmp_path / "ambiguous.yaml"
    ambiguous.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(EpisodeReplayError, match="multiple replay routes"):
        build_replay_plan(ReplaySpec.load(ambiguous), episodes)

    plan = build_replay_plan(ReplaySpec.load(_write_replay_spec(tmp_path)), episodes)
    output = tmp_path / "plan.json"
    write_replay_plan(output, plan)
    assert load_replay_plan(output) == plan
    original = output.read_bytes()
    with pytest.raises(EpisodeReplayError, match="already exists"):
        write_replay_plan(output, plan)
    assert output.read_bytes() == original
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_replay_plan_loader_rejects_tampering_and_unsafe_artifact_path(tmp_path: Path):
    episodes = tmp_path / "episodes"
    _record(episodes, "episode-one", stage="scene_resolve")
    _index(tmp_path, episodes, "episode-one")
    plan = build_replay_plan(ReplaySpec.load(_write_replay_spec(tmp_path)), episodes)
    output = tmp_path / "plan.json"
    write_replay_plan(output, plan)

    tampered = json.loads(output.read_text(encoding="utf-8"))
    tampered["entries"][0]["route_id"] = "different_route"
    output.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(EpisodeReplayError, match="route counts"):
        load_replay_plan(output)

    unsafe = json.loads(json.dumps(plan))
    unsafe["entries"][0]["artifacts"][0]["path"] = "../outside.mp4"
    with pytest.raises(EpisodeReplayError, match="artifact path"):
        write_replay_plan(tmp_path / "unsafe.json", unsafe)


def test_evidence_manifest_detects_episode_manifest_replacement(tmp_path: Path):
    episodes = tmp_path / "episodes"
    _record(episodes, "episode-one")
    _index(tmp_path, episodes, "episode-one")
    manifest_path = episodes / "episode-one" / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["result"]["extra"] = "tampered"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(EpisodeError, match="hash does not match"):
        load_episode_evidence(episodes / "episode-one")
