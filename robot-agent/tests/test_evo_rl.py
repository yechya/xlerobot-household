from __future__ import annotations

from pathlib import Path

import pytest

from robot_agent.episodes import EpisodeContext, EpisodeManifest
from robot_agent.evo_rl import (
    EvoRLBridgeError,
    build_evo_rl_export,
    load_evo_rl_export,
    write_evo_rl_export,
)


def _episode(
    episode_id: str,
    *,
    success: bool,
    intervention: bool = False,
    policy: bool = True,
) -> EpisodeManifest:
    binding = {
        "task_fingerprint": "a" * 64,
        "task_executor": "learned_policy",
    }
    if policy:
        binding.update({"policy_id": "smolvla-fold-v1", "policy_fingerprint": "b" * 64})
    result = {
        "steps": []
        if success
        else [{"action": "task_backend_execute", "success": False}],
        "dataset_repo_id": f"local/eval_fold_{episode_id}",
    }
    return EpisodeManifest.create(
        task_id="fold_simple",
        execution_binding=binding,
        started_at=10.0,
        ended_at=11.0,
        success=success,
        terminal_state="DRY_RUN",
        result=result,
        context=EpisodeContext(
            deployment_id="home-1",
            tags={"lighting": "bright", "layout": "layout-a"},
            fingerprint="c" * 64,
        ),
        human_intervention=intervention,
        episode_id=episode_id,
    )


def test_export_preserves_evo_rl_labels_and_rgb_contract(tmp_path: Path):
    export = build_evo_rl_export(
        [_episode("episode-2", success=False, intervention=True), _episode("episode-1", success=True)],
        generated_at=20.0,
    )

    assert export.camera_roles == ("head_cam", "left_cam", "right_cam")
    assert [episode.episode_id for episode in export.episodes] == ["episode-1", "episode-2"]
    failed = export.episodes[1].to_dict()
    assert failed["episode_success"] == "failure"
    assert failed["complementary_info"] == {
        "is_intervention": 1,
        "intervention_scope": "episode",
        "collector_policy_id": "smolvla-fold-v1",
        "policy_action_available": False,
    }
    assert failed["dataset_refs"] == ["local/eval_fold_episode-2"]

    path = tmp_path / "evo-rl" / "export.json"
    write_evo_rl_export(path, export)
    assert load_evo_rl_export(path) == export
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_export_marks_unbound_human_collection_without_fabricating_policy_data():
    record = build_evo_rl_export(
        [_episode("episode-human", success=False, intervention=True, policy=False)],
        generated_at=20.0,
    ).episodes[0].to_dict()

    assert record["policy_id"] == ""
    assert record["policy_fingerprint"] == ""
    assert record["complementary_info"]["collector_policy_id"] == "human"
    assert record["complementary_info"]["policy_action_available"] is False


def test_export_rejects_unknown_camera_roles():
    export = build_evo_rl_export([_episode("episode-1", success=True)], generated_at=20.0)
    with pytest.raises(EvoRLBridgeError, match="camera_roles"):
        export.__class__(
            **{
                **export.__dict__,
                "camera_roles": ("head_cam", "depth_cam"),
            }
        ).validate()
