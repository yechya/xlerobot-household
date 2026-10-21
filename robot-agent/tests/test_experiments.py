from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from robot_agent.episodes import EpisodeContext, EpisodeManifest, load_episodes
from robot_agent.experiments import (
    ExperimentError,
    ExperimentSpec,
    evaluate_experiment,
    evaluate_experiment_store,
    load_report,
    write_report,
)

TASK_FINGERPRINT = "a" * 64
POLICY_FINGERPRINT = "b" * 64


def _context(tmp_path: Path, episode_id: str, lighting: str, layout: str) -> EpisodeContext:
    path = tmp_path / f"{episode_id}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "deployment_id": "lab-kitchen",
                "tags": {"lighting": lighting, "layout": layout},
            }
        ),
        encoding="utf-8",
    )
    return EpisodeContext.load(path)


def _episode(
    tmp_path: Path,
    episode_id: str,
    mode: str,
    *,
    success: bool = True,
    lighting: str = "bright",
    layout: str = "layout-a",
    policy_fingerprint: str = POLICY_FINGERPRINT,
    task_id: str = "pick_up_bottle",
) -> EpisodeManifest:
    result = {"steps": []}
    if not success:
        result["steps"] = [{"action": "postcondition_evidence", "success": False}]
    return EpisodeManifest.create(
        task_id=task_id,
        execution_binding={
            "task_fingerprint": TASK_FINGERPRINT,
            "task_executor": "learned_policy",
            "policy_fingerprint": policy_fingerprint,
            "experiment_mode": mode,
        },
        started_at=10.0,
        ended_at=11.0,
        success=success,
        terminal_state="DRY_RUN",
        result=result,
        context=_context(tmp_path, episode_id, lighting, layout),
        episode_id=episode_id,
    )


def _spec(tmp_path: Path, *, min_episodes: int = 1) -> ExperimentSpec:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "experiment_id": "rgb-vla-ablation-v1",
                "task_id": "pick_up_bottle",
                "task_fingerprint": TASK_FINGERPRINT,
                "modes": ["direct_vla", "rgb_precondition_vla", "rgb_pre_post_vla"],
                "binding_equals": {"policy_fingerprint": POLICY_FINGERPRINT},
                "coverage": {
                    "lighting": ["bright", "dim"],
                    "layout": ["layout-a", "layout-b"],
                },
                "min_episodes_per_mode": min_episodes,
            }
        ),
        encoding="utf-8",
    )
    return ExperimentSpec.load(path)


def test_spec_loads_and_fingerprints_canonical_content(tmp_path: Path):
    spec = _spec(tmp_path)
    assert spec.modes[0] == "direct_vla"
    assert len(spec.report_fingerprint) == 64

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "experiment_id": "missing-baseline",
                "task_id": "pick_up_bottle",
                "task_fingerprint": TASK_FINGERPRINT,
                "modes": ["rgb_pre_post_vla"],
                "binding_equals": {"policy_fingerprint": POLICY_FINGERPRINT},
                "coverage": {"lighting": ["bright"]},
                "min_episodes_per_mode": 1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ExperimentError, match="include direct_vla"):
        ExperimentSpec.load(invalid)


def test_evaluation_filters_task_policy_and_mode_exactly(tmp_path: Path):
    episodes = [
        _episode(tmp_path, "direct-1", "direct_vla", success=True),
        _episode(tmp_path, "direct-2", "direct_vla", success=False, lighting="dim"),
        _episode(tmp_path, "pre-1", "rgb_precondition_vla", success=True, layout="layout-b"),
        _episode(tmp_path, "post-1", "rgb_pre_post_vla", success=True),
        _episode(tmp_path, "other-policy", "direct_vla", success=True, policy_fingerprint="c" * 64),
        _episode(tmp_path, "other-task", "direct_vla", success=True, task_id="fold_simple"),
    ]
    report = evaluate_experiment(_spec(tmp_path), episodes)
    by_mode = {mode.mode: mode for mode in report.modes}
    assert report.passed is True
    assert by_mode["direct_vla"].episode_ids == ("direct-1", "direct-2")
    assert by_mode["direct_vla"].success_rate == 0.5
    assert by_mode["direct_vla"].coverage_counts["lighting"] == {"bright": 1, "dim": 1}
    assert report.comparison["best_mode"] in {"rgb_precondition_vla", "rgb_pre_post_vla"}
    assert report.comparison["direct_vla_episode_count"] == 2


def test_insufficient_modes_and_missing_direct_baseline_fail_without_fabrication(tmp_path: Path):
    report = evaluate_experiment(
        _spec(tmp_path, min_episodes=2),
        [
            _episode(tmp_path, "direct-1", "direct_vla"),
            _episode(tmp_path, "pre-1", "rgb_precondition_vla"),
            _episode(tmp_path, "post-1", "rgb_pre_post_vla"),
        ],
    )
    assert report.passed is False
    assert any("below 2" in failure for failure in report.failures)

    no_direct = evaluate_experiment(
        _spec(tmp_path),
        [
            _episode(tmp_path, "pre-only", "rgb_precondition_vla"),
            _episode(tmp_path, "post-only", "rgb_pre_post_vla"),
        ],
    )
    assert no_direct.passed is False
    assert "direct_vla has no usable episodes for comparison" in no_direct.failures


def test_report_is_hashed_and_tampering_is_rejected(tmp_path: Path):
    episodes_root = tmp_path / "episodes"
    for episode in (
        _episode(tmp_path, "direct-1", "direct_vla"),
        _episode(tmp_path, "pre-1", "rgb_precondition_vla"),
        _episode(tmp_path, "post-1", "rgb_pre_post_vla"),
    ):
        episode_dir = episodes_root / episode.episode_id
        episode_dir.mkdir(parents=True)
        (episode_dir / "manifest.json").write_text(
            json.dumps(episode.to_dict(), sort_keys=True), encoding="utf-8"
        )
    report = evaluate_experiment_store(_spec(tmp_path), episodes_root)
    report_path = tmp_path / "reports" / "experiment.json"
    write_report(report_path, report)
    assert load_report(report_path) == report

    raw = json.loads(report_path.read_text(encoding="utf-8"))
    raw["comparison"]["best_mode"] = "rgb_precondition_vla"
    report_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ExperimentError, match="hash does not match"):
        load_report(report_path)


def test_duplicate_input_episode_ids_are_rejected(tmp_path: Path):
    episode = _episode(tmp_path, "duplicate", "direct_vla")
    with pytest.raises(ExperimentError, match="duplicate episode IDs"):
        evaluate_experiment(_spec(tmp_path), [episode, episode])
