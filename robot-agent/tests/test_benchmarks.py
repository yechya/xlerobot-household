from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from robot_agent.benchmarks import (
    BenchmarkError,
    BenchmarkSpec,
    evaluate_benchmark,
    evaluate_benchmark_store,
    load_report,
    write_report,
)
from robot_agent.episode_evidence import EvidenceIndexSpec, index_episode_evidence
from robot_agent.episodes import EpisodeContext, EpisodeRecorder, load_episodes

POLICY_FINGERPRINT = "b" * 64
TASK_FINGERPRINT = "a" * 64


def _context(tmp_path: Path, name: str, lighting: str, layout: str) -> EpisodeContext:
    path = tmp_path / f"context-{name}.yaml"
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


def _record(
    tmp_path: Path,
    episode_id: str,
    lighting: str,
    layout: str,
    *,
    success: bool,
    policy_fingerprint: str = POLICY_FINGERPRINT,
    intervention: bool = False,
) -> None:
    result = {"steps": []}
    if not success:
        result["steps"] = [
            {"action": "task_backend_execute", "success": False, "data": {}}
        ]
    EpisodeRecorder(
        tmp_path / "episodes",
        context=_context(tmp_path, episode_id, lighting, layout),
    ).record(
        task_id="fold_simple",
        execution_binding={
            "task_fingerprint": TASK_FINGERPRINT,
            "task_executor": "learned_policy",
            "policy_fingerprint": policy_fingerprint,
        },
        started_at=10.0,
        ended_at=11.0,
        success=success,
        terminal_state="DRY_RUN",
        result=result,
        human_intervention=intervention,
        episode_id=episode_id,
    )


def _spec(
    tmp_path: Path,
    *,
    required_evidence_kinds: tuple[str, ...] = (),
    **gate_overrides,
) -> BenchmarkSpec:
    gates = {
        "min_episodes": 4,
        "min_episodes_per_value": 2,
        "min_success_rate": 0.75,
        "max_intervention_rate": 0.0,
        "max_failure_stage_rates": {"task_backend_execute": 0.25},
    }
    gates.update(gate_overrides)
    path = tmp_path / "benchmark.yaml"
    raw = {
                "version": 2 if required_evidence_kinds else 1,
                "benchmark_id": "fold-household-v1",
                "task_id": "fold_simple",
                "task_fingerprint": TASK_FINGERPRINT,
                "binding_equals": {"policy_fingerprint": POLICY_FINGERPRINT},
                "coverage": {
                    "lighting": ["bright", "dim"],
                    "layout": ["layout-a", "layout-b"],
                },
                "gates": gates,
            }
    if required_evidence_kinds:
        raw["required_evidence_kinds"] = list(required_evidence_kinds)
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return BenchmarkSpec.load(path)


def _passing_episodes(tmp_path: Path) -> None:
    _record(tmp_path, "episode-1", "bright", "layout-a", success=True)
    _record(tmp_path, "episode-2", "dim", "layout-a", success=True)
    _record(tmp_path, "episode-3", "bright", "layout-b", success=True)
    _record(tmp_path, "episode-4", "dim", "layout-b", success=False)


def _index_evidence(tmp_path: Path, episode_id: str, payload: bytes = b"rgb") -> str:
    source = tmp_path / f"{episode_id}.mp4"
    source.write_bytes(payload)
    spec_path = tmp_path / f"{episode_id}-evidence.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "approved": True,
                "episode_id": episode_id,
                "max_total_bytes": 1024,
                "max_time_margin_seconds": 0.0,
                "artifacts": [
                    {
                        "artifact_id": "head_rgb",
                        "kind": "rgb_video",
                        "source": str(source),
                        "mime_type": "video/mp4",
                        "captured_at_start": 10.0,
                        "captured_at_end": 11.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return index_episode_evidence(
        tmp_path / "episodes" / episode_id,
        EvidenceIndexSpec.load(spec_path),
        created_at=20.0,
    ).evidence_sha256


def test_benchmark_filters_exact_candidate_and_passes_all_gates(tmp_path: Path):
    _passing_episodes(tmp_path)
    _record(
        tmp_path,
        "episode-other-policy",
        "bright",
        "layout-a",
        success=False,
        policy_fingerprint="c" * 64,
    )
    report = evaluate_benchmark(_spec(tmp_path), load_episodes(tmp_path / "episodes"))
    assert report.passed is True
    assert len(report.episode_ids) == 4
    assert report.success_rate == 0.75
    assert report.failure_stage_rates == {"task_backend_execute": 0.25}
    assert report.coverage_counts["lighting"] == {"bright": 2, "dim": 2}
    assert len(report.episode_set_digest) == 64


def test_benchmark_reports_coverage_quality_and_intervention_failures(tmp_path: Path):
    _record(
        tmp_path,
        "episode-1",
        "bright",
        "layout-a",
        success=False,
        intervention=True,
    )
    report = evaluate_benchmark(_spec(tmp_path), load_episodes(tmp_path / "episodes"))
    assert report.passed is False
    assert any("episode count" in failure for failure in report.failures)
    assert any("success rate" in failure for failure in report.failures)
    assert any("intervention rate" in failure for failure in report.failures)
    assert any("coverage lighting=dim" in failure for failure in report.failures)


def test_report_is_atomic_hashed_and_candidate_bound(tmp_path: Path):
    _passing_episodes(tmp_path)
    report = evaluate_benchmark(_spec(tmp_path), load_episodes(tmp_path / "episodes"))
    path = tmp_path / "reports" / "fold.json"
    write_report(path, report)
    loaded = load_report(path)
    assert loaded == report
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    loaded.require_candidate(
        task_id="fold_simple",
        task_fingerprint=TASK_FINGERPRINT,
        execution_binding={"policy_fingerprint": POLICY_FINGERPRINT},
        now=loaded.generated_at + 10,
        max_age_seconds=20,
    )
    with pytest.raises(BenchmarkError, match="binding key"):
        loaded.require_candidate(
            task_id="fold_simple",
            task_fingerprint=TASK_FINGERPRINT,
            execution_binding={"policy_fingerprint": "c" * 64},
        )
    with pytest.raises(BenchmarkError, match="stale"):
        loaded.require_candidate(
            task_id="fold_simple",
            task_fingerprint=TASK_FINGERPRINT,
            execution_binding={"policy_fingerprint": POLICY_FINGERPRINT},
            now=loaded.generated_at + 21,
            max_age_seconds=20,
        )


def test_report_tampering_and_invalid_specs_are_rejected(tmp_path: Path):
    _passing_episodes(tmp_path)
    report = evaluate_benchmark(_spec(tmp_path), load_episodes(tmp_path / "episodes"))
    path = tmp_path / "report.json"
    write_report(path, report)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["success_rate"] = 1.0
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(BenchmarkError, match="hash does not match"):
        load_report(path)

    spec_path = tmp_path / "invalid.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "benchmark_id": "bad",
                "task_id": "fold_simple",
                "task_fingerprint": TASK_FINGERPRINT,
                "binding_equals": {"policy_fingerprint": POLICY_FINGERPRINT},
                "coverage": {"lighting": ["bright", "bright"]},
                "gates": {
                    "min_episodes": 1,
                    "min_episodes_per_value": 1,
                    "min_success_rate": 1.0,
                    "max_intervention_rate": 0.0,
                    "max_failure_stage_rates": {},
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(BenchmarkError, match="duplicate"):
        BenchmarkSpec.load(spec_path)


def test_evidence_bound_benchmark_requires_only_matching_episodes(tmp_path: Path):
    _record(tmp_path, "episode-selected", "bright", "layout-a", success=True)
    _record(
        tmp_path,
        "episode-other-policy",
        "bright",
        "layout-a",
        success=False,
        policy_fingerprint="c" * 64,
    )
    spec = _spec(
        tmp_path,
        required_evidence_kinds=("rgb_video",),
        min_episodes=1,
        min_episodes_per_value=1,
        min_success_rate=1.0,
    )
    with pytest.raises(BenchmarkError, match="episode-selected evidence is invalid"):
        evaluate_benchmark_store(spec, tmp_path / "episodes")

    evidence_sha = _index_evidence(tmp_path, "episode-selected")
    report = evaluate_benchmark_store(spec, tmp_path / "episodes")
    assert report.episode_ids == ("episode-selected",)
    assert report.required_evidence_kinds == ("rgb_video",)
    assert evidence_sha not in report.to_dict().values()


def test_evidence_digest_changes_episode_set_identity(tmp_path: Path):
    _record(tmp_path, "episode-selected", "bright", "layout-a", success=True)
    episodes = load_episodes(tmp_path / "episodes")
    spec = _spec(
        tmp_path,
        required_evidence_kinds=("rgb_video",),
        min_episodes=1,
        min_episodes_per_value=1,
        min_success_rate=1.0,
    )
    first = evaluate_benchmark(
        spec,
        episodes,
        evidence_sha256_by_episode={"episode-selected": "1" * 64},
    )
    second = evaluate_benchmark(
        spec,
        episodes,
        evidence_sha256_by_episode={"episode-selected": "2" * 64},
    )
    assert first.episode_set_digest != second.episode_set_digest


def test_legacy_v1_report_hash_remains_loadable(tmp_path: Path):
    _passing_episodes(tmp_path)
    report = evaluate_benchmark(_spec(tmp_path), load_episodes(tmp_path / "episodes"))
    raw = report.to_dict()
    raw["schema_version"] = 1
    raw.pop("required_evidence_kinds")
    raw.pop("report_sha256")
    encoded = json.dumps(
        raw,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    raw["report_sha256"] = hashlib.sha256(encoded).hexdigest()
    path = tmp_path / "legacy-report.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    loaded = load_report(path)
    assert loaded.schema_version == 1
    assert loaded.required_evidence_kinds == ()
