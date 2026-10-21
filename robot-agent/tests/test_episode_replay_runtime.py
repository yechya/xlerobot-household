from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest
import yaml

from robot_agent.episode_evidence import EvidenceIndexSpec, index_episode_evidence
from robot_agent.episode_replay import ReplaySpec, build_replay_plan, write_replay_plan
from robot_agent.episode_replay_runtime import (
    ReplayHandlerResult,
    ReplayProducedArtifact,
    ReplayRuntimeError,
    ReplayRuntimeSpec,
    create_evidence_export_handler,
    execute_replay,
    load_handlers,
    load_replay_run,
)
from robot_agent.episodes import EpisodeRecorder


class FakeDistribution:
    name = "xlerobot-agent"
    version = "0.1.0"
    metadata = {"Name": name}


class FakeEntryPoint:
    name = "evidence_export_v1"
    dist = FakeDistribution()

    @staticmethod
    def load():
        return create_evidence_export_handler


class FakeEntryPoints(list):
    def select(self, *, group: str, name: str):
        return [
            entry
            for entry in self
            if group == "xlerobot.replay_handlers" and entry.name == name
        ]


class FailingHandler:
    @staticmethod
    def execute(invocation):
        (invocation.output_dir / "diagnostic.log").write_bytes(b"bounded failure detail")
        raise RuntimeError("offline evaluator failed")


class FailingEntryPoint(FakeEntryPoint):
    @staticmethod
    def load():
        return lambda **_: FailingHandler()


class NamedFailingEntryPoint(FailingEntryPoint):
    name = "failing_v1"


class ProducerHandler:
    @staticmethod
    def execute(invocation):
        path = invocation.output_dir / "derived.jsonl"
        path.write_text('{"derived":true}\n', encoding="utf-8")
        return ReplayHandlerResult(
            metadata={"produced": True},
            produced_artifacts=(
                ReplayProducedArtifact(
                    artifact_id="derived_frames",
                    kind="derived_frames",
                    relative_path=path.name,
                    mime_type="application/jsonl",
                    captured_at_start=invocation.started_at,
                    captured_at_end=invocation.ended_at,
                ),
            ),
        )


class ConsumerHandler:
    @staticmethod
    def execute(invocation):
        derived = [
            artifact
            for artifact in invocation.artifacts
            if artifact.kind == "derived_frames"
        ]
        if len(derived) != 1:
            raise RuntimeError("validated derived artifact was not propagated")
        content = derived[0].source.read_text(encoding="utf-8")
        (invocation.output_dir / "consumed.txt").write_text(content, encoding="utf-8")
        return {"consumed_sha256": derived[0].sha256}


class ProducerEntryPoint(FakeEntryPoint):
    name = "producer_v1"

    @staticmethod
    def load():
        return lambda **_: ProducerHandler()


class ConsumerEntryPoint(FakeEntryPoint):
    name = "consumer_v1"

    @staticmethod
    def load():
        return lambda **_: ConsumerHandler()


class UnsafeProducerHandler:
    @staticmethod
    def execute(invocation):
        (invocation.output_dir / "diagnostic.jsonl").write_text("{}\n", encoding="utf-8")
        return ReplayHandlerResult(
            metadata={},
            produced_artifacts=(
                ReplayProducedArtifact(
                    artifact_id="unsafe_frames",
                    kind="derived_frames",
                    relative_path="../outside.jsonl",
                    mime_type="application/jsonl",
                    captured_at_start=invocation.started_at,
                    captured_at_end=invocation.ended_at,
                ),
            ),
        )


class UnsafeProducerEntryPoint(FakeEntryPoint):
    name = "unsafe_producer_v1"

    @staticmethod
    def load():
        return lambda **_: UnsafeProducerHandler()


def _record(root: Path, episode_id: str, stage: str = "scene_resolve") -> None:
    EpisodeRecorder(root).record(
        task_id="tidy_table",
        execution_binding={
            "task_fingerprint": "a" * 64,
            "task_executor": "primitive_graph",
        },
        started_at=10.0,
        ended_at=12.0,
        success=False,
        terminal_state="ERROR",
        result={
            "steps": [{"action": stage, "success": False}],
            "outcome_code": "FAILED_STAGE",
        },
        episode_id=episode_id,
    )


def _index(tmp_path: Path, episodes: Path, episode_id: str) -> None:
    rgb = tmp_path / f"{episode_id}.mp4"
    rgb.write_bytes(f"rgb-{episode_id}".encode())
    state = tmp_path / f"{episode_id}.mcap"
    state.write_bytes(f"state-{episode_id}".encode())
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
                        "source": str(rgb),
                        "mime_type": "video/mp4",
                        "captured_at_start": 10.0,
                        "captured_at_end": 12.0,
                    },
                    {
                        "artifact_id": "robot_state",
                        "kind": "robot_state",
                        "source": str(state),
                        "mime_type": "application/mcap",
                        "captured_at_start": 10.0,
                        "captured_at_end": 12.0,
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    index_episode_evidence(
        episodes / episode_id,
        EvidenceIndexSpec.load(spec_path),
        created_at=20.0,
    )


def _plan(tmp_path: Path, episodes: Path) -> Path:
    spec_path = tmp_path / "replay-plan.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
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
                    }
                ],
                "default_route": "system_review",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "replay-plan.json"
    write_replay_plan(output, build_replay_plan(ReplaySpec.load(spec_path), episodes))
    return output


def _runtime_config(
    tmp_path: Path,
    episodes: Path,
    plan: Path,
    *,
    routes: dict[str, object] | None = None,
    failure_policy: str = "continue",
    distribution: str = "xlerobot-agent",
    version: int = 1,
) -> Path:
    config = tmp_path / "replay-runtime.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "version": version,
                "approved": True,
                "runtime_id": "triage-run-v1",
                "plan": {
                    "path": str(plan),
                    "sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
                },
                "episode_root": str(episodes),
                "output_root": str(tmp_path / "replay-runs"),
                "max_entries": 100,
                "max_output_bytes_per_entry": 1024,
                "failure_policy": failure_policy,
                "handlers": {
                    "evidence_export": {
                        "entry_point": "evidence_export_v1",
                        "distribution": distribution,
                        "distribution_version": "0.1.0",
                        "artifacts": [],
                        "options": {
                            "include_kinds": ["rgb_video", "robot_state"],
                            "preserve_extension": True,
                        },
                    }
                },
                "routes": routes or {"perception_fix": "evidence_export"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config


def test_runtime_exports_bound_evidence_and_verifies_immutable_report(tmp_path: Path):
    episodes = tmp_path / "episodes"
    _record(episodes, "episode-one")
    _index(tmp_path, episodes, "episode-one")
    plan = _plan(tmp_path, episodes)
    spec = ReplayRuntimeSpec.load(_runtime_config(tmp_path, episodes, plan))

    output = execute_replay(
        spec,
        entry_points=FakeEntryPoints([FakeEntryPoint()]),
        generated_at=30.0,
    )
    report = load_replay_run(output)
    assert report["schema_version"] == 3
    assert report["passed"] is True
    assert report["counts"] == {"failed": 0, "skipped": 0, "succeeded": 1}
    assert report["entries"][0]["handler_metadata"] == {
        "exported_artifact_ids": ["head_rgb", "robot_state"],
        "exported_kinds": ["rgb_video", "robot_state"],
    }
    for artifact in report["entries"][0]["outputs"]:
        path = output / artifact["path"]
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    assert stat.S_IMODE((output / "report.json").stat().st_mode) == 0o600
    with pytest.raises(ReplayRuntimeError, match="already exists"):
        execute_replay(spec, entry_points=FakeEntryPoints([FakeEntryPoint()]))


def test_runtime_loader_preserves_legacy_v1_single_handler_reports(tmp_path: Path):
    episodes = tmp_path / "episodes"
    _record(episodes, "episode-one")
    _index(tmp_path, episodes, "episode-one")
    plan = _plan(tmp_path, episodes)
    spec = ReplayRuntimeSpec.load(_runtime_config(tmp_path, episodes, plan))
    output = execute_replay(
        spec,
        entry_points=FakeEntryPoints([FakeEntryPoint()]),
        generated_at=30.0,
    )
    raw = json.loads((output / "report.json").read_text(encoding="utf-8"))
    raw["schema_version"] = 1
    handler_dir = output / "entries" / "episode-one" / "evidence_export"
    for result_file in handler_dir.iterdir():
        result_file.rename(handler_dir.parent / result_file.name)
    handler_dir.rmdir()
    for result in raw["entries"]:
        result.pop("produced_artifacts")
        for artifact in result["outputs"]:
            artifact["path"] = artifact["path"].replace("/evidence_export/", "/")
    raw.pop("report_sha256")
    encoded = json.dumps(
        raw,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    raw["report_sha256"] = hashlib.sha256(encoded).hexdigest()
    (output / "report.json").write_text(json.dumps(raw), encoding="utf-8")

    loaded = load_replay_run(output)
    assert loaded["schema_version"] == 1


def test_runtime_rejects_plugin_identity_and_incomplete_route_mapping(tmp_path: Path):
    episodes = tmp_path / "episodes"
    _record(episodes, "episode-one")
    _index(tmp_path, episodes, "episode-one")
    plan = _plan(tmp_path, episodes)
    wrong_identity = ReplayRuntimeSpec.load(
        _runtime_config(tmp_path, episodes, plan, distribution="other-package")
    )
    with pytest.raises(ReplayRuntimeError, match="distribution does not match"):
        load_handlers(wrong_identity, entry_points=FakeEntryPoints([FakeEntryPoint()]))

    incomplete = ReplayRuntimeSpec.load(
        _runtime_config(tmp_path, episodes, plan, routes={"other_route": "evidence_export"})
    )
    with pytest.raises(ReplayRuntimeError, match="exactly cover"):
        incomplete.validate_bindings()


def test_runtime_records_tampered_evidence_as_failed_result(tmp_path: Path):
    episodes = tmp_path / "episodes"
    _record(episodes, "episode-one")
    _index(tmp_path, episodes, "episode-one")
    plan = _plan(tmp_path, episodes)
    source = episodes / "episode-one" / "evidence" / "files" / "head_rgb.mp4"
    source.write_bytes(b"tampered")
    spec = ReplayRuntimeSpec.load(_runtime_config(tmp_path, episodes, plan))

    output = execute_replay(
        spec,
        entry_points=FakeEntryPoints([FakeEntryPoint()]),
        generated_at=30.0,
    )
    report = load_replay_run(output)
    assert report["passed"] is False
    assert report["counts"] == {"failed": 1, "skipped": 0, "succeeded": 0}
    assert "evidence is invalid" in report["entries"][0]["error"]


def test_runtime_loader_rejects_export_tampering(tmp_path: Path):
    episodes = tmp_path / "episodes"
    _record(episodes, "episode-one")
    _index(tmp_path, episodes, "episode-one")
    plan = _plan(tmp_path, episodes)
    spec = ReplayRuntimeSpec.load(_runtime_config(tmp_path, episodes, plan))
    output = execute_replay(
        spec,
        entry_points=FakeEntryPoints([FakeEntryPoint()]),
        generated_at=30.0,
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    exported = output / report["entries"][0]["outputs"][0]["path"]
    exported.write_bytes(b"tampered")
    with pytest.raises(ReplayRuntimeError, match="size changed|hash changed"):
        load_replay_run(output)


def test_runtime_stop_policy_marks_remaining_entries_skipped(tmp_path: Path):
    episodes = tmp_path / "episodes"
    for episode_id in ("episode-one", "episode-two"):
        _record(episodes, episode_id)
        _index(tmp_path, episodes, episode_id)
    plan = _plan(tmp_path, episodes)
    source = episodes / "episode-one" / "evidence" / "files" / "head_rgb.mp4"
    source.write_bytes(b"tampered")
    spec = ReplayRuntimeSpec.load(
        _runtime_config(tmp_path, episodes, plan, failure_policy="stop")
    )

    output = execute_replay(
        spec,
        entry_points=FakeEntryPoints([FakeEntryPoint()]),
        generated_at=30.0,
    )
    report = load_replay_run(output)
    assert report["counts"] == {"failed": 1, "skipped": 1, "succeeded": 0}
    assert [entry["status"] for entry in report["entries"]] == ["failed", "skipped"]


def test_runtime_preserves_bounded_failure_diagnostics(tmp_path: Path):
    episodes = tmp_path / "episodes"
    _record(episodes, "episode-one")
    _index(tmp_path, episodes, "episode-one")
    plan = _plan(tmp_path, episodes)
    spec = ReplayRuntimeSpec.load(_runtime_config(tmp_path, episodes, plan))

    output = execute_replay(
        spec,
        entry_points=FakeEntryPoints([FailingEntryPoint()]),
        generated_at=30.0,
    )
    report = load_replay_run(output)
    entry = report["entries"][0]
    assert entry["status"] == "failed"
    assert "offline evaluator failed" in entry["error"]
    assert len(entry["outputs"]) == 1
    assert (output / entry["outputs"][0]["path"]).read_bytes() == b"bounded failure detail"


def test_v2_route_runs_ordered_handlers_in_isolated_output_directories(tmp_path: Path):
    episodes = tmp_path / "episodes"
    _record(episodes, "episode-one")
    _index(tmp_path, episodes, "episode-one")
    plan = _plan(tmp_path, episodes)
    config = _runtime_config(
        tmp_path,
        episodes,
        plan,
        version=2,
        routes={"perception_fix": ["first_export", "second_export"]},
    )
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    first = raw["handlers"].pop("evidence_export")
    first["options"]["include_kinds"] = ["rgb_video"]
    second = json.loads(json.dumps(first))
    second["options"]["include_kinds"] = ["robot_state"]
    raw["handlers"] = {"first_export": first, "second_export": second}
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")

    spec = ReplayRuntimeSpec.load(config)
    assert spec.routes == {"perception_fix": ("first_export", "second_export")}
    output = execute_replay(
        spec,
        entry_points=FakeEntryPoints([FakeEntryPoint()]),
        generated_at=30.0,
    )
    report = load_replay_run(output)
    assert report["counts"] == {"failed": 0, "skipped": 0, "succeeded": 2}
    assert [entry["handler_id"] for entry in report["entries"]] == [
        "first_export",
        "second_export",
    ]
    assert all(
        f"/episode-one/{entry['handler_id']}/" in entry["outputs"][0]["path"]
        for entry in report["entries"]
    )


@pytest.mark.parametrize(
    ("failure_policy", "expected_statuses"),
    [
        ("continue", ["failed", "succeeded"]),
        ("stop", ["failed", "skipped"]),
    ],
)
def test_v2_pipeline_failure_policy_controls_following_handlers(
    tmp_path: Path,
    failure_policy: str,
    expected_statuses: list[str],
):
    episodes = tmp_path / "episodes"
    _record(episodes, "episode-one")
    _index(tmp_path, episodes, "episode-one")
    plan = _plan(tmp_path, episodes)
    config = _runtime_config(
        tmp_path,
        episodes,
        plan,
        version=2,
        failure_policy=failure_policy,
        routes={"perception_fix": ["diagnose", "export_after"]},
    )
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    export = raw["handlers"].pop("evidence_export")
    diagnose = json.loads(json.dumps(export))
    diagnose["entry_point"] = "failing_v1"
    raw["handlers"] = {"diagnose": diagnose, "export_after": export}
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")

    output = execute_replay(
        ReplayRuntimeSpec.load(config),
        entry_points=FakeEntryPoints(
            [NamedFailingEntryPoint(), FakeEntryPoint()]
        ),
        generated_at=30.0,
    )
    report = load_replay_run(output)
    assert [entry["status"] for entry in report["entries"]] == expected_statuses


def test_v3_pipeline_propagates_only_validated_derived_artifacts(tmp_path: Path):
    episodes = tmp_path / "episodes"
    _record(episodes, "episode-one")
    _index(tmp_path, episodes, "episode-one")
    plan = _plan(tmp_path, episodes)
    config = _runtime_config(
        tmp_path,
        episodes,
        plan,
        version=2,
        routes={"perception_fix": ["producer", "consumer"]},
    )
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    template = raw["handlers"].pop("evidence_export")
    producer = json.loads(json.dumps(template))
    producer["entry_point"] = "producer_v1"
    consumer = json.loads(json.dumps(template))
    consumer["entry_point"] = "consumer_v1"
    raw["handlers"] = {"producer": producer, "consumer": consumer}
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")

    output = execute_replay(
        ReplayRuntimeSpec.load(config),
        entry_points=FakeEntryPoints([ProducerEntryPoint(), ConsumerEntryPoint()]),
        generated_at=30.0,
    )
    report = load_replay_run(output)
    produced = report["entries"][0]["produced_artifacts"][0]
    consumed = report["entries"][1]
    assert produced["artifact_id"] == "derived_frames"
    assert produced["sha256"] == consumed["handler_metadata"]["consumed_sha256"]
    assert (output / consumed["outputs"][0]["path"]).read_text(encoding="utf-8") == (
        '{"derived":true}\n'
    )


def test_v3_pipeline_rejects_unsafe_derived_artifact_descriptor(tmp_path: Path):
    episodes = tmp_path / "episodes"
    _record(episodes, "episode-one")
    _index(tmp_path, episodes, "episode-one")
    plan = _plan(tmp_path, episodes)
    config = _runtime_config(
        tmp_path,
        episodes,
        plan,
        version=2,
        routes={"perception_fix": ["unsafe"]},
    )
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    handler = raw["handlers"].pop("evidence_export")
    handler["entry_point"] = "unsafe_producer_v1"
    raw["handlers"] = {"unsafe": handler}
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")

    output = execute_replay(
        ReplayRuntimeSpec.load(config),
        entry_points=FakeEntryPoints([UnsafeProducerEntryPoint()]),
        generated_at=30.0,
    )
    report = load_replay_run(output)
    result = report["entries"][0]
    assert result["status"] == "failed"
    assert "relative path is unsafe" in result["error"]
    assert result["produced_artifacts"] == []
