from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from robot_agent.benchmarks import BenchmarkSpec, evaluate_benchmark, write_report
from robot_agent.config import RuntimeConfig
from robot_agent.episodes import EpisodeContext, EpisodeRecorder, load_episodes
from robot_agent.hardware.authorization import (
    _make_robotd_hardware_authorization,
    _robotd_authorized_context,
)
from robot_agent.hardware.lerobot import LeRobotClient
from robot_agent.policies import PolicyRegistry, PolicyRegistryError
from robot_agent.service import RobotRuntimeServer
from robot_agent.task_catalog import TaskCatalog


def write_registry(tmp_path: Path, *, enabled: bool = True) -> Path:
    executable = tmp_path / "bin" / "lerobot-record"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    executable_digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime_manifest = workspace / "requirements.lock"
    runtime_manifest.write_text("lerobot==0.3.3\n", encoding="utf-8")
    runtime_digest = hashlib.sha256(runtime_manifest.read_bytes()).hexdigest()
    model = tmp_path / "models" / "fold-v1"
    model.mkdir(parents=True)
    artifact = model / "model.safetensors"
    artifact.write_bytes(b"verified-model")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    raw = {
        "version": 1,
        "robot": {
            "type": "bi_so100_follower",
            "id": "robot0",
            "left_arm_port": "/dev/ttyACM0",
            "right_arm_port": "/dev/ttyACM1",
            "dataset_owner": "local",
            "cameras": {
                "head_cam": {
                    "type": "opencv",
                    "device": 4,
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                },
                "left_cam": {
                    "type": "opencv",
                    "device": 0,
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                },
                "right_cam": {
                    "type": "opencv",
                    "device": 2,
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                },
            },
        },
        "task_defaults": {"fold_simple": "fold-v1"},
        "policies": {
            "fold-v1": {
                "task_id": "fold_simple",
                "family": "smolvla",
                "version": "1",
                "description": "fold the flattened rag",
                "executable": str(executable),
                "executable_sha256": executable_digest,
                "workspace": str(workspace),
                "runtime_manifest": str(runtime_manifest),
                "runtime_sha256": runtime_digest,
                "model_path": str(model),
                "model_file": "model.safetensors",
                "model_sha256": digest,
                "timeout_seconds": 120,
                "enabled": enabled,
            }
        },
    }
    path = tmp_path / "policies.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_registry_resolves_verified_policy_and_builds_argv(tmp_path: Path):
    registry = PolicyRegistry.load(write_registry(tmp_path))
    policy = registry.resolve("fold_simple")
    command = registry.build_command(policy, run_id="test run")
    binding = registry.resolve_binding("fold_simple")
    assert binding["policy_id"] == "fold-v1"
    assert len(binding["policy_fingerprint"]) == 64
    assert command[0].endswith("lerobot-record")
    assert "--policy.path=" + str(policy.model_path) in command
    assert "--dataset.repo_id=local/eval_fold_simple_test_run" in command
    assert not any("source " in argument or "conda" in argument for argument in command)


def test_registry_rejects_disabled_or_wrong_task_policy(tmp_path: Path):
    registry = PolicyRegistry.load(write_registry(tmp_path, enabled=False))
    with pytest.raises(PolicyRegistryError, match="disabled"):
        registry.resolve("fold_simple")
    with pytest.raises(PolicyRegistryError, match="no default"):
        registry.resolve("stack_contrast")


def test_registry_detects_artifact_change(tmp_path: Path):
    registry = PolicyRegistry.load(write_registry(tmp_path))
    policy = registry.resolve("fold_simple")
    artifact = policy.model_path / policy.model_file
    artifact.write_bytes(b"tampered-model")
    os.utime(artifact, None)
    with pytest.raises(PolicyRegistryError, match="SHA-256 mismatch"):
        registry.resolve("fold_simple")


def test_registry_rejects_reserved_argument_override(tmp_path: Path):
    path = write_registry(tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["policies"]["fold-v1"]["extra_args"] = ["--policy.path=/tmp/other"]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(PolicyRegistryError, match="reserved"):
        PolicyRegistry.load(path)


def test_binding_rejects_policy_change_after_authorization(tmp_path: Path):
    registry = PolicyRegistry.load(write_registry(tmp_path))
    binding = registry.resolve_binding("fold_simple")
    binding["policy_fingerprint"] = "0" * 64
    with pytest.raises(PolicyRegistryError, match="changed after authorization"):
        registry.validate_binding("fold_simple", binding)


def test_binding_covers_robot_and_camera_configuration(tmp_path: Path):
    path = write_registry(tmp_path)
    original = PolicyRegistry.load(path)
    binding = original.resolve_binding("fold_simple")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["robot"]["cameras"]["head_cam"]["device"] = 5
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    changed = PolicyRegistry.load(path)
    with pytest.raises(PolicyRegistryError, match="changed after authorization"):
        changed.validate_binding("fold_simple", binding)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda cameras: cameras.pop("right_cam"), "exactly head_cam"),
        (
            lambda cameras: cameras["head_cam"].update({"depth_topic": "/depth"}),
            "must contain only",
        ),
        (
            lambda cameras: cameras["head_cam"].update({"type": "realsense"}),
            "opencv RGB backend",
        ),
    ],
)
def test_registry_rejects_non_rgb_camera_assumptions(
    tmp_path: Path,
    mutation,
    message: str,
):
    path = write_registry(tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutation(raw["robot"]["cameras"])
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(PolicyRegistryError, match=message):
        PolicyRegistry.load(path)


def test_registry_public_view_does_not_expose_device_or_model_paths(tmp_path: Path):
    registry = PolicyRegistry.load(write_registry(tmp_path))
    public = registry.public_view()
    serialized = str(public)
    assert "/dev/" not in serialized
    assert str(tmp_path) not in serialized


def test_lerobot_executes_only_robotd_bound_verified_policy(tmp_path: Path):
    registry = PolicyRegistry.load(write_registry(tmp_path))
    binding = registry.resolve_binding("fold_simple")
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="done", stderr="")

    client = LeRobotClient(registry=registry, process_runner=runner)
    authorization = _make_robotd_hardware_authorization(
        "secret",
        "fold_simple",
        binding,
    )
    with _robotd_authorized_context(authorization):
        result = client.execute_task("fold_simple")

    assert result["success"] is True
    assert result["policy_id"] == "fold-v1"
    assert calls[0][0][0].endswith("lerobot-record")
    assert "shell" not in calls[0][1]
    assert calls[0][1]["cwd"] == registry.policies["fold-v1"].workspace


def test_robotd_service_arms_only_registry_backed_policy(tmp_path: Path):
    registry_path = write_registry(tmp_path)
    config = RuntimeConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
        socket_path=tmp_path / "run" / "robotd.sock",
        token_store_path=tmp_path / "state" / "tokens.json",
        audit_log_path=tmp_path / "state" / "audit.jsonl",
        world_model_path=tmp_path / "state" / "world.yaml",
        token_cache_path=tmp_path / "state" / "token.json",
        require_fresh_observation=False,
        observation_ttl_seconds=30,
        shutdown_grace_seconds=0.2,
        emergency_stop_command=(),
        policy_registry_path=registry_path,
    )
    server = RobotRuntimeServer(config)
    try:
        capabilities = server.dispatch({"command": "capabilities", "params": {}})
        armed = server.dispatch(
            {
                "command": "arm",
                "params": {
                    "task_id": "fold_simple",
                    "operator_confirmation": "CONFIRM fold_simple",
                },
            }
        )
    finally:
        server.server_close()
    assert capabilities["policy_registry"]["task_defaults"]["fold_simple"] == "fold-v1"
    assert armed["success"] is True
    assert armed["data"]["execution_binding"]["policy_id"] == "fold-v1"
    assert len(armed["data"]["execution_binding"]["task_fingerprint"]) == 64
    assert armed["data"]["execution_binding"]["task_executor"] == "learned_policy"


def test_example_policy_defaults_only_reference_learned_tasks():
    raw = yaml.safe_load(Path("config/policies.example.yaml").read_text(encoding="utf-8"))
    catalog = TaskCatalog.default()
    defaults = raw["task_defaults"]
    policies = raw["policies"]

    assert set(defaults) == {
        "fold_simple",
        "stack_contrast",
        "pick_up_bottle_towel",
        "pick_up_bottle",
    }
    for task_id, policy_id in defaults.items():
        task = catalog.get(task_id)
        assert task is not None
        assert task.executor == "learned_policy"
        assert policies[policy_id]["task_id"] == task_id


def _passing_benchmark_report(
    tmp_path: Path,
    registry_path: Path,
    *,
    evidence_bound: bool = False,
) -> Path:
    registry = PolicyRegistry.load(registry_path)
    task = TaskCatalog.default().get("fold_simple")
    binding = {
        "task_fingerprint": task.fingerprint(),
        "task_executor": "learned_policy",
        **registry.resolve_binding("fold_simple"),
    }
    context_path = tmp_path / "episode-context.yaml"
    context_path.write_text(
        "version: 1\ndeployment_id: test-lab\ntags:\n  lighting: bright\n",
        encoding="utf-8",
    )
    EpisodeRecorder(
        tmp_path / "episodes",
        context=EpisodeContext.load(context_path),
    ).record(
        task_id="fold_simple",
        execution_binding=binding,
        started_at=10.0,
        ended_at=11.0,
        success=True,
        terminal_state="DRY_RUN",
        result={"steps": []},
        episode_id="episode-policy-gate",
    )
    spec_path = tmp_path / "benchmark.yaml"
    raw = {
                "version": 2 if evidence_bound else 1,
                "benchmark_id": "fold-gate-v1",
                "task_id": "fold_simple",
                "task_fingerprint": task.fingerprint(),
                "binding_equals": {
                    "policy_fingerprint": binding["policy_fingerprint"]
                },
                "coverage": {"lighting": ["bright"]},
                "gates": {
                    "min_episodes": 1,
                    "min_episodes_per_value": 1,
                    "min_success_rate": 1.0,
                    "max_intervention_rate": 0.0,
                    "max_failure_stage_rates": {},
                },
            }
    if evidence_bound:
        raw["required_evidence_kinds"] = ["rgb_video", "robot_state"]
    spec_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    report = evaluate_benchmark(
        BenchmarkSpec.load(spec_path),
        load_episodes(tmp_path / "episodes"),
        evidence_sha256_by_episode=(
            {"episode-policy-gate": "e" * 64} if evidence_bound else None
        ),
    )
    report_path = tmp_path / "benchmark-report.json"
    write_report(report_path, report)
    return report_path


def test_learned_policy_arm_requires_matching_fresh_benchmark_when_enabled(tmp_path: Path):
    registry_path = write_registry(tmp_path)
    report_path = _passing_benchmark_report(tmp_path, registry_path)
    base = RuntimeConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
        socket_path=tmp_path / "run" / "robotd.sock",
        token_store_path=tmp_path / "state" / "tokens.json",
        audit_log_path=tmp_path / "state" / "audit.jsonl",
        world_model_path=tmp_path / "state" / "world.yaml",
        token_cache_path=tmp_path / "state" / "token.json",
        require_fresh_observation=False,
        observation_ttl_seconds=30,
        shutdown_grace_seconds=0.2,
        emergency_stop_command=(),
        policy_registry_path=registry_path,
        require_policy_benchmark=True,
    )
    missing_server = RobotRuntimeServer(base)
    try:
        missing = missing_server.dispatch(
            {
                "command": "arm",
                "params": {
                    "task_id": "fold_simple",
                    "operator_confirmation": "CONFIRM fold_simple",
                },
            }
        )
    finally:
        missing_server.server_close()
    assert missing["success"] is False
    assert "BENCHMARK_REPORT" in missing["message"]

    enabled = replace(
        base,
        socket_path=tmp_path / "run" / "robotd-gated.sock",
        token_store_path=tmp_path / "state" / "tokens-gated.json",
        benchmark_report_path=report_path.parent,
    )
    server = RobotRuntimeServer(enabled)
    try:
        armed = server.dispatch(
            {
                "command": "arm",
                "params": {
                    "task_id": "fold_simple",
                    "operator_confirmation": "CONFIRM fold_simple",
                },
            }
        )
    finally:
        server.server_close()
    binding = armed["data"]["execution_binding"]
    assert armed["success"] is True
    assert binding["benchmark_id"] == "fold-gate-v1"
    assert len(binding["benchmark_report_sha256"]) == 64


def test_learned_policy_evidence_gate_rejects_legacy_and_binds_v2(tmp_path: Path):
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    legacy_registry = write_registry(legacy_root)
    legacy_report = _passing_benchmark_report(legacy_root, legacy_registry)
    base = RuntimeConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
        socket_path=tmp_path / "run" / "robotd.sock",
        token_store_path=tmp_path / "state" / "tokens.json",
        audit_log_path=tmp_path / "state" / "audit.jsonl",
        world_model_path=tmp_path / "state" / "world.yaml",
        token_cache_path=tmp_path / "state" / "token.json",
        require_fresh_observation=False,
        observation_ttl_seconds=30,
        shutdown_grace_seconds=0.2,
        emergency_stop_command=(),
        policy_registry_path=legacy_registry,
        require_policy_benchmark=True,
        require_policy_benchmark_evidence=True,
        benchmark_report_path=legacy_report,
    )
    legacy_server = RobotRuntimeServer(base)
    try:
        rejected = legacy_server.dispatch(
            {
                "command": "arm",
                "params": {
                    "task_id": "fold_simple",
                    "operator_confirmation": "CONFIRM fold_simple",
                },
            }
        )
    finally:
        legacy_server.server_close()
    assert rejected["success"] is False
    assert "required episode evidence" in rejected["message"]

    bound_root = tmp_path / "bound"
    bound_root.mkdir()
    bound_registry = write_registry(bound_root)
    bound_report = _passing_benchmark_report(
        bound_root,
        bound_registry,
        evidence_bound=True,
    )
    enabled = replace(
        base,
        socket_path=tmp_path / "run" / "robotd-bound.sock",
        token_store_path=tmp_path / "state" / "tokens-bound.json",
        policy_registry_path=bound_registry,
        benchmark_report_path=bound_report,
    )
    server = RobotRuntimeServer(enabled)
    try:
        armed = server.dispatch(
            {
                "command": "arm",
                "params": {
                    "task_id": "fold_simple",
                    "operator_confirmation": "CONFIRM fold_simple",
                },
            }
        )
    finally:
        server.server_close()
    assert armed["success"] is True
    assert armed["data"]["execution_binding"]["benchmark_evidence_kinds"] == [
        "rgb_video",
        "robot_state",
    ]
