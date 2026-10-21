from __future__ import annotations

import hashlib
import stat
from pathlib import Path

import pytest
import yaml

from robot_agent.policies import PolicyRegistry, PolicyRegistryError
from robot_agent.policy_materialize import main, materialize_registry


def _template(tmp_path: Path) -> Path:
    executable = tmp_path / "runtime" / "bin" / "lerobot-record"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    runtime_manifest = tmp_path / "runtime" / "requirements.lock"
    runtime_manifest.write_text("lerobot==0.3.3\n", encoding="utf-8")
    model_path = tmp_path / "models" / "fold"
    model_path.mkdir(parents=True)
    (model_path / "model.safetensors").write_bytes(b"reviewed-model")
    raw = {
        "version": 1,
        "robot": {
            "type": "bi_so100_follower",
            "id": "robot0",
            "left_arm_port": "/dev/ttyACM0",
            "right_arm_port": "/dev/ttyACM1",
            "dataset_owner": "local",
            "cameras": {
                name: {
                    "type": "opencv",
                    "device": device,
                    "width": 640,
                    "height": 480,
                    "fps": 15,
                }
                for name, device in (
                    ("head_cam", 2),
                    ("left_cam", 0),
                    ("right_cam", 4),
                )
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
                "executable_sha256": "0" * 64,
                "workspace": str(tmp_path),
                "runtime_manifest": str(runtime_manifest),
                "runtime_sha256": "0" * 64,
                "model_path": str(model_path),
                "model_file": "model.safetensors",
                "model_sha256": "0" * 64,
                "timeout_seconds": 120,
                "enabled": False,
            }
        },
    }
    path = tmp_path / "policies.template.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_materialize_hashes_explicit_assets_and_can_enable_reviewed_task(tmp_path: Path):
    template = _template(tmp_path)
    output = tmp_path / "sealed" / "policies.yaml"

    result = materialize_registry(
        template,
        output,
        enabled_tasks=frozenset({"fold_simple"}),
    )

    registry = PolicyRegistry.load(output)
    policy = registry.resolve("fold_simple")
    assert result["enabled_tasks"] == ["fold_simple"]
    assert policy.model_sha256 == hashlib.sha256(b"reviewed-model").hexdigest()
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_materialize_defaults_to_disabled_and_rejects_unknown_enable(tmp_path: Path):
    template = _template(tmp_path)
    output = tmp_path / "policies.yaml"
    materialize_registry(template, output)

    with pytest.raises(PolicyRegistryError, match="disabled"):
        PolicyRegistry.load(output).resolve("fold_simple")

    unknown_output = tmp_path / "unknown.yaml"
    with pytest.raises(PolicyRegistryError, match="unknown tasks"):
        materialize_registry(
            template,
            unknown_output,
            enabled_tasks=frozenset({"missing_task"}),
        )
    assert not unknown_output.exists()


def test_materialize_cli_reports_missing_assets(tmp_path: Path, capsys):
    template = _template(tmp_path)
    raw = yaml.safe_load(template.read_text(encoding="utf-8"))
    raw["policies"]["fold-v1"]["model_path"] = str(tmp_path / "missing")
    template.write_text(yaml.safe_dump(raw), encoding="utf-8")

    exit_code = main([str(template), str(tmp_path / "output.yaml")])

    assert exit_code == 2
    assert "does not exist" in capsys.readouterr().out
