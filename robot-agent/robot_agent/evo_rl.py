"""Offline bridge between robot-agent episode manifests and Evo-RL datasets.

The bridge intentionally exports an index rather than importing Evo-RL into the
robotd process. Evo-RL can then join this index with the referenced LeRobot
datasets before value training and advantage-conditioned policy training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .episodes import EpisodeError, EpisodeManifest, load_episodes

EVO_RL_BRIDGE_SCHEMA_VERSION = 1
EVO_RL_UPSTREAM_REPOSITORY = "https://github.com/MINT-SJTU/Evo-RL"
EVO_RL_UPSTREAM_COMMIT = "6f2db449a21e1bac750b996f2e27cac6739aa63f"
EVO_RL_CAMERA_ROLES = ("head_cam", "left_cam", "right_cam")

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class EvoRLBridgeError(ValueError):
    """Raised when an Evo-RL export cannot be made trustworthy."""


@dataclass(frozen=True)
class EvoRLEpisodeRecord:
    """One manifest-to-Evo-RL alignment record.

    Frame-level actions remain in the referenced LeRobot dataset. The manifest
    only supplies episode labels and provenance, so the bridge never fabricates
    a policy action or per-frame intervention mask.
    """

    episode_id: str
    manifest_sha256: str
    task_id: str
    task_fingerprint: str
    episode_success: str
    failure_stage: str
    outcome_code: str
    human_intervention: bool
    deployment_id: str
    policy_id: str
    policy_fingerprint: str
    dataset_refs: tuple[str, ...]
    tags: dict[str, str]

    @classmethod
    def from_manifest(cls, episode: EpisodeManifest) -> "EvoRLEpisodeRecord":
        binding = episode.execution_binding
        policy_id = binding.get("policy_id")
        policy_fingerprint = binding.get("policy_fingerprint")
        return cls(
            episode_id=episode.episode_id,
            manifest_sha256=episode.manifest_sha256,
            task_id=episode.task_id,
            task_fingerprint=episode.task_fingerprint,
            episode_success="success" if episode.success else "failure",
            failure_stage=episode.failure_stage,
            outcome_code=episode.outcome_code,
            human_intervention=episode.human_intervention,
            deployment_id=episode.deployment_id,
            policy_id=policy_id if isinstance(policy_id, str) else "",
            policy_fingerprint=(
                policy_fingerprint if isinstance(policy_fingerprint, str) else ""
            ),
            dataset_refs=tuple(episode.dataset_refs),
            tags=dict(episode.tags),
        )

    def validate(self) -> None:
        _identifier(self.episode_id, "episode_id")
        _sha(self.manifest_sha256, "manifest_sha256")
        _identifier(self.task_id, "task_id")
        _sha(self.task_fingerprint, "task_fingerprint")
        if self.episode_success not in {"success", "failure"}:
            raise EvoRLBridgeError("episode_success must be success or failure")
        if not isinstance(self.failure_stage, str) or len(self.failure_stage) > 128:
            raise EvoRLBridgeError("failure_stage is invalid")
        if not isinstance(self.outcome_code, str) or not 0 < len(self.outcome_code) <= 128:
            raise EvoRLBridgeError("outcome_code is invalid")
        if not isinstance(self.human_intervention, bool):
            raise EvoRLBridgeError("human_intervention must be a boolean")
        _identifier(self.deployment_id, "deployment_id")
        if self.policy_id:
            _identifier(self.policy_id, "policy_id")
        if self.policy_fingerprint:
            _sha(self.policy_fingerprint, "policy_fingerprint")
        if len(self.dataset_refs) > 64 or any(
            not isinstance(value, str) or not 0 < len(value) <= 256
            for value in self.dataset_refs
        ):
            raise EvoRLBridgeError("dataset_refs are invalid")
        if len(self.tags) > 64 or any(
            not isinstance(key, str)
            or not _IDENTIFIER.fullmatch(key)
            or not isinstance(value, str)
            or not _IDENTIFIER.fullmatch(value)
            for key, value in self.tags.items()
        ):
            raise EvoRLBridgeError("tags are invalid")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        collector_policy_id = self.policy_id or (
            "human" if self.human_intervention else "unknown"
        )
        return {
            "episode_id": self.episode_id,
            "manifest_sha256": self.manifest_sha256,
            "task_id": self.task_id,
            "task_fingerprint": self.task_fingerprint,
            "episode_success": self.episode_success,
            "failure_stage": self.failure_stage,
            "outcome_code": self.outcome_code,
            "human_intervention": self.human_intervention,
            "deployment_id": self.deployment_id,
            "policy_id": self.policy_id,
            "policy_fingerprint": self.policy_fingerprint,
            "dataset_refs": list(self.dataset_refs),
            "tags": dict(self.tags),
            "complementary_info": {
                "is_intervention": int(self.human_intervention),
                "intervention_scope": "episode",
                "collector_policy_id": collector_policy_id,
                "policy_action_available": False,
            },
        }


@dataclass(frozen=True)
class EvoRLExport:
    """Signed, reproducible index for an Evo-RL offline training round."""

    bridge_id: str
    upstream_repository: str
    upstream_commit: str
    generated_at: float
    camera_roles: tuple[str, ...]
    episode_set_digest: str
    episodes: tuple[EvoRLEpisodeRecord, ...]
    export_sha256: str

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": EVO_RL_BRIDGE_SCHEMA_VERSION,
            "bridge_id": self.bridge_id,
            "upstream_repository": self.upstream_repository,
            "upstream_commit": self.upstream_commit,
            "generated_at": self.generated_at,
            "camera_roles": list(self.camera_roles),
            "episode_set_digest": self.episode_set_digest,
            "episodes": [episode.to_dict() for episode in self.episodes],
        }

    def validate(self) -> None:
        _identifier(self.bridge_id, "bridge_id")
        if self.upstream_repository != EVO_RL_UPSTREAM_REPOSITORY:
            raise EvoRLBridgeError("upstream_repository is not the reviewed Evo-RL repository")
        if not _GIT_COMMIT.fullmatch(self.upstream_commit):
            raise EvoRLBridgeError("upstream_commit must be a 40-character git commit")
        if not math.isfinite(self.generated_at) or self.generated_at <= 0:
            raise EvoRLBridgeError("generated_at is invalid")
        if self.camera_roles != EVO_RL_CAMERA_ROLES:
            raise EvoRLBridgeError("camera_roles must be exactly the three RGB roles")
        _sha(self.episode_set_digest, "episode_set_digest")
        if len(self.episodes) > 100_000:
            raise EvoRLBridgeError("export contains too many episodes")
        episode_ids = [episode.episode_id for episode in self.episodes]
        if len(episode_ids) != len(set(episode_ids)):
            raise EvoRLBridgeError("episode IDs must be unique")
        for episode in self.episodes:
            episode.validate()
        _sha(self.export_sha256, "export_sha256")
        if self.export_sha256 != _digest(self._hash_payload()):
            raise EvoRLBridgeError("export_sha256 does not match export content")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {**self._hash_payload(), "export_sha256": self.export_sha256}

    @classmethod
    def from_dict(cls, raw: Any) -> "EvoRLExport":
        expected = {
            "schema_version",
            "bridge_id",
            "upstream_repository",
            "upstream_commit",
            "generated_at",
            "camera_roles",
            "episode_set_digest",
            "episodes",
            "export_sha256",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise EvoRLBridgeError("Evo-RL export has missing or unexpected fields")
        if raw["schema_version"] != EVO_RL_BRIDGE_SCHEMA_VERSION:
            raise EvoRLBridgeError("unsupported Evo-RL bridge schema version")
        records = raw["episodes"]
        if not isinstance(records, list):
            raise EvoRLBridgeError("episodes must be a list")
        parsed = tuple(_episode_from_dict(value) for value in records)
        export = cls(
            bridge_id=raw["bridge_id"],
            upstream_repository=raw["upstream_repository"],
            upstream_commit=raw["upstream_commit"],
            generated_at=float(raw["generated_at"]),
            camera_roles=tuple(raw["camera_roles"]),
            episode_set_digest=raw["episode_set_digest"],
            episodes=parsed,
            export_sha256=raw["export_sha256"],
        )
        export.validate()
        return export


def build_evo_rl_export(
    episodes: Iterable[EpisodeManifest],
    *,
    bridge_id: str = "xlerobot-evo-rl-v1",
    generated_at: float | None = None,
    upstream_commit: str = EVO_RL_UPSTREAM_COMMIT,
) -> EvoRLExport:
    records = tuple(
        EvoRLEpisodeRecord.from_manifest(episode)
        for episode in sorted(episodes, key=lambda value: value.episode_id)
    )
    timestamp = time.time() if generated_at is None else generated_at
    if not math.isfinite(timestamp) or timestamp <= 0:
        raise EvoRLBridgeError("generated_at must be positive and finite")
    episode_set_digest = _digest(
        [
            {"episode_id": episode.episode_id, "manifest_sha256": episode.manifest_sha256}
            for episode in records
        ]
    )
    draft = EvoRLExport(
        bridge_id=bridge_id,
        upstream_repository=EVO_RL_UPSTREAM_REPOSITORY,
        upstream_commit=upstream_commit,
        generated_at=timestamp,
        camera_roles=EVO_RL_CAMERA_ROLES,
        episode_set_digest=episode_set_digest,
        episodes=records,
        export_sha256="0" * 64,
    )
    export = EvoRLExport(
        **{
            **draft.__dict__,
            "export_sha256": _digest(draft._hash_payload()),
        }
    )
    export.validate()
    return export


def write_evo_rl_export(path: str | Path, export: EvoRLExport) -> None:
    export.validate()
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination.parent.chmod(0o700)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(export.to_dict(), handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
        destination.chmod(0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_evo_rl_export(path: str | Path) -> EvoRLExport:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvoRLBridgeError(f"cannot read Evo-RL export: {exc}") from exc
    return EvoRLExport.from_dict(raw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export verified robot-agent episodes for an Evo-RL offline round"
    )
    parser.add_argument("episode_root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument("--executor", action="append", dest="executors")
    parser.add_argument("--bridge-id", default="xlerobot-evo-rl-v1")
    args = parser.parse_args()
    try:
        episodes = load_episodes(args.episode_root)
        task_ids = set(args.task_ids or ())
        executors = set(args.executors or ())
        if task_ids:
            episodes = [episode for episode in episodes if episode.task_id in task_ids]
        if executors:
            episodes = [episode for episode in episodes if episode.executor in executors]
        export = build_evo_rl_export(episodes, bridge_id=args.bridge_id)
        write_evo_rl_export(args.output, export)
        payload = {
            "valid": True,
            "output": str(Path(args.output).expanduser()),
            "episode_count": len(export.episodes),
            "episode_set_digest": export.episode_set_digest,
            "export_sha256": export.export_sha256,
            "camera_roles": list(export.camera_roles),
        }
        exit_code = 0
    except (OSError, EpisodeError, EvoRLBridgeError) as exc:
        payload = {"valid": False, "error": str(exc)}
        exit_code = 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(exit_code)


def _episode_from_dict(raw: Any) -> EvoRLEpisodeRecord:
    expected = {
        "episode_id",
        "manifest_sha256",
        "task_id",
        "task_fingerprint",
        "episode_success",
        "failure_stage",
        "outcome_code",
        "human_intervention",
        "deployment_id",
        "policy_id",
        "policy_fingerprint",
        "dataset_refs",
        "tags",
        "complementary_info",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise EvoRLBridgeError("Evo-RL episode record has missing or unexpected fields")
    complementary = raw["complementary_info"]
    if not isinstance(complementary, dict) or set(complementary) != {
        "is_intervention",
        "intervention_scope",
        "collector_policy_id",
        "policy_action_available",
    }:
        raise EvoRLBridgeError("Evo-RL complementary_info is invalid")
    record = EvoRLEpisodeRecord(
        episode_id=raw["episode_id"],
        manifest_sha256=raw["manifest_sha256"],
        task_id=raw["task_id"],
        task_fingerprint=raw["task_fingerprint"],
        episode_success=raw["episode_success"],
        failure_stage=raw["failure_stage"],
        outcome_code=raw["outcome_code"],
        human_intervention=raw["human_intervention"],
        deployment_id=raw["deployment_id"],
        policy_id=raw["policy_id"],
        policy_fingerprint=raw["policy_fingerprint"],
        dataset_refs=tuple(raw["dataset_refs"]),
        tags=dict(raw["tags"]),
    )
    expected_intervention = int(record.human_intervention)
    if complementary["is_intervention"] != expected_intervention:
        raise EvoRLBridgeError("complementary intervention flag disagrees with manifest")
    if complementary["intervention_scope"] != "episode":
        raise EvoRLBridgeError("unsupported intervention scope")
    if not isinstance(complementary["collector_policy_id"], str):
        raise EvoRLBridgeError("collector_policy_id must be a string")
    if complementary["policy_action_available"] is not False:
        raise EvoRLBridgeError("policy_action_available must remain false in the bridge index")
    record.validate()
    return record


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise EvoRLBridgeError(f"{label} is invalid")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise EvoRLBridgeError(f"{label} must be a SHA-256 digest")
    return value


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "EVO_RL_BRIDGE_SCHEMA_VERSION",
    "EVO_RL_CAMERA_ROLES",
    "EVO_RL_UPSTREAM_COMMIT",
    "EVO_RL_UPSTREAM_REPOSITORY",
    "EvoRLEpisodeRecord",
    "EvoRLExport",
    "EvoRLBridgeError",
    "build_evo_rl_export",
    "load_evo_rl_export",
    "main",
    "write_evo_rl_export",
]


if __name__ == "__main__":
    main()
