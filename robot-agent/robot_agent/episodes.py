from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


class EpisodeError(ValueError):
    pass


_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_VALUE = re.compile(r"robotd-tok-[A-Za-z0-9_-]+")
_SENSITIVE_KEYS = frozenset({"token", "token_secret", "authorization", "api_token"})


@dataclass(frozen=True)
class EpisodeContext:
    deployment_id: str
    tags: dict[str, str]
    fingerprint: str

    @classmethod
    def load(cls, path: str | Path) -> "EpisodeContext":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict) or set(raw) != {"version", "deployment_id", "tags"}:
            raise EpisodeError("episode context has missing or unexpected fields")
        if raw.get("version") != 1:
            raise EpisodeError("episode context version must be 1")
        deployment_id = _identifier(raw.get("deployment_id"), "deployment_id")
        tags = _tags(raw.get("tags"))
        payload = {"version": 1, "deployment_id": deployment_id, "tags": tags}
        return cls(deployment_id, tags, _digest(payload))

    @classmethod
    def empty(cls) -> "EpisodeContext":
        payload = {"version": 1, "deployment_id": "unassigned", "tags": {}}
        return cls("unassigned", {}, _digest(payload))


@dataclass(frozen=True)
class EpisodeManifest:
    episode_id: str
    task_id: str
    task_fingerprint: str
    executor: str
    started_at: float
    ended_at: float
    success: bool
    terminal_state: str
    failure_stage: str
    outcome_code: str
    human_intervention: bool
    deployment_id: str
    context_fingerprint: str
    tags: dict[str, str]
    execution_binding: dict[str, Any]
    result: dict[str, Any]
    dataset_refs: tuple[str, ...]
    manifest_sha256: str

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        execution_binding: dict[str, Any],
        started_at: float,
        ended_at: float,
        success: bool,
        terminal_state: str,
        result: dict[str, Any],
        context: EpisodeContext | None = None,
        human_intervention: bool = False,
        episode_id: str | None = None,
    ) -> "EpisodeManifest":
        active_context = context or EpisodeContext.empty()
        sanitized_binding = _json_object(_redact(execution_binding), "execution_binding")
        sanitized_result = _json_object(
            _redact(result),
            "result",
            max_bytes=2 * 1024 * 1024,
        )
        task_fingerprint = str(sanitized_binding.get("task_fingerprint", ""))
        if not _HASH.fullmatch(task_fingerprint):
            task_fingerprint = _digest(
                {
                    "task_id": task_id,
                    "executor": sanitized_binding.get("task_executor", "unknown"),
                    "binding": sanitized_binding,
                }
            )
        manifest = cls(
            episode_id=_identifier(
                episode_id or f"episode-{uuid.uuid4().hex}",
                "episode_id",
            ),
            task_id=_identifier(task_id, "task_id"),
            task_fingerprint=task_fingerprint,
            executor=str(sanitized_binding.get("task_executor", "unknown")),
            started_at=float(started_at),
            ended_at=float(ended_at),
            success=bool(success),
            terminal_state=_identifier(terminal_state, "terminal_state"),
            failure_stage=_failure_stage(sanitized_result, success),
            outcome_code=_outcome_code(sanitized_result, success),
            human_intervention=bool(human_intervention),
            deployment_id=active_context.deployment_id,
            context_fingerprint=active_context.fingerprint,
            tags=dict(active_context.tags),
            execution_binding=sanitized_binding,
            result=sanitized_result,
            dataset_refs=tuple(sorted(_dataset_refs(sanitized_result))),
            manifest_sha256="",
        )
        manifest.validate(include_hash=False)
        return cls(
            **{
                **manifest.__dict__,
                "manifest_sha256": _digest(manifest._hash_payload()),
            }
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EpisodeManifest":
        expected = {
            "schema_version",
            "episode_id",
            "task_id",
            "task_fingerprint",
            "executor",
            "started_at",
            "ended_at",
            "success",
            "terminal_state",
            "failure_stage",
            "outcome_code",
            "human_intervention",
            "deployment_id",
            "context_fingerprint",
            "tags",
            "execution_binding",
            "result",
            "dataset_refs",
            "manifest_sha256",
        }
        if not isinstance(raw, dict) or set(raw) != expected or raw.get("schema_version") != 1:
            raise EpisodeError("episode manifest has missing or unexpected fields")
        manifest = cls(
            episode_id=raw["episode_id"],
            task_id=raw["task_id"],
            task_fingerprint=raw["task_fingerprint"],
            executor=raw["executor"],
            started_at=raw["started_at"],
            ended_at=raw["ended_at"],
            success=raw["success"],
            terminal_state=raw["terminal_state"],
            failure_stage=raw["failure_stage"],
            outcome_code=raw["outcome_code"],
            human_intervention=raw["human_intervention"],
            deployment_id=raw["deployment_id"],
            context_fingerprint=raw["context_fingerprint"],
            tags=raw["tags"],
            execution_binding=raw["execution_binding"],
            result=raw["result"],
            dataset_refs=tuple(raw["dataset_refs"]),
            manifest_sha256=raw["manifest_sha256"],
        )
        manifest.validate(include_hash=True)
        return manifest

    def validate(self, *, include_hash: bool) -> None:
        _identifier(self.episode_id, "episode_id")
        _identifier(self.task_id, "task_id")
        _identifier(self.terminal_state, "terminal_state")
        if not _HASH.fullmatch(self.task_fingerprint):
            raise EpisodeError("task_fingerprint must be a SHA-256 digest")
        if not self.executor or len(self.executor) > 128:
            raise EpisodeError("episode executor is invalid")
        if not all(
            math.isfinite(value) and value > 0
            for value in (self.started_at, self.ended_at)
        ):
            raise EpisodeError("episode timestamps must be positive and finite")
        if self.ended_at < self.started_at:
            raise EpisodeError("episode end precedes its start")
        if not isinstance(self.success, bool) or not isinstance(self.human_intervention, bool):
            raise EpisodeError("episode success and intervention must be booleans")
        if self.success and self.failure_stage:
            raise EpisodeError("successful episode cannot have a failure stage")
        if not self.success and not self.failure_stage:
            raise EpisodeError("failed episode requires a failure stage")
        _identifier(self.deployment_id, "deployment_id")
        if not _HASH.fullmatch(self.context_fingerprint):
            raise EpisodeError("context_fingerprint must be a SHA-256 digest")
        _tags(self.tags)
        _json_object(self.execution_binding, "execution_binding")
        _json_object(self.result, "result", max_bytes=2 * 1024 * 1024)
        if not all(isinstance(value, str) and 0 < len(value) <= 256 for value in self.dataset_refs):
            raise EpisodeError("dataset references are invalid")
        if include_hash:
            if not _HASH.fullmatch(self.manifest_sha256):
                raise EpisodeError("episode manifest hash is invalid")
            if self.manifest_sha256 != _digest(self._hash_payload()):
                raise EpisodeError("episode manifest hash does not match content")

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "task_fingerprint": self.task_fingerprint,
            "executor": self.executor,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "success": self.success,
            "terminal_state": self.terminal_state,
            "failure_stage": self.failure_stage,
            "outcome_code": self.outcome_code,
            "human_intervention": self.human_intervention,
            "deployment_id": self.deployment_id,
            "context_fingerprint": self.context_fingerprint,
            "tags": dict(self.tags),
            "execution_binding": self.execution_binding,
            "result": self.result,
            "dataset_refs": list(self.dataset_refs),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "manifest_sha256": self.manifest_sha256}


class EpisodeRecorder:
    def __init__(self, root: str | Path, *, context: EpisodeContext | None = None):
        self.root = Path(root)
        self.context = context or EpisodeContext.empty()

    def record(self, **values: Any) -> EpisodeManifest:
        manifest = EpisodeManifest.create(context=self.context, **values)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        final = self.root / manifest.episode_id
        staging = self.root / f".{manifest.episode_id}.{uuid.uuid4().hex}.tmp"
        if final.exists():
            raise EpisodeError(f"episode already exists: {manifest.episode_id}")
        try:
            staging.mkdir(mode=0o700)
            path = staging / "manifest.json"
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(manifest.to_dict(), handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            staging.replace(final)
            return manifest
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise


def load_episode(path: str | Path) -> EpisodeManifest:
    candidate = Path(path)
    manifest_path = candidate / "manifest.json" if candidate.is_dir() else candidate
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EpisodeError(f"cannot load episode manifest: {exc}") from exc
    return EpisodeManifest.from_dict(raw)


def load_episodes(root: str | Path) -> list[EpisodeManifest]:
    path = Path(root)
    manifests = [load_episode(candidate) for candidate in sorted(path.glob("*/manifest.json"))]
    identifiers = [manifest.episode_id for manifest in manifests]
    if len(identifiers) != len(set(identifiers)):
        raise EpisodeError("episode root contains duplicate episode IDs")
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate immutable robot episode manifests")
    parser.add_argument("root")
    args = parser.parse_args()
    try:
        manifests = load_episodes(args.root)
    except (OSError, EpisodeError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(2) from exc
    payload = {
        "valid": True,
        "count": len(manifests),
        "episode_ids": [manifest.episode_id for manifest in manifests],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _failure_stage(result: dict[str, Any], success: bool) -> str:
    if success:
        return ""
    for steps in reversed(_find_all(result, "steps")):
        if isinstance(steps, list):
            for step in reversed(steps):
                if isinstance(step, dict) and step.get("success") is False:
                    action = step.get("action")
                    if isinstance(action, str) and action:
                        return action[:128]
    error_type = result.get("error_type")
    return str(error_type or "execution")[:128]


def _outcome_code(result: dict[str, Any], success: bool) -> str:
    for key in ("outcome_code", "reason_code", "error_type"):
        value = _find_key(result, key)
        if isinstance(value, str) and value:
            return value[:128]
    return "SUCCESS" if success else "FAILED"


def _dataset_refs(result: dict[str, Any]) -> set[str]:
    references: set[str] = set()
    for key in ("dataset_repo_id", "dataset_id", "episode_ref"):
        for value in _find_all(result, key):
            if isinstance(value, str) and 0 < len(value) <= 256:
                references.add(value)
    return references


def _find_key(value: Any, key: str) -> Any:
    values = _find_all(value, key)
    return values[0] if values else None


def _find_all(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key == key:
                found.append(child)
            found.extend(_find_all(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_find_all(child, key))
    return found


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if str(key).lower() in _SENSITIVE_KEYS else _redact(child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(child) for child in value]
    if isinstance(value, str):
        return _TOKEN_VALUE.sub("[REDACTED]", value)
    return value


def _tags(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or len(value) > 64:
        raise EpisodeError("episode tags must be an object with at most 64 entries")
    tags: dict[str, str] = {}
    for key, item in value.items():
        tags[_identifier(key, "tag key")] = _identifier(item, f"tag {key}")
    return dict(sorted(tags.items()))


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise EpisodeError(f"{label} is not a safe identifier")
    return value


def _json_object(value: Any, label: str, *, max_bytes: int = 256 * 1024) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EpisodeError(f"{label} must be an object")
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise EpisodeError(f"{label} must be JSON-safe") from exc
    if len(encoded.encode("utf-8")) > max_bytes:
        raise EpisodeError(f"{label} is too large")
    normalized = json.loads(encoded)
    if not isinstance(normalized, dict):
        raise EpisodeError(f"{label} must be an object")
    return normalized


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "EpisodeContext",
    "EpisodeError",
    "EpisodeManifest",
    "EpisodeRecorder",
    "load_episode",
    "load_episodes",
    "main",
]


if __name__ == "__main__":
    main()
