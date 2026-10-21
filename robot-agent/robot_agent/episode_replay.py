from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .episode_evidence import EpisodeEvidenceError, load_episode_evidence
from .episodes import EpisodeManifest, load_episode

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class EpisodeReplayError(ValueError):
    pass


@dataclass(frozen=True)
class ReplayFilters:
    task_ids: tuple[str, ...]
    executors: tuple[str, ...]
    success: str
    failure_stages: tuple[str, ...]
    outcome_codes: tuple[str, ...]
    tags: dict[str, str]


@dataclass(frozen=True)
class ReplayRoute:
    route_id: str
    executors: tuple[str, ...]
    failure_stages: tuple[str, ...]
    outcome_codes: tuple[str, ...]

    def matches(self, episode: EpisodeManifest) -> bool:
        return (
            (not self.executors or episode.executor in self.executors)
            and (not self.failure_stages or episode.failure_stage in self.failure_stages)
            and (not self.outcome_codes or episode.outcome_code in self.outcome_codes)
        )


@dataclass(frozen=True)
class ReplaySpec:
    replay_id: str
    filters: ReplayFilters
    required_evidence_kinds: tuple[str, ...]
    routes: tuple[ReplayRoute, ...]
    default_route: str
    fingerprint: str

    @classmethod
    def load(cls, path: str | Path) -> ReplaySpec:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        expected = {
            "version",
            "replay_id",
            "filters",
            "required_evidence_kinds",
            "routes",
            "default_route",
        }
        if not isinstance(raw, dict) or set(raw) != expected or raw.get("version") != 1:
            raise EpisodeReplayError("episode replay spec has missing or unexpected fields")
        filters = _parse_filters(raw["filters"])
        required = _identifier_list(
            raw["required_evidence_kinds"],
            "required_evidence_kinds",
            allow_empty=False,
        )
        route_values = raw["routes"]
        if not isinstance(route_values, list) or not 1 <= len(route_values) <= 64:
            raise EpisodeReplayError("episode replay routes must contain 1..64 entries")
        routes = tuple(_parse_route(value, index) for index, value in enumerate(route_values))
        route_ids = [route.route_id for route in routes]
        if len(route_ids) != len(set(route_ids)):
            raise EpisodeReplayError("episode replay route IDs must be unique")
        default_route = _identifier(raw["default_route"], "default_route")
        if default_route in route_ids:
            raise EpisodeReplayError("default_route must differ from explicit route IDs")
        payload = {
            "version": 1,
            "replay_id": _identifier(raw["replay_id"], "replay_id"),
            "filters": {
                "task_ids": list(filters.task_ids),
                "executors": list(filters.executors),
                "success": filters.success,
                "failure_stages": list(filters.failure_stages),
                "outcome_codes": list(filters.outcome_codes),
                "tags": filters.tags,
            },
            "required_evidence_kinds": list(required),
            "routes": [
                {
                    "route_id": route.route_id,
                    "executors": list(route.executors),
                    "failure_stages": list(route.failure_stages),
                    "outcome_codes": list(route.outcome_codes),
                }
                for route in routes
            ],
            "default_route": default_route,
        }
        return cls(
            replay_id=payload["replay_id"],
            filters=filters,
            required_evidence_kinds=required,
            routes=routes,
            default_route=default_route,
            fingerprint=_digest(payload),
        )


def build_replay_plan(
    spec: ReplaySpec,
    episode_root: str | Path,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for manifest_path in sorted(Path(episode_root).glob("*/manifest.json")):
        episode_dir = manifest_path.parent
        episode = load_episode(episode_dir)
        if not _selected(spec.filters, episode):
            continue
        try:
            evidence = load_episode_evidence(
                episode_dir,
                required_kinds=spec.required_evidence_kinds,
            )
        except EpisodeEvidenceError as exc:
            raise EpisodeReplayError(
                f"selected episode {episode.episode_id} has invalid evidence: {exc}"
            ) from exc
        matching = [route.route_id for route in spec.routes if route.matches(episode)]
        if len(matching) > 1:
            raise EpisodeReplayError(
                f"selected episode {episode.episode_id} matches multiple replay routes"
            )
        route_id = matching[0] if matching else spec.default_route
        entries.append(
            {
                "episode_id": episode.episode_id,
                "episode_manifest_sha256": episode.manifest_sha256,
                "evidence_sha256": evidence.evidence_sha256,
                "task_id": episode.task_id,
                "executor": episode.executor,
                "success": episode.success,
                "failure_stage": episode.failure_stage,
                "outcome_code": episode.outcome_code,
                "route_id": route_id,
                "artifacts": [
                    {
                        "artifact_id": artifact.artifact_id,
                        "kind": artifact.kind,
                        "path": (
                            Path(episode.episode_id)
                            / "evidence"
                            / artifact.relative_path
                        ).as_posix(),
                        "mime_type": artifact.mime_type,
                        "sha256": artifact.sha256,
                        "size_bytes": artifact.size_bytes,
                    }
                    for artifact in evidence.artifacts
                    if artifact.kind in spec.required_evidence_kinds
                ],
            }
        )
    if not entries:
        raise EpisodeReplayError("episode replay selection is empty")
    route_counts: dict[str, int] = {}
    for entry in entries:
        route_id = entry["route_id"]
        route_counts[route_id] = route_counts.get(route_id, 0) + 1
    source_set_sha = _digest(
        [
            {
                "episode_id": entry["episode_id"],
                "episode_manifest_sha256": entry["episode_manifest_sha256"],
                "evidence_sha256": entry["evidence_sha256"],
            }
            for entry in entries
        ]
    )
    report = {
        "schema_version": 1,
        "replay_id": spec.replay_id,
        "replay_fingerprint": spec.fingerprint,
        "source_set_sha256": source_set_sha,
        "required_evidence_kinds": list(spec.required_evidence_kinds),
        "episode_count": len(entries),
        "route_counts": dict(sorted(route_counts.items())),
        "entries": entries,
    }
    report["report_sha256"] = _digest(report)
    return validate_replay_plan(report)


def load_replay_plan(path: str | Path) -> dict[str, Any]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EpisodeReplayError("cannot load episode replay plan") from exc
    return validate_replay_plan(raw)


def validate_replay_plan(raw: Any) -> dict[str, Any]:
    expected = {
        "schema_version",
        "replay_id",
        "replay_fingerprint",
        "source_set_sha256",
        "required_evidence_kinds",
        "episode_count",
        "route_counts",
        "entries",
        "report_sha256",
    }
    if not isinstance(raw, dict) or set(raw) != expected or raw.get("schema_version") != 1:
        raise EpisodeReplayError("episode replay plan has invalid fields")
    _identifier(raw["replay_id"], "replay_id")
    _sha(raw["replay_fingerprint"], "replay_fingerprint")
    _sha(raw["source_set_sha256"], "source_set_sha256")
    required = _identifier_list(
        raw["required_evidence_kinds"],
        "required_evidence_kinds",
        allow_empty=False,
    )
    entries = raw["entries"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 100000:
        raise EpisodeReplayError("episode replay entries must contain 1..100000 items")
    if raw["episode_count"] != len(entries):
        raise EpisodeReplayError("episode replay count does not match entries")
    episode_ids: list[str] = []
    calculated_routes: dict[str, int] = {}
    source_bindings: list[dict[str, str]] = []
    for index, entry in enumerate(entries):
        episode_id, route_id, source_binding = _validate_replay_entry(
            entry,
            index,
            required,
        )
        episode_ids.append(episode_id)
        calculated_routes[route_id] = calculated_routes.get(route_id, 0) + 1
        source_bindings.append(source_binding)
    if episode_ids != sorted(episode_ids) or len(episode_ids) != len(set(episode_ids)):
        raise EpisodeReplayError("episode replay entries must have unique sorted episode IDs")
    route_counts = raw["route_counts"]
    expected_route_counts = dict(sorted(calculated_routes.items()))
    if not isinstance(route_counts, dict) or route_counts != expected_route_counts:
        raise EpisodeReplayError("episode replay route counts do not match entries")
    if raw["source_set_sha256"] != _digest(source_bindings):
        raise EpisodeReplayError("episode replay source set hash does not match entries")
    _sha(raw["report_sha256"], "report_sha256")
    payload = dict(raw)
    report_sha = payload.pop("report_sha256")
    if report_sha != _digest(payload):
        raise EpisodeReplayError("episode replay report hash does not match content")
    return raw


def _validate_replay_entry(
    raw: Any,
    index: int,
    required_kinds: tuple[str, ...],
) -> tuple[str, str, dict[str, str]]:
    expected = {
        "episode_id",
        "episode_manifest_sha256",
        "evidence_sha256",
        "task_id",
        "executor",
        "success",
        "failure_stage",
        "outcome_code",
        "route_id",
        "artifacts",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise EpisodeReplayError(f"episode replay entry {index} has invalid fields")
    episode_id = _identifier(raw["episode_id"], "episode_id")
    manifest_sha = _sha(raw["episode_manifest_sha256"], "episode manifest sha256")
    evidence_sha = _sha(raw["evidence_sha256"], "evidence sha256")
    _identifier(raw["task_id"], "task_id")
    _identifier(raw["executor"], "executor")
    if not isinstance(raw["success"], bool):
        raise EpisodeReplayError("episode replay success must be boolean")
    _optional_identifier(raw["failure_stage"], "failure_stage")
    _identifier(raw["outcome_code"], "outcome_code")
    route_id = _identifier(raw["route_id"], "route_id")
    artifacts = raw["artifacts"]
    if not isinstance(artifacts, list) or not 1 <= len(artifacts) <= 128:
        raise EpisodeReplayError("episode replay artifacts must contain 1..128 items")
    artifact_ids: list[str] = []
    available_kinds: set[str] = set()
    for artifact_index, artifact in enumerate(artifacts):
        artifact_id, kind = _validate_replay_artifact(
            artifact,
            artifact_index,
            episode_id,
        )
        artifact_ids.append(artifact_id)
        available_kinds.add(kind)
    if len(artifact_ids) != len(set(artifact_ids)):
        raise EpisodeReplayError("episode replay artifact IDs must be unique per episode")
    if missing := sorted(set(required_kinds) - available_kinds):
        raise EpisodeReplayError(
            "episode replay entry is missing required artifact kinds: " + ", ".join(missing)
        )
    return episode_id, route_id, {
        "episode_id": episode_id,
        "episode_manifest_sha256": manifest_sha,
        "evidence_sha256": evidence_sha,
    }


def _validate_replay_artifact(
    raw: Any,
    index: int,
    episode_id: str,
) -> tuple[str, str]:
    expected = {"artifact_id", "kind", "path", "mime_type", "sha256", "size_bytes"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise EpisodeReplayError(f"episode replay artifact {index} has invalid fields")
    artifact_id = _identifier(raw["artifact_id"], "artifact_id")
    kind = _identifier(raw["kind"], "artifact kind")
    relative = raw["path"]
    if not isinstance(relative, str):
        raise EpisodeReplayError("episode replay artifact path is invalid")
    path = Path(relative)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.parts[:3] != (episode_id, "evidence", "files")
        or len(path.parts) != 4
    ):
        raise EpisodeReplayError("episode replay artifact path is invalid")
    mime_type = raw["mime_type"]
    if not isinstance(mime_type, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9.+-]{0,63}/[A-Za-z0-9][A-Za-z0-9.+-]{0,127}",
        mime_type,
    ):
        raise EpisodeReplayError("episode replay artifact MIME type is invalid")
    _sha(raw["sha256"], "artifact sha256")
    if (
        isinstance(raw["size_bytes"], bool)
        or not isinstance(raw["size_bytes"], int)
        or not 1 <= raw["size_bytes"] <= 1024**4
    ):
        raise EpisodeReplayError("episode replay artifact size is invalid")
    return artifact_id, kind


def write_replay_plan(path: str | Path, plan: dict[str, Any]) -> None:
    validate_replay_plan(plan)
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(plan, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise EpisodeReplayError("episode replay plan already exists") from exc
        temporary.unlink()
        destination.chmod(0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _selected(filters: ReplayFilters, episode: EpisodeManifest) -> bool:
    if filters.task_ids and episode.task_id not in filters.task_ids:
        return False
    if filters.executors and episode.executor not in filters.executors:
        return False
    if filters.success == "success" and not episode.success:
        return False
    if filters.success == "failure" and episode.success:
        return False
    if filters.failure_stages and episode.failure_stage not in filters.failure_stages:
        return False
    if filters.outcome_codes and episode.outcome_code not in filters.outcome_codes:
        return False
    return all(episode.tags.get(key) == value for key, value in filters.tags.items())


def _parse_filters(raw: Any) -> ReplayFilters:
    expected = {"task_ids", "executors", "success", "failure_stages", "outcome_codes", "tags"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise EpisodeReplayError("episode replay filters have invalid fields")
    success = raw["success"]
    if success not in {"any", "success", "failure"}:
        raise EpisodeReplayError("episode replay success filter is invalid")
    tags = raw["tags"]
    if not isinstance(tags, dict) or len(tags) > 32:
        raise EpisodeReplayError("episode replay tags filter is invalid")
    return ReplayFilters(
        task_ids=_identifier_list(raw["task_ids"], "task_ids"),
        executors=_identifier_list(raw["executors"], "executors"),
        success=success,
        failure_stages=_identifier_list(raw["failure_stages"], "failure_stages"),
        outcome_codes=_identifier_list(raw["outcome_codes"], "outcome_codes"),
        tags={
            _identifier(key, "tag key"): _identifier(value, "tag value")
            for key, value in sorted(tags.items())
        },
    )


def _parse_route(raw: Any, index: int) -> ReplayRoute:
    expected = {"route_id", "executors", "failure_stages", "outcome_codes"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise EpisodeReplayError(f"episode replay route {index} has invalid fields")
    executors = _identifier_list(raw["executors"], "route executors")
    stages = _identifier_list(raw["failure_stages"], "route failure_stages")
    outcomes = _identifier_list(raw["outcome_codes"], "route outcome_codes")
    if not executors and not stages and not outcomes:
        raise EpisodeReplayError("explicit replay route must contain at least one condition")
    return ReplayRoute(
        route_id=_identifier(raw["route_id"], "route_id"),
        executors=executors,
        failure_stages=stages,
        outcome_codes=outcomes,
    )


def _identifier_list(
    raw: Any,
    label: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if (
        not isinstance(raw, list)
        or len(raw) > 128
        or (not allow_empty and not raw)
    ):
        raise EpisodeReplayError(f"{label} must be a bounded list")
    values = tuple(_identifier(value, label) for value in raw)
    if len(values) != len(set(values)):
        raise EpisodeReplayError(f"{label} contains duplicates")
    return values


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise EpisodeReplayError(f"{label} is not a safe identifier")
    return value


def _optional_identifier(value: Any, label: str) -> str:
    if value == "":
        return value
    return _identifier(value, label)


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise EpisodeReplayError(f"{label} must be a SHA-256 digest")
    return value


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an immutable offline episode replay plan")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        plan = build_replay_plan(ReplaySpec.load(args.spec), args.episodes)
        write_replay_plan(args.output, plan)
    except (EpisodeReplayError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(plan, sort_keys=True, indent=2))
    return 0


__all__ = [
    "EpisodeReplayError",
    "ReplayFilters",
    "ReplayRoute",
    "ReplaySpec",
    "build_replay_plan",
    "load_replay_plan",
    "main",
    "validate_replay_plan",
    "write_replay_plan",
]


if __name__ == "__main__":
    raise SystemExit(main())
