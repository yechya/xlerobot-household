from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import time
import uuid
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

import yaml

from .episode_evidence import EpisodeEvidenceError, load_episode_evidence
from .episode_replay import EpisodeReplayError, load_replay_plan
from .episodes import load_episode

ENTRY_POINT_GROUP = "xlerobot.replay_handlers"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_DISTRIBUTION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_MIME_TYPE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9.+-]{0,63}/[A-Za-z0-9][A-Za-z0-9.+-]{0,127}$"
)


class ReplayRuntimeError(ValueError):
    pass


@dataclass(frozen=True)
class BoundArtifact:
    path: Path
    sha256: str


@dataclass(frozen=True)
class ReplayHandlerSpec:
    handler_id: str
    entry_point: str
    distribution: str
    distribution_version: str
    artifacts: tuple[BoundArtifact, ...]
    options: dict[str, Any]


@dataclass(frozen=True)
class ReplayRuntimeSpec:
    runtime_id: str
    plan_path: Path
    plan_sha256: str
    episode_root: Path
    output_root: Path
    max_entries: int
    max_output_bytes_per_entry: int
    failure_policy: str
    handlers: dict[str, ReplayHandlerSpec]
    routes: dict[str, tuple[str, ...]]
    fingerprint: str

    @classmethod
    def load(cls, path: str | Path) -> ReplayRuntimeSpec:
        config_path = Path(path).expanduser().resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        expected = {
            "version",
            "approved",
            "runtime_id",
            "plan",
            "episode_root",
            "output_root",
            "max_entries",
            "max_output_bytes_per_entry",
            "failure_policy",
            "handlers",
            "routes",
        }
        version = raw.get("version") if isinstance(raw, dict) else None
        if not isinstance(raw, dict) or set(raw) != expected or version not in {1, 2}:
            raise ReplayRuntimeError("replay runtime config has missing or unexpected fields")
        if raw["approved"] is not True:
            raise ReplayRuntimeError("replay runtime config must be approved")
        plan = raw["plan"]
        if not isinstance(plan, dict) or set(plan) != {"path", "sha256"}:
            raise ReplayRuntimeError("replay runtime plan binding is invalid")
        plan_path = _resolved_path(plan["path"], config_path.parent, "plan path")
        episode_root = _resolved_path(
            raw["episode_root"], config_path.parent, "episode_root"
        )
        output_root = _resolved_path(raw["output_root"], config_path.parent, "output_root")
        handlers = _parse_handlers(raw["handlers"], config_path.parent)
        routes = _parse_routes(raw["routes"], handlers, version=version)
        payload = {
            **raw,
            "plan": {"path": str(plan_path), "sha256": _sha(plan["sha256"], "plan sha256")},
            "episode_root": str(episode_root),
            "output_root": str(output_root),
            "handlers": {
                handler_id: {
                    "entry_point": handler.entry_point,
                    "distribution": handler.distribution,
                    "distribution_version": handler.distribution_version,
                    "artifacts": [
                        {"path": str(artifact.path), "sha256": artifact.sha256}
                        for artifact in handler.artifacts
                    ],
                    "options": handler.options,
                }
                for handler_id, handler in handlers.items()
            },
            "routes": {
                route_id: handler_ids[0] if version == 1 else list(handler_ids)
                for route_id, handler_ids in routes.items()
            },
        }
        return cls(
            runtime_id=_identifier(raw["runtime_id"], "runtime_id"),
            plan_path=plan_path,
            plan_sha256=payload["plan"]["sha256"],
            episode_root=episode_root,
            output_root=output_root,
            max_entries=_integer(raw["max_entries"], "max_entries", 1, 100000),
            max_output_bytes_per_entry=_integer(
                raw["max_output_bytes_per_entry"],
                "max_output_bytes_per_entry",
                1,
                1024**4,
            ),
            failure_policy=_choice(raw["failure_policy"], "failure_policy", {"continue", "stop"}),
            handlers=handlers,
            routes=routes,
            fingerprint=_digest(payload),
        )

    def validate_bindings(self) -> dict[str, Any]:
        if not self.plan_path.is_file() or _sha256_file(self.plan_path) != self.plan_sha256:
            raise ReplayRuntimeError("replay plan SHA-256 does not match runtime config")
        plan = load_replay_plan(self.plan_path)
        if plan["episode_count"] > self.max_entries:
            raise ReplayRuntimeError("replay plan exceeds configured max_entries")
        route_ids = set(plan["route_counts"])
        if set(self.routes) != route_ids:
            raise ReplayRuntimeError("replay runtime routes must exactly cover replay plan routes")
        if self.output_root == self.episode_root or self.output_root.is_relative_to(
            self.episode_root
        ):
            raise ReplayRuntimeError("replay output_root must be outside the episode store")
        for handler in self.handlers.values():
            for artifact in handler.artifacts:
                if not artifact.path.is_file() or artifact.path.is_symlink():
                    raise ReplayRuntimeError("replay handler artifact must be a regular file")
                if _sha256_file(artifact.path) != artifact.sha256:
                    raise ReplayRuntimeError("replay handler artifact SHA-256 mismatch")
        return plan


@dataclass(frozen=True)
class ReplayInputArtifact:
    artifact_id: str
    kind: str
    source: Path
    mime_type: str
    sha256: str
    size_bytes: int
    captured_at_start: float
    captured_at_end: float


@dataclass(frozen=True)
class ReplayInvocation:
    runtime_id: str
    replay_id: str
    episode_id: str
    route_id: str
    task_id: str
    executor: str
    success: bool
    failure_stage: str
    outcome_code: str
    started_at: float
    ended_at: float
    output_dir: Path
    artifacts: tuple[ReplayInputArtifact, ...]


@dataclass(frozen=True)
class ReplayProducedArtifact:
    artifact_id: str
    kind: str
    relative_path: str
    mime_type: str
    captured_at_start: float
    captured_at_end: float


@dataclass(frozen=True)
class ReplayHandlerResult:
    metadata: dict[str, Any]
    produced_artifacts: tuple[ReplayProducedArtifact, ...] = ()


class ReplayHandler(Protocol):
    def execute(
        self,
        invocation: ReplayInvocation,
    ) -> ReplayHandlerResult | dict[str, Any] | None: ...


def load_handlers(
    spec: ReplayRuntimeSpec,
    *,
    entry_points: Any | None = None,
) -> dict[str, ReplayHandler]:
    available = metadata.entry_points() if entry_points is None else entry_points
    loaded: dict[str, ReplayHandler] = {}
    for handler_id, handler_spec in spec.handlers.items():
        if hasattr(available, "select"):
            candidates = list(
                available.select(
                    group=ENTRY_POINT_GROUP,
                    name=handler_spec.entry_point,
                )
            )
        else:
            candidates = [
                entry
                for entry in available.get(ENTRY_POINT_GROUP, ())
                if entry.name == handler_spec.entry_point
            ]
        if len(candidates) != 1:
            raise ReplayRuntimeError(
                f"expected exactly one {ENTRY_POINT_GROUP}:{handler_spec.entry_point} entry point"
            )
        entry = candidates[0]
        distribution = getattr(entry, "dist", None)
        name = getattr(distribution, "name", None)
        version = getattr(distribution, "version", None)
        if name is None and distribution is not None:
            name = distribution.metadata.get("Name")
        if _normalized_distribution(name) != _normalized_distribution(
            handler_spec.distribution
        ):
            raise ReplayRuntimeError("replay handler distribution does not match config")
        if version != handler_spec.distribution_version:
            raise ReplayRuntimeError("replay handler distribution version does not match config")
        factory = entry.load()
        if not callable(factory):
            raise ReplayRuntimeError("replay handler entry point must be a factory")
        handler = factory(
            options=dict(handler_spec.options),
            artifacts=tuple(artifact.path for artifact in handler_spec.artifacts),
        )
        if not callable(getattr(handler, "execute", None)):
            raise ReplayRuntimeError("replay handler must provide execute(invocation)")
        loaded[handler_id] = handler
    return loaded


def execute_replay(
    spec: ReplayRuntimeSpec,
    *,
    entry_points: Any | None = None,
    generated_at: float | None = None,
) -> Path:
    plan = spec.validate_bindings()
    handlers = load_handlers(spec, entry_points=entry_points)
    final = spec.output_root / spec.runtime_id
    staging = spec.output_root / f".{spec.runtime_id}.{uuid.uuid4().hex}.tmp"
    if final.exists():
        raise ReplayRuntimeError("replay runtime output already exists")
    spec.output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    results: list[dict[str, Any]] = []
    stopped = False
    try:
        staging.mkdir(mode=0o700)
        entries_root = staging / "entries"
        entries_root.mkdir(mode=0o700)
        for entry in plan["entries"]:
            episode_output = entries_root / entry["episode_id"]
            episode_output_bytes = 0
            derived_artifacts: list[ReplayInputArtifact] = []
            for handler_id in spec.routes[entry["route_id"]]:
                handler_spec = spec.handlers[handler_id]
                if stopped:
                    results.append(_skipped_result(entry, handler_id, handler_spec))
                    continue
                handler_output = episode_output / handler_id
                try:
                    handler_output.mkdir(parents=True, mode=0o700)
                    invocation = _build_invocation(
                        spec,
                        plan,
                        entry,
                        handler_output,
                        additional_artifacts=tuple(derived_artifacts),
                    )
                    handler_value = _normalize_handler_result(
                        handlers[handler_id].execute(invocation)
                    )
                    handler_metadata = _json_object(
                        handler_value.metadata,
                        "replay handler metadata",
                    )
                    outputs = _collect_outputs(
                        staging,
                        handler_output,
                        spec.max_output_bytes_per_entry - episode_output_bytes,
                    )
                    produced, promoted = _validate_produced_artifacts(
                        staging,
                        handler_output,
                        invocation,
                        outputs,
                        handler_value.produced_artifacts,
                    )
                    existing_ids = {
                        artifact.artifact_id
                        for artifact in (*invocation.artifacts, *derived_artifacts)
                    }
                    duplicate_ids = sorted(
                        existing_ids & {artifact.artifact_id for artifact in promoted}
                    )
                    if duplicate_ids:
                        raise ReplayRuntimeError(
                            "derived artifact IDs already exist: " + ", ".join(duplicate_ids)
                        )
                    derived_artifacts.extend(promoted)
                    episode_output_bytes += sum(output["size_bytes"] for output in outputs)
                    results.append(
                        _result(
                            entry,
                            handler_id,
                            handler_spec,
                            status="succeeded",
                            error="",
                            outputs=outputs,
                            produced_artifacts=produced,
                            handler_metadata=handler_metadata,
                        )
                    )
                except Exception as exc:
                    error = _bounded_error(exc)
                    try:
                        outputs = _collect_outputs(
                            staging,
                            handler_output,
                            spec.max_output_bytes_per_entry - episode_output_bytes,
                        )
                        episode_output_bytes += sum(
                            output["size_bytes"] for output in outputs
                        )
                    except Exception as output_exc:
                        shutil.rmtree(handler_output, ignore_errors=True)
                        outputs = []
                        error = _bounded_error(
                            ReplayRuntimeError(
                                f"{error}; rejected handler diagnostics: {output_exc}"
                            )
                        )
                    results.append(
                        _result(
                            entry,
                            handler_id,
                            handler_spec,
                            status="failed",
                            error=error,
                            outputs=outputs,
                            produced_artifacts=[],
                            handler_metadata={},
                        )
                    )
                    stopped = spec.failure_policy == "stop"
        report = _create_report(spec, plan, results, generated_at=generated_at)
        _write_json_exclusive(staging / "report.json", report)
        _fsync_tree(staging)
        try:
            staging.rename(final)
        except FileExistsError as exc:
            raise ReplayRuntimeError("replay runtime output already exists") from exc
        _fsync_directory(spec.output_root)
        return final
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_replay_run(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    try:
        raw = json.loads((root / "report.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayRuntimeError("cannot load replay runtime report") from exc
    expected = {
        "schema_version",
        "runtime_id",
        "runtime_fingerprint",
        "replay_id",
        "replay_report_sha256",
        "generated_at",
        "passed",
        "counts",
        "entries",
        "report_sha256",
    }
    schema_version = raw.get("schema_version") if isinstance(raw, dict) else None
    if not isinstance(raw, dict) or set(raw) != expected or schema_version not in {1, 2, 3}:
        raise ReplayRuntimeError("replay runtime report has invalid fields")
    _identifier(raw["runtime_id"], "runtime_id")
    _sha(raw["runtime_fingerprint"], "runtime fingerprint")
    _identifier(raw["replay_id"], "replay_id")
    _sha(raw["replay_report_sha256"], "replay report sha256")
    _positive_float(raw["generated_at"], "generated_at")
    if not isinstance(raw["passed"], bool):
        raise ReplayRuntimeError("replay runtime verdict is invalid")
    entries = raw["entries"]
    if not isinstance(entries, list) or not entries:
        raise ReplayRuntimeError("replay runtime entries are invalid")
    calculated = {"succeeded": 0, "failed": 0, "skipped": 0}
    invocation_ids: set[tuple[str, ...]] = set()
    for index, entry in enumerate(entries):
        status = _validate_result_entry(root, entry, index, schema_version=schema_version)
        calculated[status] += 1
        invocation_id = (
            (entry["episode_id"],)
            if schema_version == 1
            else (entry["episode_id"], entry["handler_id"])
        )
        if invocation_id in invocation_ids:
            raise ReplayRuntimeError("replay runtime episode/handler identity is duplicated")
        invocation_ids.add(invocation_id)
    if raw["counts"] != calculated:
        raise ReplayRuntimeError("replay runtime counts do not match entries")
    if raw["passed"] != (calculated["failed"] == 0 and calculated["skipped"] == 0):
        raise ReplayRuntimeError("replay runtime verdict contradicts entries")
    _sha(raw["report_sha256"], "report sha256")
    payload = dict(raw)
    report_sha = payload.pop("report_sha256")
    if report_sha != _digest(payload):
        raise ReplayRuntimeError("replay runtime report hash does not match content")
    return raw


class EvidenceExportHandler:
    def __init__(self, include_kinds: tuple[str, ...], preserve_extension: bool) -> None:
        self.include_kinds = include_kinds
        self.preserve_extension = preserve_extension

    def execute(self, invocation: ReplayInvocation) -> ReplayHandlerResult:
        selected = [
            artifact for artifact in invocation.artifacts if artifact.kind in self.include_kinds
        ]
        missing = sorted(set(self.include_kinds) - {artifact.kind for artifact in selected})
        if missing:
            raise ReplayRuntimeError(
                "evidence export is missing configured kinds: " + ", ".join(missing)
            )
        for artifact in selected:
            suffix = artifact.source.suffix.lower() if self.preserve_extension else ""
            destination = invocation.output_dir / f"{artifact.artifact_id}{suffix}"
            _copy_regular_file(artifact.source, destination)
        return ReplayHandlerResult(
            metadata={
                "exported_artifact_ids": [artifact.artifact_id for artifact in selected],
                "exported_kinds": sorted({artifact.kind for artifact in selected}),
            },
            produced_artifacts=tuple(
                ReplayProducedArtifact(
                    artifact_id=f"exported_{artifact.artifact_id}",
                    kind=artifact.kind,
                    relative_path=(
                        f"{artifact.artifact_id}{artifact.source.suffix.lower()}"
                        if self.preserve_extension
                        else artifact.artifact_id
                    ),
                    mime_type=artifact.mime_type,
                    captured_at_start=artifact.captured_at_start,
                    captured_at_end=artifact.captured_at_end,
                )
                for artifact in selected
            ),
        )


def create_evidence_export_handler(
    *,
    options: dict[str, Any],
    artifacts: tuple[Path, ...],
) -> EvidenceExportHandler:
    if artifacts:
        raise ReplayRuntimeError("evidence export handler does not accept bound artifacts")
    expected = {"include_kinds", "preserve_extension"}
    if set(options) != expected:
        raise ReplayRuntimeError("evidence export handler options are invalid")
    include = _identifier_list(options["include_kinds"], "include_kinds", allow_empty=False)
    preserve = options["preserve_extension"]
    if not isinstance(preserve, bool):
        raise ReplayRuntimeError("preserve_extension must be boolean")
    return EvidenceExportHandler(include, preserve)


def _build_invocation(
    spec: ReplayRuntimeSpec,
    plan: dict[str, Any],
    entry: dict[str, Any],
    output_dir: Path,
    *,
    additional_artifacts: tuple[ReplayInputArtifact, ...] = (),
) -> ReplayInvocation:
    episode_dir = spec.episode_root / entry["episode_id"]
    try:
        episode = load_episode(episode_dir)
        evidence = load_episode_evidence(
            episode_dir,
            required_kinds=tuple(plan["required_evidence_kinds"]),
        )
    except EpisodeEvidenceError as exc:
        raise ReplayRuntimeError("replay episode evidence is invalid") from exc
    if evidence.episode_manifest_sha256 != entry["episode_manifest_sha256"]:
        raise ReplayRuntimeError("replay episode manifest binding changed")
    if evidence.evidence_sha256 != entry["evidence_sha256"]:
        raise ReplayRuntimeError("replay evidence binding changed")
    indexed = {artifact.artifact_id: artifact for artifact in evidence.artifacts}
    artifacts: list[ReplayInputArtifact] = []
    for selected in entry["artifacts"]:
        artifact = indexed.get(selected["artifact_id"])
        if artifact is None:
            raise ReplayRuntimeError("replay artifact no longer exists in evidence")
        expected_path = Path(entry["episode_id"]) / "evidence" / artifact.relative_path
        if (
            selected["path"] != expected_path.as_posix()
            or selected["kind"] != artifact.kind
            or selected["mime_type"] != artifact.mime_type
            or selected["sha256"] != artifact.sha256
            or selected["size_bytes"] != artifact.size_bytes
        ):
            raise ReplayRuntimeError("replay artifact binding changed")
        artifacts.append(
            ReplayInputArtifact(
                artifact_id=artifact.artifact_id,
                kind=artifact.kind,
                source=episode_dir / "evidence" / artifact.relative_path,
                mime_type=artifact.mime_type,
                sha256=artifact.sha256,
                size_bytes=artifact.size_bytes,
                captured_at_start=artifact.captured_at_start,
                captured_at_end=artifact.captured_at_end,
            )
        )
    return ReplayInvocation(
        runtime_id=spec.runtime_id,
        replay_id=plan["replay_id"],
        episode_id=entry["episode_id"],
        route_id=entry["route_id"],
        task_id=entry["task_id"],
        executor=entry["executor"],
        success=entry["success"],
        failure_stage=entry["failure_stage"],
        outcome_code=entry["outcome_code"],
        started_at=episode.started_at,
        ended_at=episode.ended_at,
        output_dir=output_dir,
        artifacts=(*artifacts, *additional_artifacts),
    )


def _normalize_handler_result(raw: Any) -> ReplayHandlerResult:
    if raw is None:
        return ReplayHandlerResult(metadata={})
    if isinstance(raw, dict):
        return ReplayHandlerResult(metadata=raw)
    if not isinstance(raw, ReplayHandlerResult):
        raise ReplayRuntimeError(
            "replay handler must return ReplayHandlerResult, a metadata object or None"
        )
    if not isinstance(raw.produced_artifacts, tuple) or len(raw.produced_artifacts) > 128:
        raise ReplayRuntimeError("replay handler produced_artifacts must be a bounded tuple")
    if not all(isinstance(value, ReplayProducedArtifact) for value in raw.produced_artifacts):
        raise ReplayRuntimeError("replay handler produced artifact type is invalid")
    return raw


def _validate_produced_artifacts(
    root: Path,
    handler_output: Path,
    invocation: ReplayInvocation,
    outputs: list[dict[str, Any]],
    produced: tuple[ReplayProducedArtifact, ...],
) -> tuple[list[dict[str, Any]], list[ReplayInputArtifact]]:
    if not produced:
        return [], []
    output_by_path = {output["path"]: output for output in outputs}
    earliest = min(artifact.captured_at_start for artifact in invocation.artifacts)
    latest = max(artifact.captured_at_end for artifact in invocation.artifacts)
    records: list[dict[str, Any]] = []
    promoted: list[ReplayInputArtifact] = []
    identifiers: list[str] = []
    paths: list[str] = []
    for descriptor in produced:
        artifact_id = _identifier(descriptor.artifact_id, "produced artifact_id")
        kind = _identifier(descriptor.kind, "produced artifact kind")
        mime_type = _mime_type(descriptor.mime_type)
        relative = descriptor.relative_path
        path = Path(relative) if isinstance(relative, str) else Path("/")
        if path.is_absolute() or ".." in path.parts or len(path.parts) == 0:
            raise ReplayRuntimeError("produced artifact relative path is unsafe")
        destination = handler_output / path
        try:
            report_path = destination.relative_to(root).as_posix()
        except ValueError as exc:
            raise ReplayRuntimeError("produced artifact escapes replay output") from exc
        output = output_by_path.get(report_path)
        if output is None:
            raise ReplayRuntimeError("produced artifact does not reference a validated output")
        started = _positive_float(
            descriptor.captured_at_start,
            "produced captured_at_start",
        )
        ended = _positive_float(
            descriptor.captured_at_end,
            "produced captured_at_end",
        )
        if ended < started or started < earliest or ended > latest:
            raise ReplayRuntimeError("produced artifact timestamps exceed source evidence")
        record = {
            "artifact_id": artifact_id,
            "kind": kind,
            "path": report_path,
            "mime_type": mime_type,
            "size_bytes": output["size_bytes"],
            "sha256": output["sha256"],
            "captured_at_start": started,
            "captured_at_end": ended,
        }
        records.append(record)
        promoted.append(
            ReplayInputArtifact(
                artifact_id=artifact_id,
                kind=kind,
                source=destination,
                mime_type=mime_type,
                sha256=output["sha256"],
                size_bytes=output["size_bytes"],
                captured_at_start=started,
                captured_at_end=ended,
            )
        )
        identifiers.append(artifact_id)
        paths.append(report_path)
    if len(identifiers) != len(set(identifiers)) or len(paths) != len(set(paths)):
        raise ReplayRuntimeError("produced artifact IDs and paths must be unique")
    return records, promoted


def _collect_outputs(root: Path, entry_dir: Path, maximum: int) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    total = 0
    for path in sorted(entry_dir.rglob("*")):
        info = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode):
            if path.is_dir() and not path.is_symlink():
                continue
            raise ReplayRuntimeError("replay handler output must be a regular file")
        total += info.st_size
        if info.st_size <= 0 or total > maximum or len(outputs) >= 128:
            raise ReplayRuntimeError("replay handler output exceeds configured bounds")
        outputs.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": info.st_size,
                "sha256": _sha256_file(path),
            }
        )
    return outputs


def _create_report(
    spec: ReplayRuntimeSpec,
    plan: dict[str, Any],
    entries: list[dict[str, Any]],
    *,
    generated_at: float | None,
) -> dict[str, Any]:
    counts = {
        status: sum(entry["status"] == status for entry in entries)
        for status in ("succeeded", "failed", "skipped")
    }
    report = {
        "schema_version": 3,
        "runtime_id": spec.runtime_id,
        "runtime_fingerprint": spec.fingerprint,
        "replay_id": plan["replay_id"],
        "replay_report_sha256": plan["report_sha256"],
        "generated_at": time.time() if generated_at is None else generated_at,
        "passed": counts["failed"] == 0 and counts["skipped"] == 0,
        "counts": counts,
        "entries": entries,
    }
    _positive_float(report["generated_at"], "generated_at")
    report["report_sha256"] = _digest(report)
    return report


def _result(
    entry: dict[str, Any],
    handler_id: str,
    handler: ReplayHandlerSpec,
    *,
    status: str,
    error: str,
    outputs: list[dict[str, Any]],
    produced_artifacts: list[dict[str, Any]],
    handler_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "episode_id": entry["episode_id"],
        "route_id": entry["route_id"],
        "handler_id": handler_id,
        "handler_entry_point": handler.entry_point,
        "handler_distribution": handler.distribution,
        "handler_distribution_version": handler.distribution_version,
        "status": status,
        "error": error,
        "outputs": outputs,
        "produced_artifacts": produced_artifacts,
        "handler_metadata": handler_metadata,
    }


def _skipped_result(
    entry: dict[str, Any],
    handler_id: str,
    handler: ReplayHandlerSpec,
) -> dict[str, Any]:
    return _result(
        entry,
        handler_id,
        handler,
        status="skipped",
        error="stopped after previous handler failure",
        outputs=[],
        produced_artifacts=[],
        handler_metadata={},
    )


def _validate_result_entry(
    root: Path,
    raw: Any,
    index: int,
    *,
    schema_version: int,
) -> str:
    expected = {
        "episode_id",
        "route_id",
        "handler_id",
        "handler_entry_point",
        "handler_distribution",
        "handler_distribution_version",
        "status",
        "error",
        "outputs",
        "handler_metadata",
    }
    if schema_version == 3:
        expected.add("produced_artifacts")
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ReplayRuntimeError(f"replay runtime entry {index} has invalid fields")
    for key in ("episode_id", "route_id", "handler_id", "handler_entry_point"):
        _identifier(raw[key], key)
    _distribution(raw["handler_distribution"])
    _nonempty(raw["handler_distribution_version"], "handler_distribution_version")
    status = _choice(raw["status"], "status", {"succeeded", "failed", "skipped"})
    error = raw["error"]
    if not isinstance(error, str) or len(error) > 512 or (status == "succeeded") == bool(error):
        raise ReplayRuntimeError("replay runtime entry error is invalid")
    outputs = raw["outputs"]
    if not isinstance(outputs, list) or len(outputs) > 128:
        raise ReplayRuntimeError("replay runtime outputs are invalid")
    if status == "skipped" and outputs:
        raise ReplayRuntimeError("skipped replay entry cannot bind outputs")
    for output in outputs:
        if not isinstance(output, dict) or set(output) != {"path", "size_bytes", "sha256"}:
            raise ReplayRuntimeError("replay runtime output has invalid fields")
        relative = output["path"]
        path = Path(relative) if isinstance(relative, str) else Path("/")
        expected_prefix = (
            ("entries", raw["episode_id"])
            if schema_version == 1
            else ("entries", raw["episode_id"], raw["handler_id"])
        )
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.parts[: len(expected_prefix)] != expected_prefix
            or len(path.parts) <= len(expected_prefix)
        ):
            raise ReplayRuntimeError("replay runtime output path is unsafe")
        destination = root / path
        if destination.is_symlink() or not destination.is_file():
            raise ReplayRuntimeError("replay runtime output file is missing")
        _integer(output["size_bytes"], "output size", 1, 1024**4)
        if destination.stat().st_size != output["size_bytes"]:
            raise ReplayRuntimeError("replay runtime output size changed")
        if _sha256_file(destination) != _sha(output["sha256"], "output sha256"):
            raise ReplayRuntimeError("replay runtime output hash changed")
    if schema_version == 3:
        _validate_report_produced_artifacts(raw["produced_artifacts"], outputs)
    _json_object(raw["handler_metadata"], "handler_metadata")
    return status


def _validate_report_produced_artifacts(raw: Any, outputs: list[dict[str, Any]]) -> None:
    if not isinstance(raw, list) or len(raw) > 128:
        raise ReplayRuntimeError("replay runtime produced artifacts are invalid")
    output_by_path = {output["path"]: output for output in outputs}
    identifiers: list[str] = []
    paths: list[str] = []
    expected = {
        "artifact_id",
        "kind",
        "path",
        "mime_type",
        "size_bytes",
        "sha256",
        "captured_at_start",
        "captured_at_end",
    }
    for value in raw:
        if not isinstance(value, dict) or set(value) != expected:
            raise ReplayRuntimeError("replay runtime produced artifact has invalid fields")
        identifiers.append(_identifier(value["artifact_id"], "produced artifact_id"))
        _identifier(value["kind"], "produced artifact kind")
        _mime_type(value["mime_type"])
        path = value["path"]
        if not isinstance(path, str) or path not in output_by_path:
            raise ReplayRuntimeError("produced artifact does not bind a report output")
        output = output_by_path[path]
        if value["size_bytes"] != output["size_bytes"] or value["sha256"] != output["sha256"]:
            raise ReplayRuntimeError("produced artifact output binding changed")
        started = _positive_float(value["captured_at_start"], "produced captured_at_start")
        ended = _positive_float(value["captured_at_end"], "produced captured_at_end")
        if ended < started:
            raise ReplayRuntimeError("produced artifact end precedes start")
        paths.append(path)
    if len(identifiers) != len(set(identifiers)) or len(paths) != len(set(paths)):
        raise ReplayRuntimeError("produced artifact IDs and paths must be unique")


def _parse_handlers(raw: Any, base: Path) -> dict[str, ReplayHandlerSpec]:
    if not isinstance(raw, dict) or not 1 <= len(raw) <= 64:
        raise ReplayRuntimeError("replay handlers must be a bounded non-empty object")
    handlers: dict[str, ReplayHandlerSpec] = {}
    expected = {
        "entry_point",
        "distribution",
        "distribution_version",
        "artifacts",
        "options",
    }
    for handler_id, value in raw.items():
        identifier = _identifier(handler_id, "handler_id")
        if not isinstance(value, dict) or set(value) != expected:
            raise ReplayRuntimeError("replay handler config has invalid fields")
        handlers[identifier] = ReplayHandlerSpec(
            handler_id=identifier,
            entry_point=_identifier(value["entry_point"], "entry_point"),
            distribution=_distribution(value["distribution"]),
            distribution_version=_nonempty(
                value["distribution_version"], "distribution_version"
            ),
            artifacts=_bound_artifacts(value["artifacts"], base),
            options=_json_object(value["options"], "handler options"),
        )
    return dict(sorted(handlers.items()))


def _parse_routes(
    raw: Any,
    handlers: dict[str, ReplayHandlerSpec],
    *,
    version: int,
) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw, dict) or not 1 <= len(raw) <= 128:
        raise ReplayRuntimeError("replay routes must be a bounded non-empty object")
    routes: dict[str, tuple[str, ...]] = {}
    for route, value in raw.items():
        route_id = _identifier(route, "route_id")
        if version == 1:
            handler_ids = (_identifier(value, "handler_id"),)
        else:
            handler_ids = _identifier_list(value, "route handler IDs", allow_empty=False)
        if len(handler_ids) > 16:
            raise ReplayRuntimeError("a replay route cannot contain more than 16 handlers")
        routes[route_id] = handler_ids
    referenced = {handler_id for handler_ids in routes.values() for handler_id in handler_ids}
    unknown = sorted(referenced - set(handlers))
    if unknown:
        raise ReplayRuntimeError("replay routes reference unknown handlers: " + ", ".join(unknown))
    return dict(sorted(routes.items()))


def _bound_artifacts(raw: Any, base: Path) -> tuple[BoundArtifact, ...]:
    if not isinstance(raw, list) or len(raw) > 64:
        raise ReplayRuntimeError("replay handler artifacts must be a bounded list")
    artifacts: list[BoundArtifact] = []
    for value in raw:
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            raise ReplayRuntimeError("replay handler artifact binding is invalid")
        artifacts.append(
            BoundArtifact(
                path=_resolved_path(value["path"], base, "handler artifact path"),
                sha256=_sha(value["sha256"], "handler artifact sha256"),
            )
        )
    paths = [artifact.path for artifact in artifacts]
    if len(paths) != len(set(paths)):
        raise ReplayRuntimeError("replay handler artifact paths must be unique")
    return tuple(artifacts)


def _copy_regular_file(source: Path, destination: Path) -> None:
    source_descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(source_descriptor).st_mode):
            raise ReplayRuntimeError("replay input must remain a regular file")
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(source_descriptor, "rb") as incoming, os.fdopen(
            destination_descriptor, "wb"
        ) as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        source_descriptor = -1
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        if source_descriptor >= 0:
            os.close(source_descriptor)


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_tree(root: Path) -> None:
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        _fsync_directory(directory)
    _fsync_directory(root)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _bounded_error(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}".replace("\n", " ").strip()
    return (message or type(exc).__name__)[:512]


def _json_object(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or len(raw) > 128:
        raise ReplayRuntimeError(f"{label} must be a bounded JSON object")
    try:
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ReplayRuntimeError(f"{label} must contain JSON values") from exc
    if len(encoded.encode("utf-8")) > 64 * 1024:
        raise ReplayRuntimeError(f"{label} is too large")
    return json.loads(encoded)


def _identifier_list(raw: Any, label: str, *, allow_empty: bool) -> tuple[str, ...]:
    if (
        not isinstance(raw, list)
        or len(raw) > 128
        or (not allow_empty and not raw)
    ):
        raise ReplayRuntimeError(f"{label} must be a bounded list")
    values = tuple(_identifier(value, label) for value in raw)
    if len(values) != len(set(values)):
        raise ReplayRuntimeError(f"{label} contains duplicates")
    return values


def _resolved_path(raw: Any, base: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ReplayRuntimeError(f"{label} is invalid")
    path = Path(raw).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ReplayRuntimeError(f"{label} is not a safe identifier")
    return value


def _distribution(value: Any) -> str:
    if not isinstance(value, str) or not _DISTRIBUTION.fullmatch(value):
        raise ReplayRuntimeError("distribution is invalid")
    return value


def _mime_type(value: Any) -> str:
    if not isinstance(value, str) or not _MIME_TYPE.fullmatch(value):
        raise ReplayRuntimeError("produced artifact MIME type is invalid")
    return value


def _normalized_distribution(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[-_.]+", "-", value).lower()


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ReplayRuntimeError(f"{label} is invalid")
    return value


def _choice(value: Any, label: str, choices: set[str]) -> str:
    if value not in choices:
        raise ReplayRuntimeError(f"{label} is invalid")
    return value


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ReplayRuntimeError(f"{label} must be in [{minimum}, {maximum}]")
    return value


def _positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayRuntimeError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ReplayRuntimeError(f"{label} must be positive")
    return parsed


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ReplayRuntimeError(f"{label} must be a SHA-256 digest")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute a bound offline episode replay plan")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            output = execute_replay(ReplayRuntimeSpec.load(args.config))
            report = load_replay_run(output)
        else:
            output = Path(args.run).expanduser().resolve()
            report = load_replay_run(output)
    except (ReplayRuntimeError, EpisodeReplayError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": str(output), "report": report}, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


__all__ = [
    "ENTRY_POINT_GROUP",
    "EvidenceExportHandler",
    "ReplayHandler",
    "ReplayHandlerResult",
    "ReplayHandlerSpec",
    "ReplayInputArtifact",
    "ReplayInvocation",
    "ReplayProducedArtifact",
    "ReplayRuntimeError",
    "ReplayRuntimeSpec",
    "create_evidence_export_handler",
    "execute_replay",
    "load_handlers",
    "load_replay_run",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
