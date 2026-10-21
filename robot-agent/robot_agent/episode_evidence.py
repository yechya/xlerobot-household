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
from pathlib import Path
from typing import Any

import yaml

from .episodes import EpisodeManifest, load_episode

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_MIME_TYPE = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]{0,63}/[a-z0-9][a-z0-9!#$&^_.+-]{0,63}$")
_SUFFIX = re.compile(r"^\.[A-Za-z0-9]{1,12}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")


class EpisodeEvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class EvidenceSource:
    artifact_id: str
    kind: str
    source: Path
    mime_type: str
    captured_at_start: float
    captured_at_end: float


@dataclass(frozen=True)
class EvidenceIndexSpec:
    episode_id: str
    max_total_bytes: int
    max_time_margin_seconds: float
    artifacts: tuple[EvidenceSource, ...]

    @classmethod
    def load(cls, path: str | Path) -> EvidenceIndexSpec:
        config_path = Path(path).expanduser().resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        expected = {
            "version",
            "approved",
            "episode_id",
            "max_total_bytes",
            "max_time_margin_seconds",
            "artifacts",
        }
        if not isinstance(raw, dict) or set(raw) != expected or raw.get("version") != 1:
            raise EpisodeEvidenceError("evidence index spec has missing or unexpected fields")
        if raw["approved"] is not True:
            raise EpisodeEvidenceError("evidence indexing must be approved")
        values = raw["artifacts"]
        if not isinstance(values, list) or not 1 <= len(values) <= 128:
            raise EpisodeEvidenceError("evidence spec must contain 1..128 artifacts")
        artifacts = tuple(
            _parse_source(value, config_path.parent, index)
            for index, value in enumerate(values)
        )
        identifiers = [artifact.artifact_id for artifact in artifacts]
        if len(identifiers) != len(set(identifiers)):
            raise EpisodeEvidenceError("evidence artifact IDs must be unique")
        return cls(
            episode_id=_identifier(raw["episode_id"], "episode_id"),
            max_total_bytes=_integer(
                raw["max_total_bytes"], "max_total_bytes", 1, 1024 * 1024 * 1024 * 1024
            ),
            max_time_margin_seconds=_bounded_float(
                raw["max_time_margin_seconds"], "max_time_margin_seconds", 0.0, 60.0
            ),
            artifacts=artifacts,
        )


@dataclass(frozen=True)
class EvidenceArtifact:
    artifact_id: str
    kind: str
    relative_path: str
    mime_type: str
    size_bytes: int
    sha256: str
    captured_at_start: float
    captured_at_end: float

    @classmethod
    def from_dict(cls, raw: Any, index: int) -> EvidenceArtifact:
        expected = {
            "artifact_id",
            "kind",
            "relative_path",
            "mime_type",
            "size_bytes",
            "sha256",
            "captured_at_start",
            "captured_at_end",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise EpisodeEvidenceError(f"evidence artifact {index} has invalid fields")
        artifact_id = _identifier(raw["artifact_id"], "artifact_id")
        kind = _identifier(raw["kind"], "artifact kind")
        relative_path = raw["relative_path"]
        if (
            not isinstance(relative_path, str)
            or Path(relative_path).is_absolute()
            or Path(relative_path).parts[:1] != ("files",)
            or ".." in Path(relative_path).parts
        ):
            raise EpisodeEvidenceError("evidence relative path is invalid")
        return cls(
            artifact_id=artifact_id,
            kind=kind,
            relative_path=relative_path,
            mime_type=_mime_type(raw["mime_type"]),
            size_bytes=_integer(raw["size_bytes"], "artifact size", 1, 1024**4),
            sha256=_sha(raw["sha256"], "artifact sha256"),
            captured_at_start=_positive_float(raw["captured_at_start"], "captured_at_start"),
            captured_at_end=_positive_float(raw["captured_at_end"], "captured_at_end"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "relative_path": self.relative_path,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "captured_at_start": self.captured_at_start,
            "captured_at_end": self.captured_at_end,
        }


@dataclass(frozen=True)
class EpisodeEvidenceManifest:
    episode_id: str
    episode_manifest_sha256: str
    created_at: float
    artifacts: tuple[EvidenceArtifact, ...]
    evidence_sha256: str

    @classmethod
    def create(
        cls,
        episode: EpisodeManifest,
        artifacts: tuple[EvidenceArtifact, ...],
        *,
        created_at: float | None = None,
    ) -> EpisodeEvidenceManifest:
        manifest = cls(
            episode_id=episode.episode_id,
            episode_manifest_sha256=episode.manifest_sha256,
            created_at=time.time() if created_at is None else created_at,
            artifacts=artifacts,
            evidence_sha256="",
        )
        manifest.validate(include_hash=False)
        return cls(**{**manifest.__dict__, "evidence_sha256": _digest(manifest._hash_payload())})

    @classmethod
    def from_dict(cls, raw: Any) -> EpisodeEvidenceManifest:
        expected = {
            "schema_version",
            "episode_id",
            "episode_manifest_sha256",
            "created_at",
            "artifacts",
            "evidence_sha256",
        }
        if not isinstance(raw, dict) or set(raw) != expected or raw.get("schema_version") != 1:
            raise EpisodeEvidenceError("episode evidence manifest has invalid fields")
        values = raw["artifacts"]
        if not isinstance(values, list) or not 1 <= len(values) <= 128:
            raise EpisodeEvidenceError("episode evidence artifact list is invalid")
        manifest = cls(
            episode_id=raw["episode_id"],
            episode_manifest_sha256=raw["episode_manifest_sha256"],
            created_at=raw["created_at"],
            artifacts=tuple(
                EvidenceArtifact.from_dict(value, index) for index, value in enumerate(values)
            ),
            evidence_sha256=raw["evidence_sha256"],
        )
        manifest.validate(include_hash=True)
        return manifest

    def validate(self, *, include_hash: bool) -> None:
        _identifier(self.episode_id, "episode_id")
        _sha(self.episode_manifest_sha256, "episode manifest sha256")
        _positive_float(self.created_at, "evidence created_at")
        identifiers = [artifact.artifact_id for artifact in self.artifacts]
        paths = [artifact.relative_path for artifact in self.artifacts]
        if len(identifiers) != len(set(identifiers)) or len(paths) != len(set(paths)):
            raise EpisodeEvidenceError("evidence artifact IDs and paths must be unique")
        for artifact in self.artifacts:
            if artifact.captured_at_end < artifact.captured_at_start:
                raise EpisodeEvidenceError("evidence artifact end precedes start")
        if include_hash:
            _sha(self.evidence_sha256, "evidence sha256")
            if self.evidence_sha256 != _digest(self._hash_payload()):
                raise EpisodeEvidenceError("episode evidence hash does not match content")

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "episode_id": self.episode_id,
            "episode_manifest_sha256": self.episode_manifest_sha256,
            "created_at": self.created_at,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "evidence_sha256": self.evidence_sha256}


def index_episode_evidence(
    episode_path: str | Path,
    spec: EvidenceIndexSpec,
    *,
    created_at: float | None = None,
) -> EpisodeEvidenceManifest:
    episode_dir = Path(episode_path).expanduser().resolve()
    episode = load_episode(episode_dir)
    if spec.episode_id != episode.episode_id:
        raise EpisodeEvidenceError("evidence spec episode_id does not match episode")
    margin = spec.max_time_margin_seconds
    total = 0
    for artifact in spec.artifacts:
        _validate_source_file(artifact.source)
        size = artifact.source.stat().st_size
        total += size
        if total > spec.max_total_bytes:
            raise EpisodeEvidenceError("evidence artifacts exceed max_total_bytes")
        if (
            artifact.captured_at_start < episode.started_at - margin
            or artifact.captured_at_end > episode.ended_at + margin
        ):
            raise EpisodeEvidenceError("evidence artifact timestamp is outside episode bounds")
    final = episode_dir / "evidence"
    staging = episode_dir / f".evidence.{uuid.uuid4().hex}.tmp"
    if final.exists():
        raise EpisodeEvidenceError("episode evidence already exists")
    try:
        staging.mkdir(mode=0o700)
        files = staging / "files"
        files.mkdir(mode=0o700)
        indexed: list[EvidenceArtifact] = []
        for source in spec.artifacts:
            suffix = source.source.suffix if _SUFFIX.fullmatch(source.source.suffix) else ""
            relative = Path("files") / f"{source.artifact_id}{suffix.lower()}"
            destination = staging / relative
            _copy_regular_file(source.source, destination)
            indexed.append(
                EvidenceArtifact(
                    artifact_id=source.artifact_id,
                    kind=source.kind,
                    relative_path=relative.as_posix(),
                    mime_type=source.mime_type,
                    size_bytes=destination.stat().st_size,
                    sha256=_sha256_file(destination),
                    captured_at_start=source.captured_at_start,
                    captured_at_end=source.captured_at_end,
                )
            )
        manifest = EpisodeEvidenceManifest.create(
            episode, tuple(indexed), created_at=created_at
        )
        _write_json_exclusive(staging / "manifest.json", manifest.to_dict())
        _fsync_directory(files)
        _fsync_directory(staging)
        staging.replace(final)
        _fsync_directory(episode_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_episode_evidence(
    episode_path: str | Path,
    *,
    required_kinds: tuple[str, ...] = (),
) -> EpisodeEvidenceManifest:
    episode_dir = Path(episode_path).expanduser().resolve()
    episode = load_episode(episode_dir)
    evidence_dir = episode_dir / "evidence"
    try:
        raw = json.loads((evidence_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EpisodeEvidenceError("cannot load episode evidence manifest") from exc
    manifest = EpisodeEvidenceManifest.from_dict(raw)
    if manifest.episode_id != episode.episode_id:
        raise EpisodeEvidenceError("evidence episode_id does not match episode")
    if manifest.episode_manifest_sha256 != episode.manifest_sha256:
        raise EpisodeEvidenceError("evidence does not bind current episode manifest")
    for artifact in manifest.artifacts:
        path = evidence_dir / artifact.relative_path
        _validate_source_file(path)
        if path.stat().st_size != artifact.size_bytes or _sha256_file(path) != artifact.sha256:
            raise EpisodeEvidenceError(f"evidence artifact was modified: {artifact.artifact_id}")
    available = {artifact.kind for artifact in manifest.artifacts}
    missing = sorted(set(required_kinds) - available)
    if missing:
        raise EpisodeEvidenceError(
            "episode evidence is missing required kinds: " + ", ".join(missing)
        )
    return manifest


def load_evidence_store(
    root: str | Path,
    *,
    required_kinds: tuple[str, ...] = (),
) -> list[tuple[EpisodeManifest, EpisodeEvidenceManifest]]:
    records = []
    for manifest_path in sorted(Path(root).glob("*/manifest.json")):
        episode_dir = manifest_path.parent
        episode = load_episode(episode_dir)
        evidence = load_episode_evidence(episode_dir, required_kinds=required_kinds)
        records.append((episode, evidence))
    return records


def _parse_source(raw: Any, base: Path, index: int) -> EvidenceSource:
    expected = {
        "artifact_id",
        "kind",
        "source",
        "mime_type",
        "captured_at_start",
        "captured_at_end",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise EpisodeEvidenceError(f"evidence source {index} has invalid fields")
    source = raw["source"]
    if not isinstance(source, str) or not source:
        raise EpisodeEvidenceError("evidence source path is invalid")
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = base / path
    started = _positive_float(raw["captured_at_start"], "captured_at_start")
    ended = _positive_float(raw["captured_at_end"], "captured_at_end")
    if ended < started:
        raise EpisodeEvidenceError("evidence source end precedes start")
    return EvidenceSource(
        artifact_id=_identifier(raw["artifact_id"], "artifact_id"),
        kind=_identifier(raw["kind"], "artifact kind"),
        source=Path(os.path.abspath(path)),
        mime_type=_mime_type(raw["mime_type"]),
        captured_at_start=started,
        captured_at_end=ended,
    )


def _copy_regular_file(source: Path, destination: Path) -> None:
    source_descriptor = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if not stat.S_ISREG(os.fstat(source_descriptor).st_mode):
            raise EpisodeEvidenceError("evidence source must remain a regular file")
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(source_descriptor, "rb") as incoming, os.fdopen(
            descriptor, "wb"
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


def _validate_source_file(path: Path) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise EpisodeEvidenceError(f"evidence artifact cannot be read: {path}") from exc
    if not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_size <= 0:
        raise EpisodeEvidenceError(f"evidence artifact must be a non-empty regular file: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise EpisodeEvidenceError(f"{label} is not a safe identifier")
    return value


def _mime_type(value: Any) -> str:
    if not isinstance(value, str) or not _MIME_TYPE.fullmatch(value):
        raise EpisodeEvidenceError("evidence mime_type is invalid")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise EpisodeEvidenceError(f"{label} must be a SHA-256 digest")
    return value


def _positive_float(value: Any, label: str) -> float:
    result = _bounded_float(value, label, 0.0, float("inf"))
    if result <= 0:
        raise EpisodeEvidenceError(f"{label} must be positive")
    return result


def _bounded_float(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EpisodeEvidenceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise EpisodeEvidenceError(f"{label} must be in [{minimum}, {maximum}]")
    return result


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise EpisodeEvidenceError(f"{label} must be in [{minimum}, {maximum}]")
    return value


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index and validate immutable episode evidence")
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_parser = subparsers.add_parser("index")
    index_parser.add_argument("--episode", required=True)
    index_parser.add_argument("--spec", required=True)
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--episodes", required=True)
    check_parser.add_argument("--require-kind", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        if args.command == "index":
            manifest = index_episode_evidence(
                args.episode, EvidenceIndexSpec.load(args.spec)
            )
            payload = manifest.to_dict()
        else:
            records = load_evidence_store(
                args.episodes,
                required_kinds=tuple(args.require_kind),
            )
            payload = {
                "valid": True,
                "count": len(records),
                "episode_ids": [episode.episode_id for episode, _ in records],
                "evidence_sha256": [evidence.evidence_sha256 for _, evidence in records],
            }
    except (EpisodeEvidenceError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


__all__ = [
    "EpisodeEvidenceError",
    "EpisodeEvidenceManifest",
    "EvidenceArtifact",
    "EvidenceIndexSpec",
    "EvidenceSource",
    "index_episode_evidence",
    "load_episode_evidence",
    "load_evidence_store",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
