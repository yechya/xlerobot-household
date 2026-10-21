from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .protocol import ProtocolError, validate_manifest


class ServerConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Artifact:
    path: Path
    sha256: str


@dataclass(frozen=True)
class ServerConfig:
    path: Path
    bind_host: str
    port: int
    server_id: str
    model_sha256: str
    allowed_labels: frozenset[str]
    provider_entry_point: str
    provider_distribution: str
    provider_version: str
    provider_artifacts: tuple[Artifact, ...]
    provider_options: dict[str, Any]
    token: str | None
    max_request_bytes: int
    max_image_bytes: int
    max_response_bytes: int
    max_instances: int
    max_inference_seconds: float

    @classmethod
    def load(cls, path: str | Path) -> ServerConfig:
        config_path = Path(path).expanduser().resolve()
        raw = _read_json(config_path, "server config")
        expected = {
            "schema_version",
            "approved",
            "bind_host",
            "port",
            "manifest",
            "provider",
            "model_artifact_index",
            "token",
            "max_request_bytes",
            "max_image_bytes",
            "max_response_bytes",
            "max_instances",
            "max_inference_seconds",
        }
        if not isinstance(raw, dict) or set(raw) != expected or raw.get("schema_version") != 1:
            raise ServerConfigError("server config has missing or unexpected fields")
        if raw["approved"] is not True:
            raise ServerConfigError("server config must be approved")
        if raw["bind_host"] not in {"127.0.0.1", "::1"}:
            raise ServerConfigError("server must bind to a numeric loopback address")
        port = _integer(raw["port"], "port", 1, 65535)
        manifest_artifact = _artifact(raw["manifest"], config_path.parent, "manifest")
        _validate_artifact(manifest_artifact)
        try:
            server_id, model_sha256, allowed_labels = validate_manifest(
                _read_json(manifest_artifact.path, "server manifest")
            )
        except ProtocolError as exc:
            raise ServerConfigError(str(exc)) from exc
        provider = raw["provider"]
        provider_expected = {"entry_point", "distribution", "version", "artifacts", "options"}
        if not isinstance(provider, dict) or set(provider) != provider_expected:
            raise ServerConfigError("provider config has missing or unexpected fields")
        artifacts_raw = provider["artifacts"]
        if not isinstance(artifacts_raw, list) or not artifacts_raw or len(artifacts_raw) > 128:
            raise ServerConfigError("provider artifacts are invalid")
        artifacts = tuple(
            _artifact(value, config_path.parent, f"provider artifact {index}")
            for index, value in enumerate(artifacts_raw)
        )
        for artifact in artifacts:
            _validate_artifact(artifact)
        model_index = _integer(
            raw["model_artifact_index"], "model_artifact_index", 0, len(artifacts) - 1
        )
        if artifacts[model_index].sha256 != model_sha256:
            raise ServerConfigError("model artifact digest does not match server manifest")
        options = provider["options"]
        try:
            json.dumps(options, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ServerConfigError("provider options must be bounded JSON") from exc
        if not isinstance(options, dict) or len(json.dumps(options)) > 65536:
            raise ServerConfigError("provider options must be a bounded JSON object")
        token = _load_token(raw["token"], config_path.parent)
        max_request_bytes = _integer(
            raw["max_request_bytes"], "max_request_bytes", 1024, 512 * 1024 * 1024
        )
        max_image_bytes = _integer(
            raw["max_image_bytes"], "max_image_bytes", 1024, 512 * 1024 * 1024
        )
        if max_image_bytes > max_request_bytes:
            raise ServerConfigError("max_image_bytes cannot exceed max_request_bytes")
        return cls(
            path=config_path,
            bind_host=raw["bind_host"],
            port=port,
            server_id=server_id,
            model_sha256=model_sha256,
            allowed_labels=allowed_labels,
            provider_entry_point=_text(provider["entry_point"], "provider entry_point"),
            provider_distribution=_text(provider["distribution"], "provider distribution"),
            provider_version=_text(provider["version"], "provider version"),
            provider_artifacts=artifacts,
            provider_options=dict(options),
            token=token,
            max_request_bytes=max_request_bytes,
            max_image_bytes=max_image_bytes,
            max_response_bytes=_integer(
                raw["max_response_bytes"], "max_response_bytes", 1024, 512 * 1024 * 1024
            ),
            max_instances=_integer(raw["max_instances"], "max_instances", 1, 4096),
            max_inference_seconds=_finite_float(
                raw["max_inference_seconds"], "max_inference_seconds", 0.001, 60.0
            ),
        )


def _load_token(raw: Any, base: Path) -> str | None:
    if raw is None:
        return None
    artifact = _artifact(raw, base, "token")
    _validate_artifact(artifact)
    try:
        token = artifact.path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise ServerConfigError("token cannot be read") from exc
    if not 16 <= len(token) <= 4096 or any(character.isspace() for character in token):
        raise ServerConfigError("token is invalid")
    return token


def _artifact(raw: Any, base: Path, label: str) -> Artifact:
    if not isinstance(raw, dict) or set(raw) != {"path", "sha256"}:
        raise ServerConfigError(f"{label} is invalid")
    path = raw["path"]
    digest = raw["sha256"]
    if not isinstance(path, str) or not path or not isinstance(digest, str) or len(digest) != 64:
        raise ServerConfigError(f"{label} is invalid")
    if any(character not in "0123456789abcdef" for character in digest):
        raise ServerConfigError(f"{label} digest is invalid")
    return Artifact(path=(base / path).resolve(), sha256=digest)


def _validate_artifact(artifact: Artifact) -> None:
    if not artifact.path.is_file():
        raise ServerConfigError(f"artifact does not exist: {artifact.path}")
    digest = hashlib.sha256()
    with artifact.path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != artifact.sha256:
        raise ServerConfigError(f"artifact digest mismatch: {artifact.path}")


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ServerConfigError(f"{label} is invalid") from exc


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 128:
        raise ServerConfigError(f"{label} is invalid")
    return value


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ServerConfigError(f"{label} must be in [{minimum}, {maximum}]")
    return value


def _finite_float(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ServerConfigError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ServerConfigError(f"{label} must be in [{minimum}, {maximum}]")
    return result


__all__ = ["Artifact", "ServerConfig", "ServerConfigError"]
