from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from robot_agent.segmentation_backends import MaskPrediction, SegmentationInput


class HttpBackendError(ValueError):
    pass


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class LocalHttpSegmentationBackend:
    def __init__(
        self,
        *,
        options: dict[str, Any],
        artifacts: tuple[Path, ...],
        opener: Any | None = None,
    ):
        expected = {
            "endpoint",
            "timeout_seconds",
            "max_response_bytes",
            "manifest_artifact_index",
            "token_artifact_index",
        }
        if not isinstance(options, dict) or set(options) != expected:
            raise HttpBackendError("HTTP backend options have missing or unexpected fields")
        self.endpoint = _loopback_endpoint(options["endpoint"])
        self.timeout_seconds = _bounded_float(
            options["timeout_seconds"], "timeout_seconds", 0.01, 10.0
        )
        self.max_response_bytes = _bounded_integer(
            options["max_response_bytes"], "max_response_bytes", 1024, 64 * 1024 * 1024
        )
        manifest_index = _artifact_index(
            options["manifest_artifact_index"], artifacts, "manifest_artifact_index"
        )
        token_index = options["token_artifact_index"]
        if token_index is not None:
            token_index = _artifact_index(token_index, artifacts, "token_artifact_index")
            if token_index == manifest_index:
                raise HttpBackendError("manifest and token artifacts must differ")
        self.manifest = _load_manifest(artifacts[manifest_index])
        self.token = _load_token(artifacts[token_index]) if token_index is not None else None
        self.opener = opener or request.build_opener(_NoRedirect())

    def infer(self, segmentation_input: SegmentationInput) -> list[MaskPrediction]:
        unknown_labels = sorted(
            set(segmentation_input.queries) - set(self.manifest["allowed_labels"])
        )
        if unknown_labels:
            raise HttpBackendError(
                "HTTP segmentation manifest does not allow labels: " + ", ".join(unknown_labels)
            )
        payload = {
            "schema_version": 1,
            "server_id": self.manifest["server_id"],
            "model_sha256": self.manifest["model_sha256"],
            "captured_at": segmentation_input.captured_at,
            "frame_id": segmentation_input.frame_id,
            "width": segmentation_input.width,
            "height": segmentation_input.height,
            "encoding": segmentation_input.encoding,
            "step": segmentation_input.step,
            "queries": segmentation_input.queries,
            "image_base64": base64.b64encode(segmentation_input.data).decode("ascii"),
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        outbound = request.Request(
            self.endpoint,
            data=json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with self.opener.open(outbound, timeout=self.timeout_seconds) as response:
                status = int(getattr(response, "status", response.getcode()))
                if status != 200:
                    raise HttpBackendError(f"HTTP segmentation server returned status {status}")
                content_type = str(response.headers.get("Content-Type", "")).split(";", 1)[0]
                if content_type.lower() != "application/json":
                    raise HttpBackendError("HTTP segmentation response must be application/json")
                body = response.read(self.max_response_bytes + 1)
        except (error.HTTPError, error.URLError, TimeoutError, OSError) as exc:
            raise HttpBackendError("HTTP segmentation request failed") from exc
        if len(body) > self.max_response_bytes:
            raise HttpBackendError("HTTP segmentation response exceeds reviewed size")
        try:
            raw = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HttpBackendError("HTTP segmentation response JSON is invalid") from exc
        return _parse_response(raw, self.manifest, segmentation_input)


def create_backend(*, options: dict[str, Any], artifacts: tuple[Path, ...]):
    return LocalHttpSegmentationBackend(options=options, artifacts=artifacts)


def _parse_response(
    raw: Any,
    manifest: dict[str, Any],
    segmentation_input: SegmentationInput,
) -> list[MaskPrediction]:
    expected = {"schema_version", "server_id", "model_sha256", "instances"}
    if not isinstance(raw, dict) or set(raw) != expected or raw.get("schema_version") != 1:
        raise HttpBackendError("HTTP segmentation response has missing or unexpected fields")
    if raw.get("server_id") != manifest["server_id"]:
        raise HttpBackendError("HTTP segmentation server identity does not match manifest")
    if raw.get("model_sha256") != manifest["model_sha256"]:
        raise HttpBackendError("HTTP segmentation model digest does not match manifest")
    instances = raw.get("instances")
    if not isinstance(instances, list) or len(instances) > 4096:
        raise HttpBackendError("HTTP segmentation instances are invalid")
    output: list[MaskPrediction] = []
    expected_fields = {
        "instance_id",
        "label",
        "confidence",
        "x_offset",
        "y_offset",
        "width",
        "height",
        "mask_base64",
    }
    for index, instance in enumerate(instances):
        if not isinstance(instance, dict) or set(instance) != expected_fields:
            raise HttpBackendError(f"HTTP segmentation instance {index} is invalid")
        if instance["label"] not in segmentation_input.queries:
            raise HttpBackendError("HTTP segmentation response contains an unrequested label")
        try:
            mask = base64.b64decode(instance["mask_base64"], validate=True)
        except (TypeError, ValueError) as exc:
            raise HttpBackendError("HTTP segmentation mask base64 is invalid") from exc
        output.append(
            MaskPrediction(
                instance_id=instance["instance_id"],
                label=instance["label"],
                confidence=instance["confidence"],
                x_offset=instance["x_offset"],
                y_offset=instance["y_offset"],
                width=instance["width"],
                height=instance["height"],
                mask=mask,
            )
        )
    return output


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HttpBackendError("HTTP segmentation manifest is invalid") from exc
    expected = {"schema_version", "server_id", "model_sha256", "allowed_labels"}
    if not isinstance(raw, dict) or set(raw) != expected or raw.get("schema_version") != 1:
        raise HttpBackendError("HTTP segmentation manifest has missing or unexpected fields")
    if not _identifier(raw["server_id"]):
        raise HttpBackendError("HTTP segmentation server_id is invalid")
    if not _sha(raw["model_sha256"]):
        raise HttpBackendError("HTTP segmentation model_sha256 is invalid")
    labels = raw["allowed_labels"]
    if (
        not isinstance(labels, list)
        or not labels
        or len(labels) != len(set(labels))
        or not all(_identifier(label) for label in labels)
    ):
        raise HttpBackendError("HTTP segmentation allowed_labels are invalid")
    return raw


def _load_token(path: Path) -> str:
    try:
        token = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise HttpBackendError("HTTP segmentation token cannot be read") from exc
    if not 16 <= len(token) <= 4096 or any(character.isspace() for character in token):
        raise HttpBackendError("HTTP segmentation token is invalid")
    return token


def _loopback_endpoint(value: Any) -> str:
    if not isinstance(value, str):
        raise HttpBackendError("HTTP segmentation endpoint must be a string")
    parsed = parse.urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != "/v1/segment"
        or parsed.port is None
    ):
        raise HttpBackendError(
            "HTTP segmentation endpoint must be loopback http://host:port/v1/segment"
        )
    return value


def _artifact_index(value: Any, artifacts: tuple[Path, ...], label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(artifacts):
        raise HttpBackendError(f"{label} is outside the artifact list")
    return value


def _bounded_float(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HttpBackendError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise HttpBackendError(f"{label} must be in [{minimum}, {maximum}]")
    return result


def _bounded_integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise HttpBackendError(f"{label} must be in [{minimum}, {maximum}]")
    return value


def _identifier(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 128
        and value[0].isalnum()
        and all(character.isalnum() or character in "_.-" for character in value)
    )


def _sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = ["HttpBackendError", "LocalHttpSegmentationBackend", "create_backend"]
