from __future__ import annotations

import base64
import math
import re
from dataclasses import dataclass
from typing import Any

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class SegmentationRequest:
    server_id: str
    model_sha256: str
    captured_at: float
    frame_id: str
    width: int
    height: int
    encoding: str
    step: int
    queries: dict[str, str]
    image: bytes


@dataclass(frozen=True)
class SegmentationInstance:
    instance_id: str
    label: str
    confidence: float
    x_offset: int
    y_offset: int
    width: int
    height: int
    mask: bytes


def parse_request(
    raw: Any,
    *,
    server_id: str,
    model_sha256: str,
    allowed_labels: frozenset[str],
    max_image_bytes: int,
) -> SegmentationRequest:
    expected = {
        "schema_version",
        "server_id",
        "model_sha256",
        "captured_at",
        "frame_id",
        "width",
        "height",
        "encoding",
        "step",
        "queries",
        "image_base64",
    }
    if not isinstance(raw, dict) or set(raw) != expected or raw.get("schema_version") != 1:
        raise ProtocolError("request has missing or unexpected fields")
    if raw["server_id"] != server_id or raw["model_sha256"] != model_sha256:
        raise ProtocolError("request identity does not match the running service")
    captured_at = _finite_float(raw["captured_at"], "captured_at")
    if captured_at <= 0:
        raise ProtocolError("captured_at must be positive")
    frame_id = _identifier(raw["frame_id"], "frame_id")
    width = _integer(raw["width"], "width", 1, 16384)
    height = _integer(raw["height"], "height", 1, 16384)
    step = _integer(raw["step"], "step", 1, max_image_bytes)
    encoding = raw["encoding"]
    if not isinstance(encoding, str) or not 1 <= len(encoding) <= 64:
        raise ProtocolError("encoding is invalid")
    queries = raw["queries"]
    if (
        not isinstance(queries, dict)
        or not queries
        or len(queries) > 1024
        or any(not _IDENTIFIER.fullmatch(label) for label in queries)
        or any(
            not isinstance(prompt, str) or not 1 <= len(prompt) <= 512
            for prompt in queries.values()
        )
    ):
        raise ProtocolError("queries are invalid")
    unknown = sorted(set(queries) - allowed_labels)
    if unknown:
        raise ProtocolError("request contains labels not allowed by the service")
    try:
        image = base64.b64decode(raw["image_base64"], validate=True)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("image_base64 is invalid") from exc
    if len(image) != step * height or len(image) > max_image_bytes:
        raise ProtocolError("decoded image size is invalid")
    return SegmentationRequest(
        server_id=server_id,
        model_sha256=model_sha256,
        captured_at=captured_at,
        frame_id=frame_id,
        width=width,
        height=height,
        encoding=encoding,
        step=step,
        queries=dict(queries),
        image=image,
    )


def encode_response(
    instances: Any,
    request: SegmentationRequest,
    *,
    max_instances: int,
) -> dict[str, Any]:
    if not isinstance(instances, (list, tuple)) or len(instances) > max_instances:
        raise ProtocolError("provider returned an invalid instance collection")
    encoded: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for instance in instances:
        if not isinstance(instance, SegmentationInstance):
            raise ProtocolError("provider returned an invalid instance")
        instance_id = _identifier(instance.instance_id, "instance_id")
        label = _identifier(instance.label, "label")
        if instance_id in seen_ids:
            raise ProtocolError("provider returned duplicate instance IDs")
        if label not in request.queries:
            raise ProtocolError("provider returned an unrequested label")
        seen_ids.add(instance_id)
        confidence = _finite_float(instance.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ProtocolError("confidence must be in [0, 1]")
        x_offset = _integer(instance.x_offset, "x_offset", 0, request.width - 1)
        y_offset = _integer(instance.y_offset, "y_offset", 0, request.height - 1)
        width = _integer(instance.width, "width", 1, request.width)
        height = _integer(instance.height, "height", 1, request.height)
        if x_offset + width > request.width or y_offset + height > request.height:
            raise ProtocolError("provider mask ROI is outside the image")
        mask = bytes(instance.mask)
        if len(mask) != width * height or any(value not in (0, 1) for value in mask):
            raise ProtocolError("provider mask must be a binary ROI mask")
        encoded.append(
            {
                "instance_id": instance_id,
                "label": label,
                "confidence": confidence,
                "x_offset": x_offset,
                "y_offset": y_offset,
                "width": width,
                "height": height,
                "mask_base64": base64.b64encode(mask).decode("ascii"),
            }
        )
    return {
        "schema_version": 1,
        "server_id": request.server_id,
        "model_sha256": request.model_sha256,
        "instances": encoded,
    }


def validate_manifest(raw: Any) -> tuple[str, str, frozenset[str]]:
    expected = {"schema_version", "server_id", "model_sha256", "allowed_labels"}
    if not isinstance(raw, dict) or set(raw) != expected or raw.get("schema_version") != 1:
        raise ProtocolError("server manifest has missing or unexpected fields")
    server_id = _identifier(raw["server_id"], "server_id")
    model_sha256 = raw["model_sha256"]
    if not isinstance(model_sha256, str) or not _SHA256.fullmatch(model_sha256):
        raise ProtocolError("model_sha256 is invalid")
    labels = raw["allowed_labels"]
    if (
        not isinstance(labels, list)
        or not labels
        or len(labels) != len(set(labels))
        or any(not isinstance(label, str) or not _IDENTIFIER.fullmatch(label) for label in labels)
    ):
        raise ProtocolError("allowed_labels are invalid")
    return server_id, model_sha256, frozenset(labels)


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ProtocolError(f"{label} is invalid")
    return value


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ProtocolError(f"{label} must be in [{minimum}, {maximum}]")
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{label} must be finite")
    return result


__all__ = [
    "ProtocolError",
    "SegmentationInstance",
    "SegmentationRequest",
    "encode_response",
    "parse_request",
    "validate_manifest",
]
