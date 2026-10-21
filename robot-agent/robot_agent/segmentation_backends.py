from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

import yaml

ENTRY_POINT_GROUP = "xlerobot.segmentation_backends"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_DISTRIBUTION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class SegmentationBackendError(ValueError):
    pass


@dataclass(frozen=True)
class BackendArtifact:
    path: Path
    sha256: str


@dataclass(frozen=True)
class SegmentationBackendSpec:
    path: Path
    backend_id: str
    entry_point: str
    distribution: str
    distribution_version: str
    source_id: str
    image_topic: str
    output_topic: str
    camera_frame: str
    queries: dict[str, str]
    min_confidence: float
    max_instances: int
    max_image_bytes: int
    max_input_age_seconds: float
    max_inference_seconds: float
    artifacts: tuple[BackendArtifact, ...]
    options: dict[str, Any]
    fingerprint: str

    @classmethod
    def load(cls, path: str | Path) -> SegmentationBackendSpec:
        config_path = Path(path).expanduser().resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        expected = {
            "version",
            "approved",
            "backend_id",
            "entry_point",
            "distribution",
            "distribution_version",
            "source_id",
            "image_topic",
            "output_topic",
            "camera_frame",
            "queries",
            "min_confidence",
            "max_instances",
            "max_image_bytes",
            "max_input_age_seconds",
            "max_inference_seconds",
            "artifacts",
            "options",
        }
        if not isinstance(raw, dict) or set(raw) != expected or raw.get("version") != 1:
            raise SegmentationBackendError(
                "segmentation backend config has missing or unexpected fields"
            )
        if raw.get("approved") is not True:
            raise SegmentationBackendError("segmentation backend config must be approved")
        backend_id = _identifier(raw["backend_id"], "backend_id")
        entry_point = _identifier(raw["entry_point"], "entry_point")
        distribution = _distribution(raw["distribution"])
        distribution_version = _nonempty(raw["distribution_version"], "distribution_version")
        source_id = _source_id(raw["source_id"])
        image_topic = _absolute_topic(raw["image_topic"], "image_topic")
        output_topic = _absolute_topic(raw["output_topic"], "output_topic")
        if image_topic == output_topic:
            raise SegmentationBackendError("segmentation input and output topics must differ")
        camera_frame = _identifier(raw["camera_frame"], "camera_frame")
        queries = _queries(raw["queries"])
        min_confidence = _bounded_float(raw["min_confidence"], "min_confidence", 0.0, 1.0)
        max_instances = _bounded_integer(raw["max_instances"], "max_instances", 1, 1024)
        max_image_bytes = _bounded_integer(
            raw["max_image_bytes"], "max_image_bytes", 1024, 512 * 1024 * 1024
        )
        max_input_age = _bounded_float(
            raw["max_input_age_seconds"],
            "max_input_age_seconds",
            0.001,
            2.0,
        )
        max_inference = _bounded_float(
            raw["max_inference_seconds"],
            "max_inference_seconds",
            0.001,
            10.0,
        )
        artifacts = _artifacts(raw["artifacts"], config_path.parent)
        options = _json_object(raw["options"], "options")
        payload = {
            **raw,
            "artifacts": [
                {"path": str(artifact.path), "sha256": artifact.sha256}
                for artifact in artifacts
            ],
        }
        fingerprint = _digest(payload)
        return cls(
            path=config_path,
            backend_id=backend_id,
            entry_point=entry_point,
            distribution=distribution,
            distribution_version=distribution_version,
            source_id=source_id,
            image_topic=image_topic,
            output_topic=output_topic,
            camera_frame=camera_frame,
            queries=queries,
            min_confidence=min_confidence,
            max_instances=max_instances,
            max_image_bytes=max_image_bytes,
            max_input_age_seconds=max_input_age,
            max_inference_seconds=max_inference,
            artifacts=artifacts,
            options=options,
            fingerprint=fingerprint,
        )

    def validate_artifacts(self) -> None:
        for artifact in self.artifacts:
            if not artifact.path.is_file():
                raise SegmentationBackendError(
                    f"segmentation backend artifact does not exist: {artifact.path}"
                )
            if _sha256_file(artifact.path) != artifact.sha256:
                raise SegmentationBackendError(
                    f"segmentation backend artifact SHA-256 mismatch: {artifact.path}"
                )


@dataclass(frozen=True)
class SegmentationInput:
    width: int
    height: int
    encoding: str
    step: int
    data: bytes
    captured_at: float
    frame_id: str
    queries: dict[str, str]


@dataclass(frozen=True)
class MaskPrediction:
    instance_id: str
    label: str
    confidence: float
    x_offset: int
    y_offset: int
    width: int
    height: int
    mask: bytes


class SegmentationBackend(Protocol):
    def infer(
        self,
        request: SegmentationInput,
    ) -> list[MaskPrediction] | tuple[MaskPrediction, ...]: ...


def load_backend(
    spec: SegmentationBackendSpec,
    *,
    entry_points: Any | None = None,
) -> SegmentationBackend:
    spec.validate_artifacts()
    available = metadata.entry_points() if entry_points is None else entry_points
    if hasattr(available, "select"):
        candidates = list(available.select(group=ENTRY_POINT_GROUP, name=spec.entry_point))
    else:
        candidates = [
            entry
            for entry in available.get(ENTRY_POINT_GROUP, ())
            if entry.name == spec.entry_point
        ]
    if len(candidates) != 1:
        raise SegmentationBackendError(
            f"expected exactly one {ENTRY_POINT_GROUP}:{spec.entry_point} entry point"
        )
    entry = candidates[0]
    distribution = getattr(entry, "dist", None)
    name = getattr(distribution, "name", None)
    version = getattr(distribution, "version", None)
    if name is None and distribution is not None:
        name = distribution.metadata.get("Name")
    if _normalized_distribution(name) != _normalized_distribution(spec.distribution):
        raise SegmentationBackendError("segmentation backend distribution does not match config")
    if version != spec.distribution_version:
        raise SegmentationBackendError("segmentation backend version does not match config")
    factory = entry.load()
    if not callable(factory):
        raise SegmentationBackendError("segmentation backend entry point must be a factory")
    backend = factory(
        options=dict(spec.options),
        artifacts=tuple(artifact.path for artifact in spec.artifacts),
    )
    if not callable(getattr(backend, "infer", None)):
        raise SegmentationBackendError("segmentation backend must provide infer(request)")
    return backend


def image_request_from_ros(
    message: Any,
    spec: SegmentationBackendSpec,
    *,
    captured_at: float,
) -> SegmentationInput:
    width = int(message.width)
    height = int(message.height)
    step = int(message.step)
    if not 1 <= width <= 16384 or not 1 <= height <= 16384:
        raise SegmentationBackendError("segmentation input image dimensions are invalid")
    if step <= 0:
        raise SegmentationBackendError("segmentation input image step must be positive")
    data = bytes(message.data)
    if len(data) != step * height or len(data) > spec.max_image_bytes:
        raise SegmentationBackendError("segmentation input image byte size is invalid")
    encoding = _nonempty(str(message.encoding), "image encoding")
    frame_id = _identifier(str(message.header.frame_id), "image frame_id")
    if frame_id != spec.camera_frame:
        raise SegmentationBackendError("segmentation input frame does not match camera_frame")
    if not math.isfinite(captured_at) or captured_at <= 0:
        raise SegmentationBackendError("segmentation input timestamp is invalid")
    return SegmentationInput(
        width=width,
        height=height,
        encoding=encoding,
        step=step,
        data=data,
        captured_at=captured_at,
        frame_id=frame_id,
        queries=dict(spec.queries),
    )


def validate_predictions(
    predictions: Any,
    request: SegmentationInput,
    spec: SegmentationBackendSpec,
) -> tuple[MaskPrediction, ...]:
    if not isinstance(predictions, (list, tuple)):
        raise SegmentationBackendError("segmentation backend output must be a list or tuple")
    if len(predictions) > 4096:
        raise SegmentationBackendError("segmentation backend returned too many raw instances")
    accepted: list[MaskPrediction] = []
    seen_ids: set[str] = set()
    for prediction in predictions:
        if not isinstance(prediction, MaskPrediction):
            raise SegmentationBackendError("segmentation backend returned an invalid instance")
        instance_id = _identifier(prediction.instance_id, "prediction instance_id")
        label = _identifier(prediction.label, "prediction label")
        if label not in spec.queries:
            raise SegmentationBackendError(
                f"segmentation backend returned undeclared label {label!r}"
            )
        if instance_id in seen_ids:
            raise SegmentationBackendError("segmentation backend returned duplicate instance IDs")
        seen_ids.add(instance_id)
        confidence = _bounded_float(prediction.confidence, "prediction confidence", 0.0, 1.0)
        bounds = (
            prediction.x_offset,
            prediction.y_offset,
            prediction.width,
            prediction.height,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in bounds):
            raise SegmentationBackendError("segmentation prediction ROI must contain integers")
        if (
            prediction.x_offset < 0
            or prediction.y_offset < 0
            or prediction.width <= 0
            or prediction.height <= 0
            or prediction.x_offset + prediction.width > request.width
            or prediction.y_offset + prediction.height > request.height
        ):
            raise SegmentationBackendError("segmentation prediction ROI is outside the image")
        mask = bytes(prediction.mask)
        if len(mask) != prediction.width * prediction.height or any(
            value not in (0, 1) for value in mask
        ):
            raise SegmentationBackendError("segmentation prediction mask must be binary ROI data")
        if confidence >= spec.min_confidence:
            accepted.append(prediction)
    if len(accepted) > spec.max_instances:
        raise SegmentationBackendError("segmentation backend exceeds reviewed instance limit")
    return tuple(accepted)


def run_inference(
    backend: SegmentationBackend,
    request: SegmentationInput,
    spec: SegmentationBackendSpec,
    *,
    clock: Any = time.monotonic,
) -> tuple[MaskPrediction, ...]:
    started = float(clock())
    try:
        predictions = backend.infer(request)
    except Exception as exc:
        raise SegmentationBackendError("segmentation backend inference failed") from exc
    elapsed = float(clock()) - started
    if not math.isfinite(elapsed) or elapsed < 0:
        raise SegmentationBackendError("segmentation backend clock is invalid")
    if elapsed > spec.max_inference_seconds:
        raise SegmentationBackendError(
            "segmentation backend exceeded reviewed inference deadline"
        )
    return validate_predictions(predictions, request, spec)


def _artifacts(raw: Any, root: Path) -> tuple[BackendArtifact, ...]:
    if not isinstance(raw, list) or not raw:
        raise SegmentationBackendError("segmentation backend artifacts must be non-empty")
    output: list[BackendArtifact] = []
    seen: set[Path] = set()
    for index, value in enumerate(raw):
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            raise SegmentationBackendError(f"artifact {index} must contain path and sha256")
        configured = _nonempty(value["path"], f"artifact {index} path")
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if path in seen:
            raise SegmentationBackendError("segmentation backend artifacts contain duplicates")
        seen.add(path)
        output.append(BackendArtifact(path, _sha(value["sha256"], f"artifact {index} sha256")))
    return tuple(output)


def _queries(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict) or not raw:
        raise SegmentationBackendError("segmentation backend queries must be non-empty")
    if len(raw) > 1024:
        raise SegmentationBackendError("segmentation backend has too many queries")
    return {
        _identifier(label, "query label"): _nonempty(prompt, f"query {label}")
        for label, prompt in raw.items()
    }


def _json_object(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise SegmentationBackendError(f"{label} must be an object")
    try:
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SegmentationBackendError(f"{label} must contain JSON values") from exc
    if len(encoded.encode("utf-8")) > 64 * 1024:
        raise SegmentationBackendError(f"{label} exceeds 64 KiB")
    return json.loads(encoded)


def _bounded_float(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SegmentationBackendError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise SegmentationBackendError(f"{label} must be in [{minimum}, {maximum}]")
    return result


def _bounded_integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise SegmentationBackendError(f"{label} must be in [{minimum}, {maximum}]")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise SegmentationBackendError(f"{label} is not a valid identifier")
    return value


def _distribution(value: Any) -> str:
    if not isinstance(value, str) or not _DISTRIBUTION.fullmatch(value):
        raise SegmentationBackendError("distribution is invalid")
    return value


def _normalized_distribution(value: Any) -> str:
    return re.sub(r"[-_.]+", "-", str(value or "")).lower()


def _source_id(value: Any) -> str:
    source = _nonempty(value, "source_id")
    if len(source) > 256 or source.startswith("/") or source.endswith("/"):
        raise SegmentationBackendError("source_id must be a bounded relative source path")
    parts = source.split("/")
    if not all(_IDENTIFIER.fullmatch(part) for part in parts):
        raise SegmentationBackendError("source_id contains invalid path components")
    return source


def _absolute_topic(value: Any, label: str) -> str:
    topic = _nonempty(value, label)
    if not topic.startswith("/") or any(character.isspace() for character in topic):
        raise SegmentationBackendError(f"{label} must be an absolute ROS name")
    return topic


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SegmentationBackendError(f"{label} must be non-empty")
    return value.strip()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise SegmentationBackendError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "BackendArtifact",
    "ENTRY_POINT_GROUP",
    "MaskPrediction",
    "SegmentationBackend",
    "SegmentationBackendError",
    "SegmentationBackendSpec",
    "SegmentationInput",
    "image_request_from_ros",
    "load_backend",
    "run_inference",
    "validate_predictions",
]
