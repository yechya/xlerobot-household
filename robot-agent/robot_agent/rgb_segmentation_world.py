from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any

from .world_graph import WorldEvent, WorldGraphError
from .world_model import ObjectState, WorldModel


class RgbSegmentationWorldError(ValueError):
    pass


@dataclass(frozen=True)
class RgbSegmentationFusion:
    observation_id: str
    fusion_fingerprint: str
    frame_id: str
    objects: tuple[ObjectState, ...]
    event: WorldEvent | None


def fuse_segmentation_frame(
    world_model: WorldModel,
    frame: Any,
    *,
    captured_at: float,
    source: str = "rgb_segmentation",
    observation_id: str | None = None,
) -> RgbSegmentationFusion:
    """Convert one RGB segmentation frame into qualitative world-model objects."""

    image_width = _positive_int(getattr(frame, "image_width", None), "image_width")
    image_height = _positive_int(getattr(frame, "image_height", None), "image_height")
    if not math.isfinite(captured_at) or captured_at <= 0:
        raise RgbSegmentationWorldError("captured_at must be positive and finite")
    source = _text(source, "source", max_length=128)
    header = getattr(frame, "header", None)
    frame_id = _text(getattr(header, "frame_id", ""), "frame_id", max_length=128)
    instances = getattr(frame, "instances", None)
    if not isinstance(instances, (list, tuple)):
        raise RgbSegmentationWorldError("segmentation frame instances must be a list")
    if len(instances) > 128:
        raise RgbSegmentationWorldError("segmentation frame contains too many instances")

    parsed: list[dict[str, Any]] = []
    objects: list[ObjectState] = []
    seen_ids: set[str] = set()
    for instance in instances:
        instance_id = _text(getattr(instance, "instance_id", ""), "instance_id")
        label = _text(getattr(instance, "label", ""), "label")
        if instance_id in seen_ids:
            raise RgbSegmentationWorldError("segmentation frame contains duplicate instance IDs")
        seen_ids.add(instance_id)
        confidence = _confidence(getattr(instance, "confidence", None))
        x_offset = _nonnegative_int(getattr(instance, "x_offset", None), "x_offset")
        y_offset = _nonnegative_int(getattr(instance, "y_offset", None), "y_offset")
        width = _positive_int(getattr(instance, "width", None), "width")
        height = _positive_int(getattr(instance, "height", None), "height")
        if x_offset + width > image_width or y_offset + height > image_height:
            raise RgbSegmentationWorldError("segmentation instance ROI is outside the image")
        try:
            mask = bytes(getattr(instance, "mask", b""))
        except (TypeError, ValueError) as exc:
            raise RgbSegmentationWorldError("segmentation instance mask is invalid") from exc
        if len(mask) != width * height or any(value not in (0, 1) for value in mask):
            raise RgbSegmentationWorldError("segmentation instance mask is invalid")
        instance_source = _text(
            getattr(instance, "source", "") or source,
            "instance source",
            max_length=128,
        )
        mask_sha256 = hashlib.sha256(mask).hexdigest()
        parsed.append(
            {
                "instance_id": instance_id,
                "label": label,
                "confidence": confidence,
                "x_offset": x_offset,
                "y_offset": y_offset,
                "width": width,
                "height": height,
                "mask_sha256": mask_sha256,
                "instance_source": instance_source,
            }
        )
        object_id = "rgb:" + hashlib.sha256(
            f"{frame_id}\0{label}\0{instance_id}".encode("utf-8")
        ).hexdigest()[:48]
        objects.append(
            ObjectState(
                object_id=object_id,
                label=label,
                confidence=confidence,
                state="visible",
                last_seen=captured_at,
                properties={
                    "rgb_view": frame_id,
                    "image_width": image_width,
                    "image_height": image_height,
                    "roi": {
                        "x": x_offset,
                        "y": y_offset,
                        "width": width,
                        "height": height,
                    },
                    "mask_sha256": mask_sha256,
                    "instance_id": instance_id,
                },
                frame_id=frame_id,
                source=instance_source,
            )
        )

    payload = {
        "frame_id": frame_id,
        "image_width": image_width,
        "image_height": image_height,
        "captured_at": captured_at,
        "source": source,
        "instances": parsed,
    }
    digest = _digest(payload)
    resolved_observation_id = observation_id or f"rgb-{digest[:48]}"
    _text(resolved_observation_id, "observation_id", max_length=128)
    fusion_fingerprint = _digest({"schema_version": 1, **payload})
    objects = [replace(obj, observation_id=resolved_observation_id) for obj in objects]

    stale_ids = [
        obj.object_id
        for obj in objects
        if (current := world_model.get_object(obj.object_id)) is not None
        and current.last_seen > captured_at
    ]
    if stale_ids:
        raise RgbSegmentationWorldError(
            "segmentation frame is older than authoritative object state: "
            + ", ".join(stale_ids)
        )

    event = None
    if objects:
        try:
            event = world_model.apply_observation_fusion(
                objects,
                observation_id=resolved_observation_id,
                captured_at=captured_at,
                fusion_fingerprint=fusion_fingerprint,
                source=source,
            )
        except WorldGraphError as exc:
            raise RgbSegmentationWorldError(str(exc)) from exc

    return RgbSegmentationFusion(
        observation_id=resolved_observation_id,
        fusion_fingerprint=fusion_fingerprint,
        frame_id=frame_id,
        objects=tuple(objects),
        event=event,
    )


def _text(value: Any, label: str, *, max_length: int = 128) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > max_length:
        raise RgbSegmentationWorldError(f"{label} must be a bounded non-empty string")
    return value.strip()


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RgbSegmentationWorldError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RgbSegmentationWorldError(f"{label} must be a non-negative integer")
    return value


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RgbSegmentationWorldError("confidence must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise RgbSegmentationWorldError("confidence must be in [0, 1]")
    return normalized


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "RgbSegmentationFusion",
    "RgbSegmentationWorldError",
    "fuse_segmentation_frame",
]
