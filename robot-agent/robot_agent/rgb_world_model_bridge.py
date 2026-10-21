from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .rgb_segmentation_world import (
    RgbSegmentationWorldError,
    fuse_segmentation_frame,
)
from .world_model import WorldModel


class RgbWorldModelBridgeError(ValueError):
    pass


@dataclass(frozen=True)
class RgbWorldModelBridgeResult:
    success: bool
    message: str
    observation_id: str = ""
    frame_id: str = ""
    captured_at: float = 0.0
    object_count: int = 0
    event_sequence: int | None = None


class RgbWorldModelBridge:
    """Bridge reviewed ROS RGB segmentation frames into a persistent world model."""

    def __init__(
        self,
        world_model_path: str | Path,
        *,
        max_input_age_seconds: float = 0.5,
        source: str = "rgb_segmentation_bridge",
        world_model: WorldModel | None = None,
    ):
        path = Path(world_model_path).expanduser()
        if not path.is_absolute() or path.name in {"", ".", ".."}:
            raise RgbWorldModelBridgeError("world_model_path must be an absolute file path")
        if path.exists() and not path.is_file():
            raise RgbWorldModelBridgeError("world_model_path must point to a file")
        if (
            isinstance(max_input_age_seconds, bool)
            or not isinstance(max_input_age_seconds, (int, float))
            or not math.isfinite(float(max_input_age_seconds))
            or not 0.001 <= float(max_input_age_seconds) <= 60.0
        ):
            raise RgbWorldModelBridgeError(
                "max_input_age_seconds must be between 0.001 and 60 seconds"
            )
        if not isinstance(source, str) or not source.strip() or len(source.strip()) > 256:
            raise RgbWorldModelBridgeError("source must be a bounded non-empty string")
        self.path = path
        self.max_input_age_seconds = float(max_input_age_seconds)
        self.source = source.strip()
        self.world_model = world_model or WorldModel(path)

    def handle_frame(
        self,
        frame: Any,
        *,
        now: float | None = None,
        observation_id: str | None = None,
    ) -> RgbWorldModelBridgeResult:
        captured_at = 0.0
        try:
            captured_at = ros_time_to_seconds(frame.header.stamp)
            current_time = time.time() if now is None else float(now)
            self._validate_timing(captured_at, current_time)
            fusion = fuse_segmentation_frame(
                self.world_model,
                frame,
                captured_at=captured_at,
                source=self.source,
                observation_id=observation_id,
            )
        except (AttributeError, TypeError, ValueError, RgbSegmentationWorldError) as exc:
            return RgbWorldModelBridgeResult(
                success=False,
                message=f"RGB world-model frame rejected: {exc}",
                captured_at=captured_at,
            )
        return RgbWorldModelBridgeResult(
            success=True,
            message=(
                f"fused {len(fusion.objects)} RGB objects"
                if fusion.objects
                else "accepted empty RGB segmentation frame"
            ),
            observation_id=fusion.observation_id,
            frame_id=fusion.frame_id,
            captured_at=captured_at,
            object_count=len(fusion.objects),
            event_sequence=fusion.event.sequence if fusion.event is not None else None,
        )

    def _validate_timing(self, captured_at: float, now: float) -> None:
        if not math.isfinite(captured_at) or captured_at <= 0:
            raise RgbWorldModelBridgeError("segmentation timestamp is invalid")
        if not math.isfinite(now) or now <= 0:
            raise RgbWorldModelBridgeError("bridge clock is invalid")
        age = now - captured_at
        if age < 0:
            raise RgbWorldModelBridgeError("segmentation frame is from the future")
        if age > self.max_input_age_seconds:
            raise RgbWorldModelBridgeError("segmentation frame is stale")


def ros_time_to_seconds(stamp: Any) -> float:
    try:
        seconds = float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000
    except (AttributeError, TypeError, ValueError) as exc:
        raise RgbWorldModelBridgeError("segmentation timestamp is invalid") from exc
    return seconds


__all__ = [
    "RgbWorldModelBridge",
    "RgbWorldModelBridgeError",
    "RgbWorldModelBridgeResult",
    "ros_time_to_seconds",
]
