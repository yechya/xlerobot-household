from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_agent.rgb_world_model_bridge import (
    RgbWorldModelBridge,
    RgbWorldModelBridgeError,
    ros_time_to_seconds,
)


def _frame(*, seconds: int = 100, instances: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        header=SimpleNamespace(
            frame_id="head_cam",
            stamp=SimpleNamespace(sec=seconds, nanosec=0),
        ),
        image_width=640,
        image_height=480,
        instances=list(instances or []),
    )


def test_bridge_requires_absolute_world_model_path(tmp_path: Path):
    with pytest.raises(RgbWorldModelBridgeError, match="absolute"):
        RgbWorldModelBridge("relative/world.yaml")

    directory = tmp_path / "world"
    directory.mkdir()
    with pytest.raises(RgbWorldModelBridgeError, match="file"):
        RgbWorldModelBridge(directory)


def test_bridge_accepts_fresh_empty_frame_without_creating_object(tmp_path: Path):
    bridge = RgbWorldModelBridge(tmp_path / "world.yaml", world_model=None)

    result = bridge.handle_frame(_frame(), now=100.2)

    assert result.success is True
    assert result.object_count == 0
    assert result.event_sequence is None
    assert bridge.world_model.get_all_objects() == []


def test_bridge_rejects_future_and_stale_frames(tmp_path: Path):
    bridge = RgbWorldModelBridge(tmp_path / "world.yaml", max_input_age_seconds=0.5)

    future = bridge.handle_frame(_frame(seconds=101), now=100.0)
    stale = bridge.handle_frame(_frame(seconds=99), now=100.0)

    assert future.success is False
    assert "future" in future.message
    assert stale.success is False
    assert "stale" in stale.message


def test_bridge_persists_world_events_and_rejects_duplicate_id(tmp_path: Path):
    from tests.test_rgb_segmentation_world import _frame as segmentation_frame
    from tests.test_rgb_segmentation_world import _instance

    path = tmp_path / "world.yaml"
    bridge = RgbWorldModelBridge(path)
    frame = segmentation_frame(_instance())
    frame.header.stamp = SimpleNamespace(sec=100, nanosec=0)

    first = bridge.handle_frame(frame, now=100.1, observation_id="rgb-frame-1")
    duplicate = bridge.handle_frame(frame, now=100.2, observation_id="rgb-frame-1")

    assert first.success is True
    assert first.event_sequence is not None
    assert duplicate.success is False
    assert "already been committed" in duplicate.message
    assert path.is_file()
    assert path.with_suffix(".yaml.events.jsonl").is_file()


def test_ros_time_conversion_rejects_invalid_stamp():
    assert ros_time_to_seconds(SimpleNamespace(sec=1, nanosec=500_000_000)) == 1.5
    with pytest.raises(RgbWorldModelBridgeError, match="timestamp"):
        ros_time_to_seconds(SimpleNamespace(sec="bad", nanosec=0))
