from __future__ import annotations

from types import SimpleNamespace

import pytest

from robot_agent.rgb_segmentation_world import (
    RgbSegmentationWorldError,
    fuse_segmentation_frame,
)
from robot_agent.world_model import WorldModel


def _instance(**changes: object) -> SimpleNamespace:
    values = {
        "instance_id": "bottle-1",
        "label": "bottle",
        "confidence": 0.92,
        "x_offset": 10,
        "y_offset": 20,
        "width": 4,
        "height": 3,
        "mask": [1] * 12,
        "source": "reviewed-segmentation/v1",
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _frame(*instances: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        header=SimpleNamespace(frame_id="head_cam"),
        image_width=640,
        image_height=480,
        instances=list(instances),
    )


def test_segmentation_frame_fuses_qualitative_rgb_object_state():
    model = WorldModel()

    fusion = fuse_segmentation_frame(model, _frame(_instance()), captured_at=100.0)

    assert fusion.event.event_type == "observation_fusion"
    assert len(fusion.objects) == 1
    object_state = fusion.objects[0]
    assert object_state.label == "bottle"
    assert object_state.state == "visible"
    assert object_state.frame_id == "head_cam"
    assert object_state.x == object_state.y == object_state.z == 0.0
    assert object_state.properties["roi"] == {
        "x": 10,
        "y": 20,
        "width": 4,
        "height": 3,
    }
    assert model.check_precondition("object_visible(bottle)") is True


def test_empty_segmentation_frame_does_not_delete_previous_objects():
    model = WorldModel()
    first = fuse_segmentation_frame(model, _frame(_instance()), captured_at=100.0)

    empty = fuse_segmentation_frame(model, _frame(), captured_at=101.0)

    assert empty.event is None
    assert model.get_object(first.objects[0].object_id) is not None


def test_stale_segmentation_frame_is_rejected():
    model = WorldModel()
    first = fuse_segmentation_frame(model, _frame(_instance()), captured_at=100.0)

    with pytest.raises(RgbSegmentationWorldError, match="older"):
        fuse_segmentation_frame(model, _frame(_instance()), captured_at=99.0)

    assert model.get_object(first.objects[0].object_id).last_seen == 100.0


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"instance_id": "bottle-1"}, "duplicate"),
        ({"x_offset": 639, "width": 2}, "outside"),
        ({"mask": [2] * 12}, "mask"),
        ({"confidence": 1.2}, "confidence"),
    ],
)
def test_invalid_segmentation_instances_fail_closed(changes, message):
    model = WorldModel()
    instances = [_instance(), _instance(instance_id="cup-1")]
    if message == "duplicate":
        instances[1] = _instance()
    else:
        instances[0] = _instance(**changes)

    with pytest.raises(RgbSegmentationWorldError, match=message):
        fuse_segmentation_frame(model, _frame(*instances), captured_at=100.0)


def test_duplicate_observation_id_is_rejected():
    model = WorldModel()
    frame = _frame(_instance())
    fuse_segmentation_frame(model, frame, captured_at=100.0, observation_id="rgb-frame-1")

    with pytest.raises(RgbSegmentationWorldError, match="already been committed"):
        fuse_segmentation_frame(model, frame, captured_at=101.0, observation_id="rgb-frame-1")
