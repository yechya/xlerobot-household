from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from robot_agent.segmentation_adapter_node import populate_segmentation_frame
from robot_agent.segmentation_backends import (
    ENTRY_POINT_GROUP,
    MaskPrediction,
    SegmentationBackendError,
    SegmentationBackendSpec,
    image_request_from_ros,
    load_backend,
    run_inference,
    validate_predictions,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spec(tmp_path: Path, **changes: object) -> SegmentationBackendSpec:
    artifact = tmp_path / "model.bin"
    if not artifact.exists():
        artifact.write_bytes(b"reviewed-model")
    raw = {
        "version": 1,
        "approved": True,
        "backend_id": "open_vocab_v1",
        "entry_point": "open_vocab_v1",
        "distribution": "xlerobot-segmentation-test",
        "distribution_version": "1.2.3",
        "source_id": "reviewed-segmentation/v1",
        "image_topic": "/camera/color/image_raw",
        "output_topic": "/xlerobot/segmentation",
        "camera_frame": "camera_color_optical_frame",
        "queries": {"cup": "cup", "remote": "remote control"},
        "min_confidence": 0.7,
        "max_instances": 4,
        "max_image_bytes": 1024,
        "max_input_age_seconds": 0.5,
        "max_inference_seconds": 1.0,
        "artifacts": [{"path": artifact.name, "sha256": _digest(artifact)}],
        "options": {"device": "cpu"},
    }
    raw.update(changes)
    path = tmp_path / "segmentation.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return SegmentationBackendSpec.load(path)


def _image() -> SimpleNamespace:
    return SimpleNamespace(
        header=SimpleNamespace(
            frame_id="camera_color_optical_frame",
            stamp=SimpleNamespace(sec=10, nanosec=0),
        ),
        width=4,
        height=3,
        step=12,
        encoding="rgb8",
        data=bytes(range(36)),
    )


def _prediction(**changes: object) -> MaskPrediction:
    values = {
        "instance_id": "cup-1",
        "label": "cup",
        "confidence": 0.9,
        "x_offset": 1,
        "y_offset": 1,
        "width": 2,
        "height": 1,
        "mask": b"\x01\x01",
    }
    values.update(changes)
    return MaskPrediction(**values)


class FakeDistribution:
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.metadata = {"Name": name}


class FakeEntryPoint:
    name = "open_vocab_v1"
    group = ENTRY_POINT_GROUP

    def __init__(self, factory, *, distribution="xlerobot-segmentation-test", version="1.2.3"):
        self.factory = factory
        self.dist = FakeDistribution(distribution, version)

    def load(self):
        return self.factory


class FakeEntryPoints(list):
    def select(self, *, group: str, name: str):
        return [entry for entry in self if entry.group == group and entry.name == name]


def test_spec_binds_queries_options_and_model_artifacts(tmp_path: Path):
    spec = _spec(tmp_path)
    spec.validate_artifacts()
    assert spec.queries == {"cup": "cup", "remote": "remote control"}
    assert spec.options == {"device": "cpu"}
    assert spec.artifacts[0].path == (tmp_path / "model.bin").resolve()
    assert len(spec.fingerprint) == 64

    changed = _spec(tmp_path, queries={"cup": "drinking cup"})
    assert changed.fingerprint != spec.fingerprint


def test_artifact_tampering_and_default_deny_are_rejected(tmp_path: Path):
    spec = _spec(tmp_path)
    (tmp_path / "model.bin").write_bytes(b"tampered")
    with pytest.raises(SegmentationBackendError, match="SHA-256 mismatch"):
        spec.validate_artifacts()

    with pytest.raises(SegmentationBackendError, match="must be approved"):
        _spec(tmp_path, approved=False)


def test_entry_point_loader_binds_distribution_version_and_factory_inputs(tmp_path: Path):
    spec = _spec(tmp_path)
    received = {}
    backend = SimpleNamespace(infer=lambda request: [])

    def factory(*, options, artifacts):
        received.update(options=options, artifacts=artifacts)
        return backend

    loaded = load_backend(spec, entry_points=FakeEntryPoints([FakeEntryPoint(factory)]))
    assert loaded is backend
    assert received == {
        "options": {"device": "cpu"},
        "artifacts": ((tmp_path / "model.bin").resolve(),),
    }

    with pytest.raises(SegmentationBackendError, match="distribution"):
        load_backend(
            spec,
            entry_points=FakeEntryPoints([FakeEntryPoint(factory, distribution="wrong")]),
        )
    with pytest.raises(SegmentationBackendError, match="version"):
        load_backend(
            spec,
            entry_points=FakeEntryPoints([FakeEntryPoint(factory, version="9.9.9")]),
        )


def test_image_and_prediction_contract_filters_only_reviewed_outputs(tmp_path: Path):
    spec = _spec(tmp_path)
    request = image_request_from_ros(_image(), spec, captured_at=10.0)
    accepted = validate_predictions(
        [_prediction(), _prediction(instance_id="cup-2", confidence=0.5)],
        request,
        spec,
    )
    assert [prediction.instance_id for prediction in accepted] == ["cup-1"]
    assert request.queries == spec.queries

    with pytest.raises(SegmentationBackendError, match="undeclared label"):
        validate_predictions([_prediction(label="bottle")], request, spec)
    with pytest.raises(SegmentationBackendError, match="outside the image"):
        validate_predictions([_prediction(x_offset=3, width=2)], request, spec)
    with pytest.raises(SegmentationBackendError, match="binary ROI"):
        validate_predictions([_prediction(mask=b"\x01\x02")], request, spec)
    with pytest.raises(SegmentationBackendError, match="duplicate"):
        validate_predictions([_prediction(), _prediction()], request, spec)


class FakeInstance:
    def __init__(self):
        self.instance_id = ""
        self.label = ""
        self.confidence = 0.0
        self.x_offset = 0
        self.y_offset = 0
        self.width = 0
        self.height = 0
        self.mask = []
        self.source = ""


def test_ros_output_preserves_image_stamp_and_overwrites_provenance():
    image = _image()
    message = SimpleNamespace(header=None, image_width=0, image_height=0, instances=[])
    output = populate_segmentation_frame(
        message,
        image,
        (_prediction(),),
        FakeInstance,
        source="reviewed-segmentation/v1/open_vocab_v1",
    )
    assert output.header is image.header
    assert (output.image_width, output.image_height) == (4, 3)
    assert output.instances[0].mask == [1, 1]
    assert output.instances[0].source == "reviewed-segmentation/v1/open_vocab_v1"


def test_inference_boundary_contains_plugin_errors_and_rejects_late_output(tmp_path: Path):
    spec = _spec(tmp_path)
    request = image_request_from_ros(_image(), spec, captured_at=10.0)
    failing = SimpleNamespace(
        infer=lambda request: (_ for _ in ()).throw(RuntimeError("model crashed"))
    )
    with pytest.raises(SegmentationBackendError, match="inference failed"):
        run_inference(failing, request, spec)

    clock = iter((10.0, 11.1)).__next__
    backend = SimpleNamespace(infer=lambda request: [_prediction()])
    with pytest.raises(SegmentationBackendError, match="deadline"):
        run_inference(backend, request, spec, clock=clock)
