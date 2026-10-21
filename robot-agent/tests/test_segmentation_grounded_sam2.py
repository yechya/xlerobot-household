from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from xlerobot_segmentation_grounded_sam2.provider import (
    GroundedSam2Error,
    GroundedSam2Provider,
)
from xlerobot_segmentation_server.protocol import SegmentationRequest, encode_response


class FakeTransform:
    def __call__(self, image, target):
        return np.asarray(image), target


class FakeSamPredictor:
    def __init__(self, model):
        self.model = model
        self.image = None

    def set_image(self, image):
        self.image = image

    def predict(self, *, point_coords, point_labels, box, multimask_output):
        assert point_coords is None
        assert point_labels is None
        assert multimask_output is False
        masks = np.zeros((len(box), 1, self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
        for index, (left, top, right, bottom) in enumerate(box.astype(int)):
            masks[index, 0, top:bottom, left:right] = 1
        return masks, np.full((len(box), 1), 0.8), np.zeros((len(box), 1))


class FakeDependencies:
    def __init__(self):
        self.image_transform = FakeTransform()
        self.sam2_predictor_type = FakeSamPredictor
        self.grounding_args = None
        self.sam2_args = None
        self.captions = []

    def load_grounding_model(self, **kwargs):
        self.grounding_args = kwargs
        return "grounding-model"

    def build_sam2(self, *args, **kwargs):
        self.sam2_args = (args, kwargs)
        return "sam2-model"

    def predict_boxes(self, **kwargs):
        self.captions.append(kwargs["caption"])
        if kwargs["caption"] == "drinking cup":
            return (
                np.asarray([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.48, 0.48]]),
                np.asarray([0.9, 0.7]),
                ["cup", "cup"],
            )
        return np.asarray([[0.25, 0.25, 0.25, 0.25]]), np.asarray([0.75]), ["remote"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifacts(tmp_path: Path):
    sam2 = tmp_path / "sam2.pt"
    sam2.write_bytes(b"sam2 checkpoint")
    grounding_config = tmp_path / "grounding.py"
    grounding_config.write_text("model = 'grounding'\n", encoding="utf-8")
    grounding = tmp_path / "grounding.pth"
    grounding.write_bytes(b"grounding checkpoint")
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provider_id": "grounded_sam2_v1",
                "sam2_model_config": "configs/sam2.1/sam2.1_hiera_s.yaml",
                "sam2_checkpoint_sha256": _sha(sam2),
                "grounding_dino_config_sha256": _sha(grounding_config),
                "grounding_dino_checkpoint_sha256": _sha(grounding),
                "grounded_sam2_revision": "a" * 40,
                "grounding_dino_revision": "b" * 40,
                "sam2_revision": "c" * 40,
            }
        ),
        encoding="utf-8",
    )
    return (bundle, sam2, grounding_config, grounding)


def _options(**changes):
    values = {
        "bundle_manifest_artifact_index": 0,
        "sam2_checkpoint_artifact_index": 1,
        "grounding_dino_config_artifact_index": 2,
        "grounding_dino_checkpoint_artifact_index": 3,
        "device": "cuda:0",
        "box_threshold": 0.35,
        "text_threshold": 0.25,
        "nms_iou_threshold": 0.5,
        "max_detections_per_label": 4,
    }
    values.update(changes)
    return values


def _request(*, encoding="rgb8", step=12, data=None):
    if data is None:
        data = bytes(range(step * 4))
    return SegmentationRequest(
        server_id="grounded_sam2_local_v1",
        model_sha256="unused",
        captured_at=1.0,
        frame_id="camera_frame",
        width=4,
        height=4,
        encoding=encoding,
        step=step,
        queries={"cup": "drinking cup", "remote": "remote control"},
        image=data,
    )


def _provider(tmp_path: Path, dependencies=None, **option_changes):
    artifacts = _artifacts(tmp_path)
    return GroundedSam2Provider(
        options=_options(**option_changes),
        artifacts=artifacts,
        model_sha256=_sha(artifacts[0]),
        allowed_labels=frozenset({"cup", "remote"}),
        dependencies=dependencies or FakeDependencies(),
    )


def test_provider_binds_bundle_and_loads_reviewed_model_assets(tmp_path: Path):
    dependencies = FakeDependencies()
    provider = _provider(tmp_path, dependencies)
    assert dependencies.grounding_args == {
        "model_config_path": str(provider.grounding_config),
        "model_checkpoint_path": str(provider.grounding_checkpoint),
        "device": "cuda:0",
    }
    assert dependencies.sam2_args[0][0] == "configs/sam2.1/sam2.1_hiera_s.yaml"
    assert dependencies.sam2_args[0][1] == str(provider.sam2_checkpoint)


def test_provider_maps_each_query_and_returns_binary_roi_masks(tmp_path: Path):
    dependencies = FakeDependencies()
    provider = _provider(tmp_path, dependencies)
    predictions = provider.infer(_request())
    assert dependencies.captions == ["drinking cup", "remote control"]
    assert [prediction.label for prediction in predictions] == ["cup", "remote"]
    assert predictions[0].instance_id == "cup-0001"
    assert predictions[0].confidence == pytest.approx(0.72)
    assert set(predictions[0].mask) == {1}
    response = encode_response(predictions, _request(), max_instances=8)
    assert len(response["instances"]) == 2


def test_provider_decodes_padded_bgr_without_changing_request_contract(tmp_path: Path):
    provider = _provider(tmp_path)
    rows = []
    for _ in range(4):
        rows.extend([3, 2, 1] * 4 + [99, 99, 99, 99])
    request = _request(encoding="bgr8", step=16, data=bytes(rows))
    provider.infer(request)
    assert provider.sam2_predictor.image.shape == (4, 4, 3)
    assert provider.sam2_predictor.image[0, 0].tolist() == [1, 2, 3]


def test_provider_rejects_bundle_drift_before_loading_models(tmp_path: Path):
    artifacts = _artifacts(tmp_path)
    artifacts[1].write_bytes(b"tampered")
    with pytest.raises(GroundedSam2Error, match="SAM2 checkpoint digest"):
        GroundedSam2Provider(
            options=_options(),
            artifacts=artifacts,
            model_sha256=_sha(artifacts[0]),
            allowed_labels=frozenset({"cup"}),
            dependencies=FakeDependencies(),
        )


def test_provider_rejects_unknown_labels_and_unsupported_encoding(tmp_path: Path):
    provider = _provider(tmp_path)
    request = _request()
    request = SegmentationRequest(**{**request.__dict__, "queries": {"bottle": "bottle"}})
    with pytest.raises(GroundedSam2Error, match="allowed_labels"):
        provider.infer(request)
    with pytest.raises(GroundedSam2Error, match="encoding"):
        provider.infer(_request(encoding="16UC1"))


def test_provider_distribution_registers_the_stable_entry_point():
    root = Path(__file__).resolve().parents[1]
    pyproject = (
        root / "plugins/xlerobot_segmentation_grounded_sam2/pyproject.toml"
    ).read_text(encoding="utf-8")
    assert 'name = "xlerobot-segmentation-grounded-sam2"' in pyproject
    assert "grounded_sam2_v1" in pyproject
