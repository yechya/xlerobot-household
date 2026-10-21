from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from xlerobot_segmentation_server import SegmentationInstance, SegmentationRequest

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SAM2_CONFIG = re.compile(r"^configs/sam2(?:\.1)?/[A-Za-z0-9_.+-]+\.yaml$")


class GroundedSam2Error(ValueError):
    pass


@dataclass(frozen=True)
class ModelBundle:
    provider_id: str
    sam2_model_config: str
    sam2_checkpoint_sha256: str
    grounding_dino_config_sha256: str
    grounding_dino_checkpoint_sha256: str
    grounded_sam2_revision: str
    grounding_dino_revision: str
    sam2_revision: str

    @classmethod
    def load(cls, path: Path) -> ModelBundle:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GroundedSam2Error("model bundle manifest is invalid") from exc
        expected = {
            "schema_version",
            "provider_id",
            "sam2_model_config",
            "sam2_checkpoint_sha256",
            "grounding_dino_config_sha256",
            "grounding_dino_checkpoint_sha256",
            "grounded_sam2_revision",
            "grounding_dino_revision",
            "sam2_revision",
        }
        if not isinstance(raw, dict) or set(raw) != expected or raw.get("schema_version") != 1:
            raise GroundedSam2Error("model bundle manifest has missing or unexpected fields")
        if raw["provider_id"] != "grounded_sam2_v1":
            raise GroundedSam2Error("model bundle provider_id is invalid")
        if not isinstance(raw["sam2_model_config"], str) or not _SAM2_CONFIG.fullmatch(
            raw["sam2_model_config"]
        ):
            raise GroundedSam2Error("SAM2 model config name is invalid")
        for field in (
            "sam2_checkpoint_sha256",
            "grounding_dino_config_sha256",
            "grounding_dino_checkpoint_sha256",
        ):
            if not isinstance(raw[field], str) or not _SHA256.fullmatch(raw[field]):
                raise GroundedSam2Error(f"{field} is invalid")
        for field in (
            "grounded_sam2_revision",
            "grounding_dino_revision",
            "sam2_revision",
        ):
            if not isinstance(raw[field], str) or not _REVISION.fullmatch(raw[field]):
                raise GroundedSam2Error(f"{field} must be a full Git commit")
        return cls(**{key: raw[key] for key in expected if key != "schema_version"})


class _RuntimeDependencies:
    @classmethod
    def load(cls) -> _RuntimeDependencies:
        try:
            from groundingdino.datasets import transforms as transforms
            from groundingdino.util.inference import load_model, predict
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise GroundedSam2Error(
                "reviewed GroundingDINO and SAM2 runtime packages are required"
            ) from exc
        instance = cls()
        instance.image_transform = transforms.Compose(
            [
                transforms.RandomResize([800], max_size=1333),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )
        instance.load_grounding_model = load_model
        instance.predict_boxes = predict
        instance.build_sam2 = build_sam2
        instance.sam2_predictor_type = SAM2ImagePredictor
        return instance


class GroundedSam2Provider:
    def __init__(
        self,
        *,
        options: dict[str, Any],
        artifacts: tuple[Path, ...],
        model_sha256: str,
        allowed_labels: frozenset[str],
        dependencies: Any | None = None,
    ):
        expected = {
            "bundle_manifest_artifact_index",
            "sam2_checkpoint_artifact_index",
            "grounding_dino_config_artifact_index",
            "grounding_dino_checkpoint_artifact_index",
            "device",
            "box_threshold",
            "text_threshold",
            "nms_iou_threshold",
            "max_detections_per_label",
        }
        if not isinstance(options, dict) or set(options) != expected:
            raise GroundedSam2Error("provider options have missing or unexpected fields")
        bundle_path = artifacts[
            _artifact_index(
                options["bundle_manifest_artifact_index"], artifacts, "bundle manifest"
            )
        ]
        if _sha256_file(bundle_path) != model_sha256:
            raise GroundedSam2Error("model identity does not match bundle manifest")
        self.bundle = ModelBundle.load(bundle_path)
        self.sam2_checkpoint = artifacts[
            _artifact_index(options["sam2_checkpoint_artifact_index"], artifacts, "SAM2 checkpoint")
        ]
        self.grounding_config = artifacts[
            _artifact_index(
                options["grounding_dino_config_artifact_index"],
                artifacts,
                "GroundingDINO config",
            )
        ]
        self.grounding_checkpoint = artifacts[
            _artifact_index(
                options["grounding_dino_checkpoint_artifact_index"],
                artifacts,
                "GroundingDINO checkpoint",
            )
        ]
        _require_digest(
            self.sam2_checkpoint, self.bundle.sam2_checkpoint_sha256, "SAM2 checkpoint"
        )
        _require_digest(
            self.grounding_config,
            self.bundle.grounding_dino_config_sha256,
            "GroundingDINO config",
        )
        _require_digest(
            self.grounding_checkpoint,
            self.bundle.grounding_dino_checkpoint_sha256,
            "GroundingDINO checkpoint",
        )
        self.device = _device(options["device"])
        self.box_threshold = _bounded_float(options["box_threshold"], "box_threshold", 0, 1)
        self.text_threshold = _bounded_float(
            options["text_threshold"], "text_threshold", 0, 1
        )
        self.nms_iou_threshold = _bounded_float(
            options["nms_iou_threshold"], "nms_iou_threshold", 0, 1
        )
        self.max_detections_per_label = _bounded_integer(
            options["max_detections_per_label"], "max_detections_per_label", 1, 128
        )
        if not allowed_labels:
            raise GroundedSam2Error("allowed_labels cannot be empty")
        self.allowed_labels = allowed_labels
        self.dependencies = dependencies or _RuntimeDependencies.load()
        self.grounding_model = self.dependencies.load_grounding_model(
            model_config_path=str(self.grounding_config),
            model_checkpoint_path=str(self.grounding_checkpoint),
            device=self.device,
        )
        sam2_model = self.dependencies.build_sam2(
            self.bundle.sam2_model_config,
            str(self.sam2_checkpoint),
            device=self.device,
        )
        self.sam2_predictor = self.dependencies.sam2_predictor_type(sam2_model)

    def infer(self, request: SegmentationRequest) -> list[SegmentationInstance]:
        unknown = sorted(set(request.queries) - self.allowed_labels)
        if unknown:
            raise GroundedSam2Error("request contains labels outside allowed_labels")
        image = _decode_image(request)
        pil_image = Image.fromarray(image, mode="RGB")
        transformed, _ = self.dependencies.image_transform(pil_image, None)
        boxes: list[np.ndarray] = []
        confidences: list[float] = []
        labels: list[str] = []
        for label, prompt in request.queries.items():
            raw_boxes, raw_confidences, _ = self.dependencies.predict_boxes(
                model=self.grounding_model,
                image=transformed,
                caption=prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )
            normalized_boxes = _as_numpy(raw_boxes, "boxes")
            normalized_confidences = _as_numpy(raw_confidences, "confidences").reshape(-1)
            if normalized_boxes.size == 0:
                continue
            if normalized_boxes.ndim != 2 or normalized_boxes.shape[1] != 4:
                raise GroundedSam2Error("GroundingDINO boxes have an invalid shape")
            if len(normalized_boxes) != len(normalized_confidences):
                raise GroundedSam2Error("GroundingDINO boxes and confidences do not align")
            pixel_boxes = _cxcywh_to_xyxy(normalized_boxes, request.width, request.height)
            keep = _nms(pixel_boxes, normalized_confidences, self.nms_iou_threshold)
            for index in keep[: self.max_detections_per_label]:
                boxes.append(pixel_boxes[index])
                confidences.append(float(normalized_confidences[index]))
                labels.append(label)
        if not boxes:
            return []
        input_boxes = np.asarray(boxes, dtype=np.float32)
        self.sam2_predictor.set_image(image)
        raw_masks, raw_scores, _ = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        masks = _as_numpy(raw_masks, "SAM2 masks")
        scores = _as_numpy(raw_scores, "SAM2 scores").reshape(-1)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim != 3 or masks.shape[1:] != (request.height, request.width):
            raise GroundedSam2Error("SAM2 masks have an invalid shape")
        if len(masks) != len(boxes) or len(scores) != len(boxes):
            raise GroundedSam2Error("SAM2 output count does not match input boxes")
        instances: list[SegmentationInstance] = []
        counters: dict[str, int] = {}
        for label, detector_confidence, mask, sam_score in zip(
            labels, confidences, masks, scores, strict=True
        ):
            cropped = _crop_binary_mask(mask)
            if cropped is None:
                continue
            x_offset, y_offset, width, height, mask_bytes = cropped
            counters[label] = counters.get(label, 0) + 1
            instances.append(
                SegmentationInstance(
                    instance_id=f"{label}-{counters[label]:04d}",
                    label=label,
                    confidence=max(0.0, min(1.0, detector_confidence * float(sam_score))),
                    x_offset=x_offset,
                    y_offset=y_offset,
                    width=width,
                    height=height,
                    mask=mask_bytes,
                )
            )
        return instances


def create_provider(
    *,
    options: dict[str, Any],
    artifacts: tuple[Path, ...],
    server_id: str,
    model_sha256: str,
    allowed_labels: frozenset[str],
) -> GroundedSam2Provider:
    if not server_id:
        raise GroundedSam2Error("server_id cannot be empty")
    return GroundedSam2Provider(
        options=options,
        artifacts=artifacts,
        model_sha256=model_sha256,
        allowed_labels=allowed_labels,
    )


def _decode_image(request: SegmentationRequest) -> np.ndarray:
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(
        request.encoding
    )
    if channels is None or request.step < request.width * channels:
        raise GroundedSam2Error("image encoding or step is unsupported")
    rows = np.frombuffer(request.image, dtype=np.uint8).reshape(request.height, request.step)
    pixels = rows[:, : request.width * channels].reshape(request.height, request.width, channels)
    if request.encoding == "bgr8":
        return np.ascontiguousarray(pixels[:, :, ::-1])
    if request.encoding == "rgba8":
        return np.ascontiguousarray(pixels[:, :, :3])
    if request.encoding == "bgra8":
        return np.ascontiguousarray(pixels[:, :, 2::-1])
    if request.encoding == "mono8":
        return np.repeat(pixels, 3, axis=2)
    return np.ascontiguousarray(pixels)


def _cxcywh_to_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    if not np.isfinite(boxes).all():
        raise GroundedSam2Error("GroundingDINO boxes must be finite")
    result = np.empty_like(boxes, dtype=np.float32)
    result[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * width
    result[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * height
    result[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * width
    result[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * height
    result[:, (0, 2)] = np.clip(result[:, (0, 2)], 0, width)
    result[:, (1, 3)] = np.clip(result[:, (1, 3)], 0, height)
    valid = (result[:, 2] > result[:, 0]) & (result[:, 3] > result[:, 1])
    if not valid.all():
        raise GroundedSam2Error("GroundingDINO returned an empty or inverted box")
    return result


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    if (
        len(boxes) != len(scores)
        or not np.isfinite(scores).all()
        or not ((0 <= scores) & (scores <= 1)).all()
    ):
        raise GroundedSam2Error("GroundingDINO confidence output is invalid")
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while len(order):
        current = int(order[0])
        keep.append(current)
        if len(order) == 1:
            break
        remaining = order[1:]
        left = np.maximum(boxes[current, 0], boxes[remaining, 0])
        top = np.maximum(boxes[current, 1], boxes[remaining, 1])
        right = np.minimum(boxes[current, 2], boxes[remaining, 2])
        bottom = np.minimum(boxes[current, 3], boxes[remaining, 3])
        intersection = np.maximum(0, right - left) * np.maximum(0, bottom - top)
        current_area = (boxes[current, 2] - boxes[current, 0]) * (
            boxes[current, 3] - boxes[current, 1]
        )
        remaining_area = (boxes[remaining, 2] - boxes[remaining, 0]) * (
            boxes[remaining, 3] - boxes[remaining, 1]
        )
        union = current_area + remaining_area - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = remaining[iou <= threshold]
    return keep


def _crop_binary_mask(mask: np.ndarray) -> tuple[int, int, int, int, bytes] | None:
    if not np.isfinite(mask).all():
        raise GroundedSam2Error("SAM2 mask contains non-finite values")
    binary = np.asarray(mask) > 0
    locations = np.argwhere(binary)
    if not len(locations):
        return None
    y_min, x_min = locations.min(axis=0)
    y_max, x_max = locations.max(axis=0) + 1
    cropped = np.ascontiguousarray(binary[y_min:y_max, x_min:x_max], dtype=np.uint8)
    return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min), cropped.tobytes()


def _as_numpy(value: Any, label: str) -> np.ndarray:
    try:
        if callable(getattr(value, "detach", None)):
            value = value.detach()
        if callable(getattr(value, "cpu", None)):
            value = value.cpu()
        if callable(getattr(value, "numpy", None)):
            value = value.numpy()
        return np.asarray(value)
    except Exception as exc:
        raise GroundedSam2Error(f"{label} cannot be converted to an array") from exc


def _artifact_index(value: Any, artifacts: tuple[Path, ...], label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(artifacts):
        raise GroundedSam2Error(f"{label} artifact index is invalid")
    return value


def _require_digest(path: Path, expected: str, label: str) -> None:
    if _sha256_file(path) != expected:
        raise GroundedSam2Error(f"{label} digest does not match the model bundle")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise GroundedSam2Error(f"model artifact cannot be read: {path}") from exc
    return digest.hexdigest()


def _device(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"(?:cpu|cuda(?::[0-9]{1,2})?)", value):
        raise GroundedSam2Error("device must be cpu, cuda or cuda:N")
    return value


def _bounded_float(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GroundedSam2Error(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise GroundedSam2Error(f"{label} must be in [{minimum}, {maximum}]")
    return result


def _bounded_integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise GroundedSam2Error(f"{label} must be in [{minimum}, {maximum}]")
    return value


__all__ = ["GroundedSam2Error", "GroundedSam2Provider", "ModelBundle", "create_provider"]
