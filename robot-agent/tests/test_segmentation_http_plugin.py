from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from xlerobot_segmentation_http.backend import (
    HttpBackendError,
    LocalHttpSegmentationBackend,
)

from robot_agent.segmentation_backends import SegmentationInput

MODEL_SHA = "a" * 64


class FakeResponse:
    def __init__(self, payload, *, content_type="application/json", status=200):
        self.body = (
            payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        )
        self.headers = {"Content-Type": content_type}
        self.status = status

    def getcode(self):
        return self.status

    def read(self, amount):
        return self.body[:amount]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeOpener:
    def __init__(self, response):
        self.response = response
        self.request = None
        self.timeout = None

    def open(self, outbound, timeout):
        self.request = outbound
        self.timeout = timeout
        return self.response


def _artifacts(tmp_path: Path):
    manifest = tmp_path / "server-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "server_id": "grounded_sam_local_v1",
                "model_sha256": MODEL_SHA,
                "allowed_labels": ["cup", "remote"],
            }
        ),
        encoding="utf-8",
    )
    token = tmp_path / "token"
    token.write_text("0123456789abcdef0123456789abcdef\n", encoding="utf-8")
    return manifest, token


def _options(**changes):
    values = {
        "endpoint": "http://127.0.0.1:8090/v1/segment",
        "timeout_seconds": 0.5,
        "max_response_bytes": 4096,
        "manifest_artifact_index": 0,
        "token_artifact_index": 1,
    }
    values.update(changes)
    return values


def _input():
    return SegmentationInput(
        width=2,
        height=2,
        encoding="rgb8",
        step=6,
        data=bytes(range(12)),
        captured_at=10.0,
        frame_id="camera_color_optical_frame",
        queries={"cup": "drinking cup"},
    )


def _response(**changes):
    values = {
        "schema_version": 1,
        "server_id": "grounded_sam_local_v1",
        "model_sha256": MODEL_SHA,
        "instances": [
            {
                "instance_id": "cup-1",
                "label": "cup",
                "confidence": 0.9,
                "x_offset": 0,
                "y_offset": 0,
                "width": 2,
                "height": 2,
                "mask_base64": base64.b64encode(b"\x01\x01\x00\x00").decode(),
            }
        ],
    }
    values.update(changes)
    return values


def test_http_plugin_sends_bound_identity_queries_image_and_token(tmp_path: Path):
    artifacts = _artifacts(tmp_path)
    opener = FakeOpener(FakeResponse(_response()))
    backend = LocalHttpSegmentationBackend(
        options=_options(), artifacts=artifacts, opener=opener
    )
    predictions = backend.infer(_input())
    assert predictions[0].mask == b"\x01\x01\x00\x00"
    assert opener.timeout == 0.5
    assert opener.request.full_url == "http://127.0.0.1:8090/v1/segment"
    assert opener.request.headers["Authorization"] == (
        "Bearer 0123456789abcdef0123456789abcdef"
    )
    request_payload = json.loads(opener.request.data)
    assert request_payload["model_sha256"] == MODEL_SHA
    assert request_payload["queries"] == {"cup": "drinking cup"}
    assert base64.b64decode(request_payload["image_base64"]) == bytes(range(12))


def test_http_plugin_rejects_remote_endpoints_and_unbound_labels(tmp_path: Path):
    artifacts = _artifacts(tmp_path)
    with pytest.raises(HttpBackendError, match="loopback"):
        LocalHttpSegmentationBackend(
            options=_options(endpoint="https://example.com/v1/segment"),
            artifacts=artifacts,
        )
    backend = LocalHttpSegmentationBackend(
        options=_options(),
        artifacts=artifacts,
        opener=FakeOpener(FakeResponse(_response())),
    )
    invalid_input = SegmentationInput(**{**_input().__dict__, "queries": {"bottle": "bottle"}})
    with pytest.raises(HttpBackendError, match="does not allow labels"):
        backend.infer(invalid_input)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (_response(server_id="wrong"), "server identity"),
        (_response(model_sha256="b" * 64), "model digest"),
        (
            _response(
                instances=[
                    {
                        **_response()["instances"][0],
                        "label": "remote",
                    }
                ]
            ),
            "unrequested label",
        ),
        (
            _response(
                instances=[
                    {
                        **_response()["instances"][0],
                        "mask_base64": "not base64!",
                    }
                ]
            ),
            "base64",
        ),
    ],
)
def test_http_plugin_rejects_untrusted_response_identity_and_masks(
    tmp_path: Path,
    response,
    message: str,
):
    backend = LocalHttpSegmentationBackend(
        options=_options(),
        artifacts=_artifacts(tmp_path),
        opener=FakeOpener(FakeResponse(response)),
    )
    with pytest.raises(HttpBackendError, match=message):
        backend.infer(_input())


def test_http_plugin_rejects_content_type_and_response_size(tmp_path: Path):
    artifacts = _artifacts(tmp_path)
    backend = LocalHttpSegmentationBackend(
        options=_options(),
        artifacts=artifacts,
        opener=FakeOpener(FakeResponse(b"{}", content_type="text/plain")),
    )
    with pytest.raises(HttpBackendError, match="application/json"):
        backend.infer(_input())

    backend = LocalHttpSegmentationBackend(
        options=_options(max_response_bytes=1024),
        artifacts=artifacts,
        opener=FakeOpener(FakeResponse(b"x" * 1025)),
    )
    with pytest.raises(HttpBackendError, match="exceeds"):
        backend.infer(_input())


def test_reference_plugin_is_an_independent_entry_point_distribution():
    root = Path(__file__).resolve().parents[1]
    pyproject = (
        root / "plugins/xlerobot_segmentation_http/pyproject.toml"
    ).read_text(encoding="utf-8")
    assert 'name = "xlerobot-segmentation-http"' in pyproject
    assert 'local_http_v1 = "xlerobot_segmentation_http.backend:create_backend"' in pyproject
