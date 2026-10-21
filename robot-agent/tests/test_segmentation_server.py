from __future__ import annotations

import base64
import hashlib
import json
import threading
from dataclasses import replace
from pathlib import Path
from urllib import error, request

import pytest
from xlerobot_segmentation_server.config import ServerConfig, ServerConfigError
from xlerobot_segmentation_server.conformance import ConformanceError, evaluate_endpoint
from xlerobot_segmentation_server.protocol import (
    ProtocolError,
    SegmentationInstance,
    SegmentationRequest,
    encode_response,
    parse_request,
)
from xlerobot_segmentation_server.runtime import ProviderError, create_server, load_provider


class TestProvider:
    def infer(self, segmentation_request):
        label = next(iter(segmentation_request.queries))
        return [
            SegmentationInstance(
                instance_id="test-1",
                label=label,
                confidence=0.9,
                x_offset=0,
                y_offset=0,
                width=1,
                height=1,
                mask=b"\x01",
            )
        ]


class FakeDistribution:
    name = "xlerobot-segmentation-test"
    version = "1.2.3"


class FakeEntryPoint:
    name = "test_v1"
    dist = FakeDistribution()

    def load(self):
        return lambda **kwargs: TestProvider()


class FakeEntryPoints(list):
    def select(self, *, group, name):
        assert group == "xlerobot.segmentation_providers"
        return [entry for entry in self if entry.name == name]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_config(tmp_path: Path, *, approved: bool = True) -> tuple[Path, Path, Path]:
    model = tmp_path / "model.bin"
    model.write_bytes(b"reviewed model")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "server_id": "test_local_v1",
                "model_sha256": _sha(model),
                "allowed_labels": ["cup"],
            }
        ),
        encoding="utf-8",
    )
    token = tmp_path / "token"
    token.write_text("0123456789abcdef0123456789abcdef\n", encoding="utf-8")
    config = tmp_path / "server.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "approved": approved,
                "bind_host": "127.0.0.1",
                "port": 8090,
                "manifest": {"path": manifest.name, "sha256": _sha(manifest)},
                "provider": {
                    "entry_point": "test_v1",
                    "distribution": "xlerobot-segmentation-test",
                    "version": "1.2.3",
                    "artifacts": [{"path": model.name, "sha256": _sha(model)}],
                    "options": {},
                },
                "model_artifact_index": 0,
                "token": {"path": token.name, "sha256": _sha(token)},
                "max_request_bytes": 4096,
                "max_image_bytes": 2048,
                "max_response_bytes": 4096,
                "max_instances": 4,
                "max_inference_seconds": 0.5,
            }
        ),
        encoding="utf-8",
    )
    return config, manifest, token


def _request() -> SegmentationRequest:
    return SegmentationRequest(
        server_id="test_local_v1",
        model_sha256="a" * 64,
        captured_at=1.0,
        frame_id="camera_frame",
        width=2,
        height=2,
        encoding="rgb8",
        step=6,
        queries={"cup": "drinking cup"},
        image=bytes(12),
    )


def test_server_config_binds_manifest_model_token_and_provider(tmp_path: Path):
    config_path, _, _ = _write_config(tmp_path)
    config = ServerConfig.load(config_path)
    assert config.server_id == "test_local_v1"
    assert config.provider_artifacts[0].sha256 == config.model_sha256
    assert config.token == "0123456789abcdef0123456789abcdef"
    provider = load_provider(config, entry_points=FakeEntryPoints([FakeEntryPoint()]))
    assert isinstance(provider, TestProvider)


def test_server_config_is_default_deny_and_checks_model_digest(tmp_path: Path):
    config_path, _, _ = _write_config(tmp_path, approved=False)
    with pytest.raises(ServerConfigError, match="approved"):
        ServerConfig.load(config_path)

    config_path, manifest, _ = _write_config(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["model_sha256"] = "b" * 64
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    config_raw = json.loads(config_path.read_text(encoding="utf-8"))
    config_raw["manifest"]["sha256"] = _sha(manifest)
    config_path.write_text(json.dumps(config_raw), encoding="utf-8")
    with pytest.raises(ServerConfigError, match="model artifact digest"):
        ServerConfig.load(config_path)


def test_provider_loader_rejects_distribution_mismatch(tmp_path: Path):
    config_path, _, _ = _write_config(tmp_path)
    config = replace(ServerConfig.load(config_path), provider_distribution="wrong")
    with pytest.raises(ProviderError, match="distribution"):
        load_provider(config, entry_points=FakeEntryPoints([FakeEntryPoint()]))


def test_protocol_rejects_unknown_labels_and_nonbinary_masks():
    raw = {
        "schema_version": 1,
        "server_id": "test_local_v1",
        "model_sha256": "a" * 64,
        "captured_at": 1.0,
        "frame_id": "camera_frame",
        "width": 2,
        "height": 2,
        "encoding": "rgb8",
        "step": 6,
        "queries": {"remote": "remote control"},
        "image_base64": "AAAAAAAAAAAAAAAA",
    }
    with pytest.raises(ProtocolError, match="not allowed"):
        parse_request(
            raw,
            server_id="test_local_v1",
            model_sha256="a" * 64,
            allowed_labels=frozenset({"cup"}),
            max_image_bytes=1024,
        )
    invalid = SegmentationInstance("cup-1", "cup", 0.8, 0, 0, 1, 1, b"\x02")
    with pytest.raises(ProtocolError, match="binary"):
        encode_response([invalid], _request(), max_instances=4)


def test_live_server_passes_black_box_conformance(tmp_path: Path):
    config_path, manifest, token = _write_config(tmp_path)
    config = replace(ServerConfig.load(config_path), port=0)
    server = create_server(config, TestProvider())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        report = evaluate_endpoint(
            endpoint=f"http://127.0.0.1:{server.server_port}/v1/segment",
            manifest_path=manifest,
            label="cup",
            prompt="drinking cup",
            token_path=token,
            timeout_seconds=1.0,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)
    assert report["passed"] is True
    assert report["instances"] == 1
    assert report["malformed_request_status"] == 400


def test_server_rejects_provider_output_outside_contract(tmp_path: Path):
    class InvalidProvider:
        def infer(self, segmentation_request):
            return [SegmentationInstance("bad", "cup", 0.5, 0, 0, 1, 1, b"\x02")]

    config_path, manifest, token = _write_config(tmp_path)
    config = replace(ServerConfig.load(config_path), port=0)
    server = create_server(config, InvalidProvider())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(ConformanceError, match="valid request returned HTTP 503"):
            evaluate_endpoint(
                endpoint=f"http://127.0.0.1:{server.server_port}/v1/segment",
                manifest_path=manifest,
                label="cup",
                prompt="cup",
                token_path=token,
                timeout_seconds=1.0,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)


def test_server_rejects_concurrent_inference_instead_of_racing_provider(tmp_path: Path):
    entered = threading.Event()
    release = threading.Event()

    class BlockingProvider:
        def infer(self, segmentation_request):
            entered.set()
            assert release.wait(timeout=2.0)
            return []

    config_path, _, _ = _write_config(tmp_path)
    config = replace(ServerConfig.load(config_path), port=0)
    server = create_server(config, BlockingProvider())
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1/segment"
    payload = {
        "schema_version": 1,
        "server_id": config.server_id,
        "model_sha256": config.model_sha256,
        "captured_at": 1.0,
        "frame_id": "camera_frame",
        "width": 2,
        "height": 2,
        "encoding": "rgb8",
        "step": 6,
        "queries": {"cup": "cup"},
        "image_base64": base64.b64encode(bytes(12)).decode("ascii"),
    }

    def send():
        outbound = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.token}",
            },
            method="POST",
        )
        with request.urlopen(outbound, timeout=2.0) as response:
            assert response.status == 200

    first = threading.Thread(target=send)
    first.start()
    assert entered.wait(timeout=1.0)
    try:
        with pytest.raises(error.HTTPError) as caught:
            send()
        assert caught.value.code == 503
        assert json.loads(caught.value.read())["error"] == "provider is busy"
    finally:
        release.set()
        first.join(timeout=2.0)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1.0)
