from __future__ import annotations

import argparse
import hmac
import json
import socket
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import metadata
from typing import Any

from .config import ServerConfig
from .protocol import ProtocolError, encode_response, parse_request

ENTRY_POINT_GROUP = "xlerobot.segmentation_providers"


class ProviderError(RuntimeError):
    pass


def load_provider(config: ServerConfig, *, entry_points: Any | None = None) -> Any:
    available = metadata.entry_points() if entry_points is None else entry_points
    if hasattr(available, "select"):
        candidates = list(
            available.select(group=ENTRY_POINT_GROUP, name=config.provider_entry_point)
        )
    else:
        candidates = [
            entry
            for entry in available.get(ENTRY_POINT_GROUP, ())
            if entry.name == config.provider_entry_point
        ]
    if len(candidates) != 1:
        raise ProviderError(f"expected exactly one {ENTRY_POINT_GROUP} provider entry point")
    entry = candidates[0]
    distribution = getattr(entry, "dist", None)
    name = getattr(distribution, "name", None)
    version = getattr(distribution, "version", None)
    if name is None and distribution is not None:
        name = distribution.metadata.get("Name")
    if _normalize(name) != _normalize(config.provider_distribution):
        raise ProviderError("provider distribution does not match config")
    if version != config.provider_version:
        raise ProviderError("provider version does not match config")
    factory = entry.load()
    provider = factory(
        options=dict(config.provider_options),
        artifacts=tuple(artifact.path for artifact in config.provider_artifacts),
        server_id=config.server_id,
        model_sha256=config.model_sha256,
        allowed_labels=config.allowed_labels,
    )
    if not callable(getattr(provider, "infer", None)):
        raise ProviderError("provider must implement infer(request)")
    return provider


def create_server(config: ServerConfig, provider: Any) -> ThreadingHTTPServer:
    inference_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "XLeRobotSegmentation/1"

        def do_POST(self) -> None:
            if self.path != "/v1/segment":
                self._error(HTTPStatus.NOT_FOUND, "route not found")
                return
            if config.token is not None:
                supplied = self.headers.get("Authorization", "")
                if not hmac.compare_digest(supplied, f"Bearer {config.token}"):
                    self._error(HTTPStatus.UNAUTHORIZED, "authentication failed")
                    return
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].lower()
            if content_type != "application/json":
                self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "application/json required")
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                length = -1
            if not 1 <= length <= config.max_request_bytes:
                self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request size is invalid")
                return
            body = self.rfile.read(length)
            try:
                raw = json.loads(body.decode("utf-8"))
                segmentation_request = parse_request(
                    raw,
                    server_id=config.server_id,
                    model_sha256=config.model_sha256,
                    allowed_labels=config.allowed_labels,
                    max_image_bytes=config.max_image_bytes,
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ProtocolError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            if not inference_lock.acquire(blocking=False):
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, "provider is busy")
                return
            started = time.monotonic()
            try:
                instances = provider.infer(segmentation_request)
                response = encode_response(
                    instances, segmentation_request, max_instances=config.max_instances
                )
            except Exception:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, "provider inference failed")
                return
            finally:
                inference_lock.release()
            if time.monotonic() - started > config.max_inference_seconds:
                self._error(HTTPStatus.GATEWAY_TIMEOUT, "provider exceeded inference deadline")
                return
            payload = json.dumps(
                response, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            if len(payload) > config.max_response_bytes:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, "response exceeds reviewed size")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            self._error(HTTPStatus.METHOD_NOT_ALLOWED, "method not allowed")

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _error(self, status: HTTPStatus, message: str) -> None:
            payload = json.dumps(
                {"schema_version": 1, "error": message}, separators=(",", ":")
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

    class Server(ThreadingHTTPServer):
        address_family = socket.AF_INET6 if config.bind_host == "::1" else socket.AF_INET
        daemon_threads = True

    return Server((config.bind_host, config.port), Handler)


def _normalize(value: Any) -> str:
    return str(value or "").lower().replace("_", "-")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local XLeRobot segmentation service")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = ServerConfig.load(args.config)
    provider = load_provider(config)
    server = create_server(config, provider)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


__all__ = ["ENTRY_POINT_GROUP", "ProviderError", "create_server", "load_provider", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
