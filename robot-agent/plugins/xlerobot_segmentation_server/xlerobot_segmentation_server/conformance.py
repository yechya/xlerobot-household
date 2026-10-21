from __future__ import annotations

import argparse
import base64
import json
import math
import re
import time
from pathlib import Path
from urllib import error, parse, request

from .protocol import validate_manifest

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class ConformanceError(ValueError):
    pass


def evaluate_endpoint(
    *,
    endpoint: str,
    manifest_path: str | Path,
    label: str,
    prompt: str,
    token_path: str | Path | None,
    timeout_seconds: float,
) -> dict[str, object]:
    endpoint = _loopback_endpoint(endpoint)
    try:
        manifest_raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        server_id, model_sha256, allowed_labels = validate_manifest(manifest_raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ConformanceError("manifest is invalid") from exc
    if label not in allowed_labels:
        raise ConformanceError("conformance label is not allowed by the manifest")
    token = None
    if token_path is not None:
        try:
            token = Path(token_path).read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as exc:
            raise ConformanceError("token cannot be read") from exc
    width, height, step = 2, 2, 6
    payload = {
        "schema_version": 1,
        "server_id": server_id,
        "model_sha256": model_sha256,
        "captured_at": time.time(),
        "frame_id": "conformance_camera_frame",
        "width": width,
        "height": height,
        "encoding": "rgb8",
        "step": step,
        "queries": {label: prompt},
        "image_base64": base64.b64encode(bytes(step * height)).decode("ascii"),
    }
    started = time.monotonic()
    status, response = _post(endpoint, payload, token, timeout_seconds)
    latency = time.monotonic() - started
    if status != 200:
        raise ConformanceError(f"valid request returned HTTP {status}")
    instance_count = _validate_response(response, payload)
    malformed = dict(payload)
    malformed["unexpected"] = True
    malformed_status, _ = _post(endpoint, malformed, token, timeout_seconds)
    if not 400 <= malformed_status < 500:
        raise ConformanceError("malformed request was not rejected with a client error")
    return {
        "schema_version": 1,
        "server_id": server_id,
        "model_sha256": model_sha256,
        "label": label,
        "latency_seconds": latency,
        "instances": instance_count,
        "malformed_request_status": malformed_status,
        "passed": True,
    }


def _post(
    endpoint: str, payload: dict[str, object], token: str | None, timeout: float
) -> tuple[int, object]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    outbound = request.Request(
        endpoint,
        data=json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with request.build_opener(_NoRedirect()).open(outbound, timeout=timeout) as response:
            status = int(response.status)
            body = response.read(64 * 1024 * 1024 + 1)
    except error.HTTPError as exc:
        status = int(exc.code)
        body = exc.read(1024 * 1024)
    except (error.URLError, TimeoutError, OSError) as exc:
        raise ConformanceError("segmentation endpoint request failed") from exc
    try:
        decoded = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConformanceError("segmentation endpoint returned invalid JSON") from exc
    return status, decoded


def _validate_response(raw: object, sent: dict[str, object]) -> int:
    expected = {"schema_version", "server_id", "model_sha256", "instances"}
    if not isinstance(raw, dict) or set(raw) != expected or raw.get("schema_version") != 1:
        raise ConformanceError("response has missing or unexpected fields")
    if raw["server_id"] != sent["server_id"] or raw["model_sha256"] != sent["model_sha256"]:
        raise ConformanceError("response identity does not match the manifest")
    instances = raw["instances"]
    if not isinstance(instances, list) or len(instances) > 4096:
        raise ConformanceError("response instances are invalid")
    expected_instance = {
        "instance_id",
        "label",
        "confidence",
        "x_offset",
        "y_offset",
        "width",
        "height",
        "mask_base64",
    }
    seen: set[str] = set()
    for instance in instances:
        if not isinstance(instance, dict) or set(instance) != expected_instance:
            raise ConformanceError("response instance has missing or unexpected fields")
        instance_id = instance["instance_id"]
        label = instance["label"]
        if (
            not isinstance(instance_id, str)
            or not _IDENTIFIER.fullmatch(instance_id)
            or not isinstance(label, str)
            or not _IDENTIFIER.fullmatch(label)
            or label not in sent["queries"]
            or instance_id in seen
        ):
            raise ConformanceError("response contains an invalid label or duplicate instance ID")
        seen.add(instance_id)
        values = (
            instance["x_offset"],
            instance["y_offset"],
            instance["width"],
            instance["height"],
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ConformanceError("response ROI is invalid")
        if (
            values[0] < 0
            or values[1] < 0
            or values[2] <= 0
            or values[3] <= 0
            or values[0] + values[2] > sent["width"]
            or values[1] + values[3] > sent["height"]
        ):
            raise ConformanceError("response ROI is outside the image")
        confidence = instance["confidence"]
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise ConformanceError("response confidence is invalid")
        try:
            mask = base64.b64decode(instance["mask_base64"], validate=True)
        except (TypeError, ValueError) as exc:
            raise ConformanceError("response mask base64 is invalid") from exc
        if len(mask) != values[2] * values[3] or any(value not in (0, 1) for value in mask):
            raise ConformanceError("response mask is not a binary ROI mask")
    return len(instances)


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _loopback_endpoint(value: str) -> str:
    parsed = parse.urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
        or parsed.port is None
        or parsed.path != "/v1/segment"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ConformanceError("endpoint must be loopback HTTP /v1/segment")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a local segmentation HTTP v1 service")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--token")
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        report = evaluate_endpoint(
            endpoint=args.endpoint,
            manifest_path=args.manifest,
            label=args.label,
            prompt=args.prompt,
            token_path=args.token,
            timeout_seconds=args.timeout_seconds,
        )
    except ConformanceError as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


__all__ = ["ConformanceError", "evaluate_endpoint", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
