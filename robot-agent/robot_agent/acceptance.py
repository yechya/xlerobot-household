from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class AcceptanceError(ValueError):
    pass


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_REPORT_TYPES = frozenset({"nav2_hil", "manipulation_hil"})


@dataclass(frozen=True)
class DeploymentAcceptanceReport:
    report_id: str
    report_type: str
    target_fingerprint: str
    generated_at: float
    valid_until: float
    payload: dict[str, Any]
    report_sha256: str

    @classmethod
    def create(
        cls,
        *,
        report_id: str,
        report_type: str,
        target_fingerprint: str,
        generated_at: float,
        valid_until: float,
        payload: dict[str, Any],
    ) -> DeploymentAcceptanceReport:
        report = cls(
            report_id=report_id,
            report_type=report_type,
            target_fingerprint=target_fingerprint,
            generated_at=generated_at,
            valid_until=valid_until,
            payload=_json_object(payload, "acceptance payload"),
            report_sha256="",
        )
        report.validate(include_hash=False)
        return cls(
            **{
                **report.__dict__,
                "report_sha256": _digest(report._hash_payload()),
            }
        )

    @classmethod
    def from_dict(cls, raw: Any) -> DeploymentAcceptanceReport:
        expected = {
            "schema_version",
            "report_id",
            "report_type",
            "target_fingerprint",
            "generated_at",
            "valid_until",
            "payload",
            "report_sha256",
        }
        if not isinstance(raw, dict) or set(raw) != expected or raw.get("schema_version") != 1:
            raise AcceptanceError("acceptance report has missing or unexpected fields")
        report = cls(
            report_id=raw["report_id"],
            report_type=raw["report_type"],
            target_fingerprint=raw["target_fingerprint"],
            generated_at=raw["generated_at"],
            valid_until=raw["valid_until"],
            payload=_json_object(raw["payload"], "acceptance payload"),
            report_sha256=raw["report_sha256"],
        )
        report.validate(include_hash=True)
        return report

    def validate(self, *, include_hash: bool) -> None:
        _identifier(self.report_id, "report_id")
        if self.report_type not in _REPORT_TYPES:
            raise AcceptanceError("acceptance report_type is unsupported")
        _sha(self.target_fingerprint, "target_fingerprint")
        for value, label in (
            (self.generated_at, "generated_at"),
            (self.valid_until, "valid_until"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise AcceptanceError(f"acceptance {label} is invalid")
        if self.valid_until <= self.generated_at:
            raise AcceptanceError("acceptance valid_until must follow generated_at")
        _json_object(self.payload, "acceptance payload")
        if (
            self.payload.get("evidence_class") in {"simulation_binding", "simulation_only"}
            or self.payload.get("production_authorization") is False
            or self.payload.get("hil_acceptance") is False
        ):
            raise AcceptanceError("simulation evidence cannot be bound as HIL acceptance")
        if include_hash:
            _sha(self.report_sha256, "report_sha256")
            if self.report_sha256 != _digest(self._hash_payload()):
                raise AcceptanceError("acceptance report hash does not match content")

    def require(
        self,
        *,
        report_type: str,
        target_fingerprint: str,
        now: float | None = None,
    ) -> dict[str, Any]:
        self.validate(include_hash=True)
        if self.report_type != report_type:
            raise AcceptanceError("acceptance report type does not match the requested gate")
        if self.target_fingerprint != target_fingerprint:
            raise AcceptanceError("acceptance target fingerprint does not match deployment")
        current = time.time() if now is None else now
        if not math.isfinite(current) or current <= 0:
            raise AcceptanceError("acceptance current time is invalid")
        if current > self.valid_until:
            raise AcceptanceError("acceptance report has expired")
        if self.payload.get("passed") is not True:
            raise AcceptanceError("acceptance payload has not passed")
        return dict(self.payload)

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "report_id": self.report_id,
            "report_type": self.report_type,
            "target_fingerprint": self.target_fingerprint,
            "generated_at": self.generated_at,
            "valid_until": self.valid_until,
            "payload": self.payload,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "report_sha256": self.report_sha256}


def load_acceptance_report(path: str | Path) -> DeploymentAcceptanceReport:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcceptanceError(f"cannot load acceptance report: {exc}") from exc
    return DeploymentAcceptanceReport.from_dict(raw)


def write_acceptance_report(
    path: str | Path,
    report: DeploymentAcceptanceReport,
) -> None:
    report.validate(include_hash=True)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
        destination.chmod(0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _json_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AcceptanceError(f"{label} must be an object")
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise AcceptanceError(f"{label} must be JSON-safe") from exc
    if len(encoded.encode("utf-8")) > 2 * 1024 * 1024:
        raise AcceptanceError(f"{label} is too large")
    return json.loads(encoded)


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise AcceptanceError(f"acceptance {label} is invalid")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise AcceptanceError(f"acceptance {label} must be SHA-256")
    return value


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bind a passed HIL result to one deployment-domain fingerprint"
    )
    parser.add_argument("report_type", choices=sorted(_REPORT_TYPES))
    parser.add_argument("target_fingerprint")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--report-id", required=True)
    parser.add_argument("--valid-seconds", type=float, required=True)
    args = parser.parse_args()
    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        now = time.time()
        report = DeploymentAcceptanceReport.create(
            report_id=args.report_id,
            report_type=args.report_type,
            target_fingerprint=args.target_fingerprint,
            generated_at=now,
            valid_until=now + args.valid_seconds,
            payload=payload,
        )
        report.require(
            report_type=args.report_type,
            target_fingerprint=args.target_fingerprint,
            now=now,
        )
        write_acceptance_report(args.output, report)
    except (OSError, json.JSONDecodeError, AcceptanceError) as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(2) from exc
    print(json.dumps({"success": True, **report.to_dict()}, ensure_ascii=False, indent=2))


__all__ = [
    "AcceptanceError",
    "DeploymentAcceptanceReport",
    "load_acceptance_report",
    "write_acceptance_report",
]
