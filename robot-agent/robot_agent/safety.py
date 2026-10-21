from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class SafetyError(ValueError):
    pass


@dataclass(frozen=True)
class SafetySnapshot:
    sequence: int
    observed_at: float
    source: str
    sensor_health: str
    physical_estop_available: bool
    emergency_stop_pressed: bool
    person_count: int
    nearest_person_m: float | None
    pet_count: int
    nearest_pet_m: float | None
    person_in_workspace: bool
    pet_in_workspace: bool
    privacy_zone_active: bool
    snapshot_sha256: str

    @classmethod
    def create(
        cls,
        *,
        sequence: int,
        observed_at: float,
        source: str,
        sensor_health: str,
        physical_estop_available: bool,
        emergency_stop_pressed: bool,
        person_count: int,
        nearest_person_m: float | None,
        pet_count: int,
        nearest_pet_m: float | None,
        person_in_workspace: bool,
        pet_in_workspace: bool,
        privacy_zone_active: bool,
    ) -> "SafetySnapshot":
        snapshot = cls(
            sequence=sequence,
            observed_at=observed_at,
            source=source,
            sensor_health=sensor_health,
            physical_estop_available=physical_estop_available,
            emergency_stop_pressed=emergency_stop_pressed,
            person_count=person_count,
            nearest_person_m=nearest_person_m,
            pet_count=pet_count,
            nearest_pet_m=nearest_pet_m,
            person_in_workspace=person_in_workspace,
            pet_in_workspace=pet_in_workspace,
            privacy_zone_active=privacy_zone_active,
            snapshot_sha256="",
        )
        snapshot.validate(include_hash=False)
        return cls(**{**snapshot.__dict__, "snapshot_sha256": _digest(snapshot._hash_payload())})

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SafetySnapshot":
        expected = {
            "schema_version",
            "sequence",
            "observed_at",
            "source",
            "sensor_health",
            "physical_estop_available",
            "emergency_stop_pressed",
            "person_count",
            "nearest_person_m",
            "pet_count",
            "nearest_pet_m",
            "person_in_workspace",
            "pet_in_workspace",
            "privacy_zone_active",
            "snapshot_sha256",
        }
        if not isinstance(raw, dict) or set(raw) != expected or raw.get("schema_version") != 1:
            raise SafetyError("safety snapshot has missing or unexpected fields")
        snapshot = cls(
            sequence=raw["sequence"],
            observed_at=raw["observed_at"],
            source=raw["source"],
            sensor_health=raw["sensor_health"],
            physical_estop_available=raw["physical_estop_available"],
            emergency_stop_pressed=raw["emergency_stop_pressed"],
            person_count=raw["person_count"],
            nearest_person_m=raw["nearest_person_m"],
            pet_count=raw["pet_count"],
            nearest_pet_m=raw["nearest_pet_m"],
            person_in_workspace=raw["person_in_workspace"],
            pet_in_workspace=raw["pet_in_workspace"],
            privacy_zone_active=raw["privacy_zone_active"],
            snapshot_sha256=raw["snapshot_sha256"],
        )
        snapshot.validate(include_hash=True)
        return snapshot

    def validate(self, *, include_hash: bool) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence <= 0
        ):
            raise SafetyError("safety snapshot sequence must be positive")
        if (
            isinstance(self.observed_at, bool)
            or not isinstance(self.observed_at, (int, float))
            or not math.isfinite(self.observed_at)
            or self.observed_at <= 0
        ):
            raise SafetyError("safety snapshot timestamp is invalid")
        if not isinstance(self.source, str) or not 0 < len(self.source) <= 256:
            raise SafetyError("safety snapshot source is invalid")
        if self.sensor_health not in {"ok", "degraded", "lost"}:
            raise SafetyError("safety sensor health is invalid")
        booleans = (
            self.physical_estop_available,
            self.emergency_stop_pressed,
            self.person_in_workspace,
            self.pet_in_workspace,
            self.privacy_zone_active,
        )
        if not all(isinstance(value, bool) for value in booleans):
            raise SafetyError("safety snapshot flags must be booleans")
        _validate_count_distance(self.person_count, self.nearest_person_m, "person")
        _validate_count_distance(self.pet_count, self.nearest_pet_m, "pet")
        if include_hash:
            if not _is_hash(self.snapshot_sha256):
                raise SafetyError("safety snapshot hash is invalid")
            if self.snapshot_sha256 != _digest(self._hash_payload()):
                raise SafetyError("safety snapshot hash does not match content")

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "sequence": self.sequence,
            "observed_at": self.observed_at,
            "source": self.source,
            "sensor_health": self.sensor_health,
            "physical_estop_available": self.physical_estop_available,
            "emergency_stop_pressed": self.emergency_stop_pressed,
            "person_count": self.person_count,
            "nearest_person_m": self.nearest_person_m,
            "pet_count": self.pet_count,
            "nearest_pet_m": self.nearest_pet_m,
            "person_in_workspace": self.person_in_workspace,
            "pet_in_workspace": self.pet_in_workspace,
            "privacy_zone_active": self.privacy_zone_active,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "snapshot_sha256": self.snapshot_sha256}


@dataclass(frozen=True)
class SafetyPolicy:
    max_age_seconds: float
    allowed_source_prefixes: tuple[str, ...]
    require_physical_estop: bool
    deny_when_privacy_active: bool
    motion_min_person_distance_m: float
    motion_min_pet_distance_m: float
    manipulation_min_person_distance_m: float
    manipulation_min_pet_distance_m: float
    require_clear_manipulation_workspace: bool

    @classmethod
    def load(cls, path: str | Path) -> "SafetyPolicy":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        expected = {
            "version",
            "max_age_seconds",
            "allowed_source_prefixes",
            "require_physical_estop",
            "deny_when_privacy_active",
            "motion",
            "manipulation",
        }
        if not isinstance(raw, dict) or set(raw) != expected or raw.get("version") != 1:
            raise SafetyError("safety policy has missing or unexpected fields")
        sources = raw["allowed_source_prefixes"]
        if not isinstance(sources, list) or not sources or not all(
            isinstance(source, str) and 0 < len(source) <= 128 for source in sources
        ):
            raise SafetyError("safety policy source prefixes are invalid")
        motion = raw["motion"]
        manipulation = raw["manipulation"]
        if not isinstance(motion, dict) or set(motion) != {
            "min_person_distance_m",
            "min_pet_distance_m",
        }:
            raise SafetyError("safety motion policy is invalid")
        if not isinstance(manipulation, dict) or set(manipulation) != {
            "min_person_distance_m",
            "min_pet_distance_m",
            "require_clear_workspace",
        }:
            raise SafetyError("safety manipulation policy is invalid")
        for key in ("require_physical_estop", "deny_when_privacy_active"):
            if not isinstance(raw[key], bool):
                raise SafetyError(f"safety policy {key} must be boolean")
        if not isinstance(manipulation["require_clear_workspace"], bool):
            raise SafetyError("safety workspace policy must be boolean")
        return cls(
            max_age_seconds=_positive_float(raw["max_age_seconds"], "max_age_seconds"),
            allowed_source_prefixes=tuple(sources),
            require_physical_estop=raw["require_physical_estop"],
            deny_when_privacy_active=raw["deny_when_privacy_active"],
            motion_min_person_distance_m=_positive_float(
                motion["min_person_distance_m"],
                "motion person distance",
            ),
            motion_min_pet_distance_m=_positive_float(
                motion["min_pet_distance_m"],
                "motion pet distance",
            ),
            manipulation_min_person_distance_m=_positive_float(
                manipulation["min_person_distance_m"],
                "manipulation person distance",
            ),
            manipulation_min_pet_distance_m=_positive_float(
                manipulation["min_pet_distance_m"],
                "manipulation pet distance",
            ),
            require_clear_manipulation_workspace=manipulation[
                "require_clear_workspace"
            ],
        )


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    operation: str
    reasons: tuple[str, ...]
    snapshot_sequence: int | None = None
    snapshot_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "operation": self.operation,
            "reasons": list(self.reasons),
            "snapshot_sequence": self.snapshot_sequence,
            "snapshot_sha256": self.snapshot_sha256,
        }


class FileSafetyInterlock:
    def __init__(self, snapshot_path: str | Path, policy: SafetyPolicy):
        self.snapshot_path = Path(snapshot_path)
        self.policy = policy
        self._lock = threading.Lock()
        self._last_sequence = 0
        self._last_hash = ""

    def check(self, operation: str, *, now: float) -> SafetyDecision:
        if operation not in {"motion", "manipulation"}:
            return SafetyDecision(False, operation, ("unsupported safety operation",))
        try:
            snapshot = self._load_snapshot()
            reasons = self._evaluate(snapshot, operation, now)
            with self._lock:
                if snapshot.sequence < self._last_sequence:
                    reasons.append("safety snapshot sequence rolled back")
                elif (
                    snapshot.sequence == self._last_sequence
                    and self._last_hash
                    and snapshot.snapshot_sha256 != self._last_hash
                ):
                    reasons.append("safety snapshot changed without a new sequence")
                if not reasons and snapshot.sequence >= self._last_sequence:
                    self._last_sequence = snapshot.sequence
                    self._last_hash = snapshot.snapshot_sha256
            return SafetyDecision(
                allowed=not reasons,
                operation=operation,
                reasons=tuple(reasons),
                snapshot_sequence=snapshot.sequence,
                snapshot_sha256=snapshot.snapshot_sha256,
            )
        except (OSError, json.JSONDecodeError, SafetyError) as exc:
            return SafetyDecision(False, operation, (f"safety snapshot unavailable: {exc}",))

    def health(self, *, now: float) -> dict[str, Any]:
        motion = self.check("motion", now=now)
        manipulation = self.check("manipulation", now=now)
        return {
            "available": motion.snapshot_sequence is not None,
            "motion_allowed": motion.allowed,
            "manipulation_allowed": manipulation.allowed,
            "motion_reasons": list(motion.reasons),
            "manipulation_reasons": list(manipulation.reasons),
            "snapshot_sequence": motion.snapshot_sequence,
            "snapshot_sha256": motion.snapshot_sha256,
        }

    def _load_snapshot(self) -> SafetySnapshot:
        if self.snapshot_path.stat().st_mode & 0o077:
            raise SafetyError("safety snapshot must not be accessible by group or others")
        raw = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        return SafetySnapshot.from_dict(raw)

    def _evaluate(
        self,
        snapshot: SafetySnapshot,
        operation: str,
        now: float,
    ) -> list[str]:
        reasons: list[str] = []
        age = now - snapshot.observed_at
        if not math.isfinite(age) or age < 0 or age > self.policy.max_age_seconds:
            reasons.append("safety snapshot is stale or from the future")
        if not any(
            snapshot.source.startswith(prefix)
            for prefix in self.policy.allowed_source_prefixes
        ):
            reasons.append("safety snapshot source is not allowlisted")
        if snapshot.sensor_health != "ok":
            reasons.append(f"safety sensor health is {snapshot.sensor_health}")
        if self.policy.require_physical_estop and not snapshot.physical_estop_available:
            reasons.append("physical emergency stop is unavailable")
        if snapshot.emergency_stop_pressed:
            reasons.append("physical emergency stop is pressed")
        if self.policy.deny_when_privacy_active and snapshot.privacy_zone_active:
            reasons.append("privacy zone is active")
        if operation == "motion":
            _distance_reason(
                reasons,
                "person",
                snapshot.person_count,
                snapshot.nearest_person_m,
                self.policy.motion_min_person_distance_m,
            )
            _distance_reason(
                reasons,
                "pet",
                snapshot.pet_count,
                snapshot.nearest_pet_m,
                self.policy.motion_min_pet_distance_m,
            )
        else:
            _distance_reason(
                reasons,
                "person",
                snapshot.person_count,
                snapshot.nearest_person_m,
                self.policy.manipulation_min_person_distance_m,
            )
            _distance_reason(
                reasons,
                "pet",
                snapshot.pet_count,
                snapshot.nearest_pet_m,
                self.policy.manipulation_min_pet_distance_m,
            )
            if self.policy.require_clear_manipulation_workspace:
                if snapshot.person_in_workspace:
                    reasons.append("person is inside the manipulation workspace")
                if snapshot.pet_in_workspace:
                    reasons.append("pet is inside the manipulation workspace")
        return reasons


class UnavailableSafetyInterlock:
    def __init__(self, reason: str = "safety interlock is not configured"):
        self.reason = reason

    def check(self, operation: str, *, now: float) -> SafetyDecision:
        return SafetyDecision(False, operation, (self.reason,))

    def health(self, *, now: float) -> dict[str, Any]:
        return {"available": False, "reason": self.reason}


def write_safety_snapshot(path: str | Path, snapshot: SafetySnapshot) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(snapshot.to_dict(), handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
        destination.chmod(0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _validate_count_distance(count: Any, distance: Any, label: str) -> None:
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 1000:
        raise SafetyError(f"{label} count is invalid")
    if count == 0 and distance is not None:
        raise SafetyError(f"{label} distance must be null when count is zero")
    if count > 0:
        if (
            isinstance(distance, bool)
            or not isinstance(distance, (int, float))
            or not math.isfinite(distance)
            or distance < 0
        ):
            raise SafetyError(f"{label} distance is invalid")


def _distance_reason(
    reasons: list[str],
    label: str,
    count: int,
    distance: float | None,
    minimum: float,
) -> None:
    if count > 0 and distance is not None and distance < minimum:
        reasons.append(f"nearest {label} distance {distance:.3f}m is below {minimum:.3f}m")


def _positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SafetyError(f"{label} must be positive")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise SafetyError(f"{label} must be positive")
    return parsed


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "FileSafetyInterlock",
    "SafetyDecision",
    "SafetyError",
    "SafetyPolicy",
    "SafetySnapshot",
    "UnavailableSafetyInterlock",
    "write_safety_snapshot",
]
