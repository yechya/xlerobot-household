from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

try:
    from .contract import AuditEvent, Refusal, TokenMetadata
except ImportError:
    _CONTRACT_PATH = Path(__file__).with_name("contract.py")
    _CONTRACT_SPEC = importlib.util.spec_from_file_location(
        "robot_agent_contract_for_audit",
        _CONTRACT_PATH,
    )
    if _CONTRACT_SPEC is None or _CONTRACT_SPEC.loader is None:
        raise
    _contract = importlib.util.module_from_spec(_CONTRACT_SPEC)
    sys.modules[_CONTRACT_SPEC.name] = _contract
    _CONTRACT_SPEC.loader.exec_module(_contract)
    AuditEvent = _contract.AuditEvent
    Refusal = _contract.Refusal
    TokenMetadata = _contract.TokenMetadata


_SECRET_KEY_PARTS = (
    "authorization",
    "api_key",
    "apikey",
    "bearer",
    "password",
    "secret",
    "token",
)

_SAFE_TOKEN_FIELDS = {
    "token_id",
    "token_hash",
    "token_redacted",
}

_SAFE_TOKEN_METADATA_FIELDS = {
    "issued_to",
    "scopes",
    "session_id",
    "token_id",
}


def token_hash(token_value: str) -> str:
    return hashlib.sha256(token_value.encode("utf-8")).hexdigest()[:16]


def redact_token_value(token_value: str) -> str:
    if not token_value:
        return "redacted"
    return f"sha256:{token_hash(token_value)}:redacted"


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in _SAFE_TOKEN_FIELDS:
        return False
    return any(part in lowered for part in _SECRET_KEY_PARTS)


def _is_safe_token_metadata(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    keys = {str(k) for k in value}
    return "token_id" in keys and keys <= _SAFE_TOKEN_METADATA_FIELDS


def _redact_string(value: str, token_value: str = "") -> str:
    if not token_value:
        return value
    return value.replace(token_value, "[REDACTED_TOKEN]")


def _redact_jsonish(
    value: Any,
    key: str = "",
    token_value: str = "",
    parent_secret: bool = False,
) -> Any:
    secret_context = parent_secret or (
        _is_secret_key(key) and not _is_safe_token_metadata(value)
    )

    if secret_context:
        if isinstance(value, str):
            return redact_token_value(value)
        if isinstance(value, dict):
            return {
                str(k): _redact_jsonish(v, str(k), token_value, True)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [_redact_jsonish(item, key, token_value, True) for item in value]
        if isinstance(value, tuple):
            return [_redact_jsonish(item, key, token_value, True) for item in value]
        return "redacted"
    if isinstance(value, dict):
        return {
            str(k): _redact_jsonish(v, str(k), token_value)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_jsonish(item, token_value=token_value) for item in value]
    if isinstance(value, tuple):
        return [_redact_jsonish(item, token_value=token_value) for item in value]
    if isinstance(value, str):
        return _redact_string(value, token_value)
    return value


def _token_metadata_details(token: TokenMetadata) -> dict[str, Any]:
    return {
        "token_id": token.token_id,
        "issued_to": token.issued_to,
        "session_id": token.session_id,
        "scopes": list(token.scopes),
    }


def build_audit_event(
    *,
    decision: str,
    state: str,
    command_id: str = "",
    task_id: str = "",
    actor: str = "",
    message: str = "",
    details: dict[str, Any] | None = None,
    refusal: Refusal | None = None,
    token: TokenMetadata | None = None,
    token_value: str = "",
    timestamp: float = 0.0,
) -> AuditEvent:
    normalized_decision = decision.strip().lower()
    if normalized_decision not in {"accepted", "refused"}:
        raise ValueError("decision must be 'accepted' or 'refused'")

    safe_details = _redact_jsonish(details or {}, token_value=token_value)
    safe_details["decision"] = normalized_decision

    if token is not None:
        safe_details["token"] = _token_metadata_details(token)
    if token_value:
        safe_details["token_hash"] = token_hash(token_value)
        safe_details["token_redacted"] = redact_token_value(token_value)
        message = message.replace(token_value, "[REDACTED_TOKEN]")
    if refusal is not None:
        safe_details["refusal"] = {
            "reason_code": refusal.reason_code,
            "state": refusal.state,
            "message": refusal.message.replace(token_value, "[REDACTED_TOKEN]")
            if token_value
            else refusal.message,
            "details": _redact_jsonish(refusal.details, token_value=token_value),
        }

    return AuditEvent(
        event_type=f"command_{normalized_decision}",
        state=state,
        command_id=command_id,
        task_id=task_id,
        actor=actor,
        message=message,
        details=safe_details,
        timestamp=timestamp,
    )


def audit_event_to_dict(event: AuditEvent) -> dict[str, Any]:
    return {
        "event_type": event.event_type,
        "state": event.state,
        "command_id": event.command_id,
        "task_id": event.task_id,
        "actor": event.actor,
        "message": _redact_jsonish(event.message),
        "details": _redact_jsonish(event.details),
        "timestamp": event.timestamp,
    }


class JsonlAuditWriter:
    """Append-only, owner-readable audit sink for the standalone daemon."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def __call__(self, event: AuditEvent) -> None:
        payload = json.dumps(
            audit_event_to_dict(event),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


__all__ = [
    "JsonlAuditWriter",
    "audit_event_to_dict",
    "build_audit_event",
    "redact_token_value",
    "token_hash",
]
