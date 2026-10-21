from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from robot_agent.acceptance import (
    AcceptanceError,
    DeploymentAcceptanceReport,
    load_acceptance_report,
    write_acceptance_report,
)


def _report(*, passed: bool = True) -> DeploymentAcceptanceReport:
    return DeploymentAcceptanceReport.create(
        report_id="nav2-home-v1",
        report_type="nav2_hil",
        target_fingerprint="a" * 64,
        generated_at=100.0,
        valid_until=200.0,
        payload={"passed": passed, "scenarios": []},
    )


def test_acceptance_report_binds_type_fingerprint_and_expiry():
    report = _report()
    assert report.require(
        report_type="nav2_hil",
        target_fingerprint="a" * 64,
        now=150.0,
    )["passed"] is True
    with pytest.raises(AcceptanceError, match="type does not match"):
        report.require(
            report_type="manipulation_hil",
            target_fingerprint="a" * 64,
            now=150.0,
        )
    with pytest.raises(AcceptanceError, match="fingerprint does not match"):
        report.require(
            report_type="nav2_hil",
            target_fingerprint="b" * 64,
            now=150.0,
        )
    with pytest.raises(AcceptanceError, match="expired"):
        report.require(
            report_type="nav2_hil",
            target_fingerprint="a" * 64,
            now=201.0,
        )


def test_acceptance_report_requires_passed_payload():
    with pytest.raises(AcceptanceError, match="has not passed"):
        _report(passed=False).require(
            report_type="nav2_hil",
            target_fingerprint="a" * 64,
            now=150.0,
        )


def test_acceptance_report_is_private_and_detects_tampering(tmp_path: Path):
    path = tmp_path / "acceptance" / "nav2.json"
    report = _report()
    write_acceptance_report(path, report)
    assert load_acceptance_report(path) == report
    assert oct(path.stat().st_mode & 0o777) == "0o600"

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["target_fingerprint"] = "b" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(AcceptanceError, match="hash does not match"):
        load_acceptance_report(path)


def test_acceptance_rejects_invalid_lifetime_and_type():
    with pytest.raises(AcceptanceError, match="must follow"):
        DeploymentAcceptanceReport.create(
            report_id="bad-lifetime",
            report_type="nav2_hil",
            target_fingerprint="a" * 64,
            generated_at=100.0,
            valid_until=100.0,
            payload={"passed": True},
        )
    with pytest.raises(AcceptanceError, match="unsupported"):
        DeploymentAcceptanceReport.create(
            report_id="bad-type",
            report_type="arbitrary",
            target_fingerprint="a" * 64,
            generated_at=100.0,
            valid_until=200.0,
            payload={"passed": True},
        )


def test_acceptance_rejects_explicit_simulation_only_evidence():
    with pytest.raises(AcceptanceError, match="simulation evidence"):
        DeploymentAcceptanceReport.create(
            report_id="sim-must-not-arm",
            report_type="manipulation_hil",
            target_fingerprint="a" * 64,
            generated_at=100.0,
            valid_until=200.0,
            payload={
                "passed": True,
                "evidence_class": "simulation_only",
                "production_authorization": False,
                "hil_acceptance": False,
            },
        )


def test_acceptance_loader_rejects_hashed_simulation_payload(tmp_path: Path):
    path = tmp_path / "simulation-as-acceptance.json"
    raw = _report().to_dict()
    raw["payload"] = {
        "passed": True,
        "evidence_class": "simulation_only",
        "production_authorization": False,
        "hil_acceptance": False,
    }
    hash_payload = {key: value for key, value in raw.items() if key != "report_sha256"}
    raw["report_sha256"] = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(AcceptanceError, match="simulation evidence"):
        load_acceptance_report(path)
