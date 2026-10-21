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

import yaml

from .episodes import EpisodeError, EpisodeManifest, load_episodes


class DataTriageError(ValueError):
    pass


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class LearningGate:
    task_ids: tuple[str, ...]
    executors: tuple[str, ...]
    failure_stages: tuple[str, ...]
    min_attempts: int
    min_learning_failures: int
    min_learning_failure_rate: float
    max_system_failure_rate: float
    max_unclassified_failure_rate: float
    max_intervention_rate: float
    coverage: dict[str, int]


@dataclass(frozen=True)
class DataTriageSpec:
    triage_id: str
    system_routes: dict[str, tuple[str, ...]]
    learning_gate: LearningGate
    fingerprint: str

    @classmethod
    def load(cls, path: str | Path) -> DataTriageSpec:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        expected = {"version", "approved", "triage_id", "system_routes", "learning_gate"}
        if not isinstance(raw, dict) or set(raw) != expected or raw.get("version") != 1:
            raise DataTriageError("data triage spec has missing or unexpected fields")
        if raw["approved"] is not True:
            raise DataTriageError("data triage spec must be approved")
        routes_raw = raw["system_routes"]
        if not isinstance(routes_raw, dict) or not 1 <= len(routes_raw) <= 32:
            raise DataTriageError("system_routes must contain 1..32 routes")
        routes: dict[str, tuple[str, ...]] = {}
        stage_owner: dict[str, str] = {}
        for route_id, stages_raw in routes_raw.items():
            route = _identifier(route_id, "system route ID")
            stages = _identifier_list(stages_raw, f"system route {route}", maximum=64)
            for stage in stages:
                owner = stage_owner.setdefault(stage, route)
                if owner != route:
                    raise DataTriageError(
                        f"failure stage {stage!r} is shared by routes {owner!r} and {route!r}"
                    )
            routes[route] = stages
        gate = _parse_learning_gate(raw["learning_gate"])
        overlap = sorted(set(stage_owner) & set(gate.failure_stages))
        if overlap:
            raise DataTriageError(
                "learning failure stages overlap system routes: " + ", ".join(overlap)
            )
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return cls(
            triage_id=_identifier(raw["triage_id"], "triage_id"),
            system_routes=dict(sorted(routes.items())),
            learning_gate=gate,
            fingerprint=hashlib.sha256(encoded).hexdigest(),
        )


@dataclass(frozen=True)
class LearningCandidate:
    candidate_id: str
    task_id: str
    task_fingerprint: str
    deployment_id: str
    deployment_fingerprint: str
    executor: str
    eligible: bool
    attempts: int
    successes: int
    learning_failures: int
    system_failures: int
    unclassified_failures: int
    intervention_rate: float
    coverage_values: dict[str, tuple[str, ...]]
    reasons: tuple[str, ...]
    recommended_episode_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "task_id": self.task_id,
            "task_fingerprint": self.task_fingerprint,
            "deployment_id": self.deployment_id,
            "deployment_fingerprint": self.deployment_fingerprint,
            "executor": self.executor,
            "eligible": self.eligible,
            "attempts": self.attempts,
            "successes": self.successes,
            "learning_failures": self.learning_failures,
            "system_failures": self.system_failures,
            "unclassified_failures": self.unclassified_failures,
            "intervention_rate": self.intervention_rate,
            "coverage_values": {
                key: list(values) for key, values in self.coverage_values.items()
            },
            "reasons": list(self.reasons),
            "recommended_episode_ids": list(self.recommended_episode_ids),
        }


@dataclass(frozen=True)
class DataTriageReport:
    triage_id: str
    spec_fingerprint: str
    generated_at: float
    episode_set_digest: str
    episode_count: int
    system_routes: dict[str, tuple[str, ...]]
    unclassified_episode_ids: tuple[str, ...]
    candidates: tuple[LearningCandidate, ...]
    report_sha256: str

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "triage_id": self.triage_id,
            "spec_fingerprint": self.spec_fingerprint,
            "generated_at": self.generated_at,
            "episode_set_digest": self.episode_set_digest,
            "episode_count": self.episode_count,
            "system_routes": {
                key: list(values) for key, values in self.system_routes.items()
            },
            "unclassified_episode_ids": list(self.unclassified_episode_ids),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "report_sha256": self.report_sha256}

    def validate(self) -> None:
        _identifier(self.triage_id, "triage_id")
        if not _HASH.fullmatch(self.spec_fingerprint):
            raise DataTriageError("triage spec fingerprint is invalid")
        if not math.isfinite(self.generated_at) or self.generated_at <= 0:
            raise DataTriageError("triage generated_at is invalid")
        if not _HASH.fullmatch(self.episode_set_digest):
            raise DataTriageError("triage episode set digest is invalid")
        if self.episode_count < 0:
            raise DataTriageError("triage episode count is invalid")
        if not _HASH.fullmatch(self.report_sha256):
            raise DataTriageError("triage report hash is invalid")
        if self.report_sha256 != _digest(self._hash_payload()):
            raise DataTriageError("triage report hash does not match content")

    @classmethod
    def from_dict(cls, raw: Any) -> DataTriageReport:
        expected = {
            "schema_version",
            "triage_id",
            "spec_fingerprint",
            "generated_at",
            "episode_set_digest",
            "episode_count",
            "system_routes",
            "unclassified_episode_ids",
            "candidates",
            "report_sha256",
        }
        if not isinstance(raw, dict) or set(raw) != expected or raw.get("schema_version") != 1:
            raise DataTriageError("triage report has missing or unexpected fields")
        routes_raw = raw["system_routes"]
        candidates_raw = raw["candidates"]
        if not isinstance(routes_raw, dict) or not isinstance(candidates_raw, list):
            raise DataTriageError("triage report collections are invalid")
        report = cls(
            triage_id=raw["triage_id"],
            spec_fingerprint=raw["spec_fingerprint"],
            generated_at=float(raw["generated_at"]),
            episode_set_digest=raw["episode_set_digest"],
            episode_count=raw["episode_count"],
            system_routes={
                str(key): tuple(value) for key, value in sorted(routes_raw.items())
            },
            unclassified_episode_ids=tuple(raw["unclassified_episode_ids"]),
            candidates=tuple(_candidate_from_dict(value) for value in candidates_raw),
            report_sha256=raw["report_sha256"],
        )
        report.validate()
        return report


def evaluate_data_triage(
    spec: DataTriageSpec,
    episodes: list[EpisodeManifest],
    *,
    generated_at: float | None = None,
) -> DataTriageReport:
    ordered = sorted(episodes, key=lambda value: value.episode_id)
    route_by_stage = {
        stage: route
        for route, stages in spec.system_routes.items()
        for stage in stages
    }
    routed: dict[str, list[str]] = {route: [] for route in spec.system_routes}
    unclassified: list[str] = []
    for episode in ordered:
        if episode.success:
            continue
        route = route_by_stage.get(episode.failure_stage)
        if route is not None:
            routed[route].append(episode.episode_id)
        elif episode.failure_stage not in spec.learning_gate.failure_stages:
            unclassified.append(episode.episode_id)
    groups: dict[tuple[str, str, str, str, str], list[EpisodeManifest]] = {}
    for episode in ordered:
        key = (
            episode.task_id,
            episode.task_fingerprint,
            episode.deployment_id,
            _deployment_fingerprint(episode),
            episode.executor,
        )
        groups.setdefault(key, []).append(episode)
    candidates = tuple(
        _evaluate_candidate(key, values, spec, route_by_stage)
        for key, values in sorted(groups.items())
    )
    timestamp = time.time() if generated_at is None else generated_at
    if not math.isfinite(timestamp) or timestamp <= 0:
        raise DataTriageError("triage generated_at must be positive and finite")
    episode_set_digest = _digest(
        [
            {"episode_id": episode.episode_id, "manifest_sha256": episode.manifest_sha256}
            for episode in ordered
        ]
    )
    system_route_values = {
        route: tuple(sorted(episode_ids))
        for route, episode_ids in sorted(routed.items())
    }
    unclassified_values = tuple(sorted(unclassified))
    payload = {
        "schema_version": 1,
        "triage_id": spec.triage_id,
        "spec_fingerprint": spec.fingerprint,
        "generated_at": timestamp,
        "episode_set_digest": episode_set_digest,
        "episode_count": len(ordered),
        "system_routes": {
            route: list(episode_ids) for route, episode_ids in system_route_values.items()
        },
        "unclassified_episode_ids": list(unclassified_values),
        "candidates": [candidate.to_dict() for candidate in candidates],
    }
    report = DataTriageReport(
        triage_id=spec.triage_id,
        spec_fingerprint=spec.fingerprint,
        generated_at=timestamp,
        episode_set_digest=episode_set_digest,
        episode_count=len(ordered),
        system_routes=system_route_values,
        unclassified_episode_ids=unclassified_values,
        candidates=candidates,
        report_sha256=_digest(payload),
    )
    report.validate()
    return report


def write_triage_report(path: str | Path, report: DataTriageReport) -> None:
    report.validate()
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


def load_triage_report(path: str | Path) -> DataTriageReport:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataTriageError(f"cannot load triage report: {exc}") from exc
    return DataTriageReport.from_dict(raw)


def _evaluate_candidate(
    key: tuple[str, str, str, str, str],
    episodes: list[EpisodeManifest],
    spec: DataTriageSpec,
    route_by_stage: dict[str, str],
) -> LearningCandidate:
    task_id, task_fingerprint, deployment_id, deployment_fingerprint, executor = key
    gate = spec.learning_gate
    successes = sum(episode.success for episode in episodes)
    learning = [
        episode
        for episode in episodes
        if not episode.success and episode.failure_stage in gate.failure_stages
    ]
    system = [
        episode
        for episode in episodes
        if not episode.success and episode.failure_stage in route_by_stage
    ]
    unclassified = [
        episode
        for episode in episodes
        if not episode.success
        and episode.failure_stage not in gate.failure_stages
        and episode.failure_stage not in route_by_stage
    ]
    attempts = len(episodes)
    reasons: list[str] = []
    if task_id not in gate.task_ids:
        reasons.append("task is not approved for learned fallback collection")
    if executor not in gate.executors:
        reasons.append("executor is not approved for learned fallback collection")
    if attempts < gate.min_attempts:
        reasons.append(f"attempts {attempts} are below {gate.min_attempts}")
    if len(learning) < gate.min_learning_failures:
        reasons.append(
            f"learning failures {len(learning)} are below {gate.min_learning_failures}"
        )
    _maximum_rate_reason(
        reasons,
        "system failure",
        len(system),
        attempts,
        gate.max_system_failure_rate,
    )
    _maximum_rate_reason(
        reasons,
        "unclassified failure",
        len(unclassified),
        attempts,
        gate.max_unclassified_failure_rate,
    )
    intervention_rate = _rate(
        sum(episode.human_intervention for episode in episodes), attempts
    )
    if intervention_rate > gate.max_intervention_rate:
        reasons.append(
            f"intervention rate {intervention_rate:.4f} exceeds "
            f"{gate.max_intervention_rate:.4f}"
        )
    effective = successes + len(learning)
    learning_rate = _rate(len(learning), effective)
    if learning_rate < gate.min_learning_failure_rate:
        reasons.append(
            f"learning failure rate {learning_rate:.4f} is below "
            f"{gate.min_learning_failure_rate:.4f}"
        )
    coverage_values: dict[str, tuple[str, ...]] = {}
    for axis, minimum in gate.coverage.items():
        values = tuple(
            sorted({episode.tags[axis] for episode in episodes if axis in episode.tags})
        )
        coverage_values[axis] = values
        if len(values) < minimum:
            reasons.append(
                f"coverage axis {axis} has {len(values)} values, below {minimum}"
            )
    candidate_id = _digest(
        {
            "task_id": task_id,
            "task_fingerprint": task_fingerprint,
            "deployment_id": deployment_id,
            "deployment_fingerprint": deployment_fingerprint,
            "executor": executor,
        }
    )[:24]
    return LearningCandidate(
        candidate_id=f"candidate-{candidate_id}",
        task_id=task_id,
        task_fingerprint=task_fingerprint,
        deployment_id=deployment_id,
        deployment_fingerprint=deployment_fingerprint,
        executor=executor,
        eligible=not reasons,
        attempts=attempts,
        successes=successes,
        learning_failures=len(learning),
        system_failures=len(system),
        unclassified_failures=len(unclassified),
        intervention_rate=intervention_rate,
        coverage_values=coverage_values,
        reasons=tuple(reasons),
        recommended_episode_ids=(
            tuple(sorted(episode.episode_id for episode in learning)) if not reasons else ()
        ),
    )


def _parse_learning_gate(raw: Any) -> LearningGate:
    expected = {
        "task_ids",
        "executors",
        "failure_stages",
        "min_attempts",
        "min_learning_failures",
        "min_learning_failure_rate",
        "max_system_failure_rate",
        "max_unclassified_failure_rate",
        "max_intervention_rate",
        "coverage",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise DataTriageError("learning_gate has missing or unexpected fields")
    coverage_raw = raw["coverage"]
    if not isinstance(coverage_raw, dict) or not 1 <= len(coverage_raw) <= 16:
        raise DataTriageError("learning coverage must contain 1..16 axes")
    coverage = {
        _identifier(key, "coverage axis"): _integer(value, f"coverage {key}", 1, 64)
        for key, value in coverage_raw.items()
    }
    return LearningGate(
        task_ids=_identifier_list(raw["task_ids"], "learning task_ids", maximum=128),
        executors=_identifier_list(raw["executors"], "learning executors", maximum=16),
        failure_stages=_identifier_list(
            raw["failure_stages"], "learning failure_stages", maximum=64
        ),
        min_attempts=_integer(raw["min_attempts"], "min_attempts", 1, 100_000),
        min_learning_failures=_integer(
            raw["min_learning_failures"], "min_learning_failures", 1, 100_000
        ),
        min_learning_failure_rate=_fraction(
            raw["min_learning_failure_rate"], "min_learning_failure_rate"
        ),
        max_system_failure_rate=_fraction(
            raw["max_system_failure_rate"], "max_system_failure_rate"
        ),
        max_unclassified_failure_rate=_fraction(
            raw["max_unclassified_failure_rate"], "max_unclassified_failure_rate"
        ),
        max_intervention_rate=_fraction(
            raw["max_intervention_rate"], "max_intervention_rate"
        ),
        coverage=dict(sorted(coverage.items())),
    )


def _candidate_from_dict(raw: Any) -> LearningCandidate:
    expected = {
        "candidate_id",
        "task_id",
        "task_fingerprint",
        "deployment_id",
        "deployment_fingerprint",
        "executor",
        "eligible",
        "attempts",
        "successes",
        "learning_failures",
        "system_failures",
        "unclassified_failures",
        "intervention_rate",
        "coverage_values",
        "reasons",
        "recommended_episode_ids",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise DataTriageError("learning candidate has invalid fields")
    coverage = raw["coverage_values"]
    if not isinstance(coverage, dict):
        raise DataTriageError("learning candidate coverage is invalid")
    return LearningCandidate(
        candidate_id=raw["candidate_id"],
        task_id=raw["task_id"],
        task_fingerprint=raw["task_fingerprint"],
        deployment_id=raw["deployment_id"],
        deployment_fingerprint=raw["deployment_fingerprint"],
        executor=raw["executor"],
        eligible=raw["eligible"],
        attempts=raw["attempts"],
        successes=raw["successes"],
        learning_failures=raw["learning_failures"],
        system_failures=raw["system_failures"],
        unclassified_failures=raw["unclassified_failures"],
        intervention_rate=raw["intervention_rate"],
        coverage_values={key: tuple(values) for key, values in sorted(coverage.items())},
        reasons=tuple(raw["reasons"]),
        recommended_episode_ids=tuple(raw["recommended_episode_ids"]),
    )


def _maximum_rate_reason(
    reasons: list[str],
    label: str,
    count: int,
    total: int,
    maximum: float,
) -> None:
    actual = _rate(count, total)
    if actual > maximum:
        reasons.append(f"{label} rate {actual:.4f} exceeds {maximum:.4f}")


def _deployment_fingerprint(episode: EpisodeManifest) -> str:
    value = episode.execution_binding.get("deployment_fingerprint")
    return value if isinstance(value, str) and _HASH.fullmatch(value) else "unbound"


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise DataTriageError(f"{label} is invalid")
    return value


def _identifier_list(value: Any, label: str, *, maximum: int) -> tuple[str, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= maximum:
        raise DataTriageError(f"{label} must contain 1..{maximum} values")
    values = tuple(_identifier(item, label) for item in value)
    if len(values) != len(set(values)):
        raise DataTriageError(f"{label} values must be unique")
    return values


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise DataTriageError(f"{label} must be in [{minimum}, {maximum}]")
    return value


def _fraction(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataTriageError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise DataTriageError(f"{label} must be in [0, 1]")
    return result


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Route system failures and gate learned-fallback data collection"
    )
    parser.add_argument("spec")
    parser.add_argument("episodes")
    parser.add_argument("output")
    args = parser.parse_args()
    try:
        report = evaluate_data_triage(
            DataTriageSpec.load(args.spec),
            load_episodes(args.episodes),
        )
        write_triage_report(args.output, report)
    except (OSError, EpisodeError, DataTriageError, yaml.YAMLError) as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(2) from exc
    print(json.dumps({"success": True, **report.to_dict()}, ensure_ascii=False, indent=2))


__all__ = [
    "DataTriageError",
    "DataTriageReport",
    "DataTriageSpec",
    "LearningCandidate",
    "LearningGate",
    "evaluate_data_triage",
    "load_triage_report",
    "write_triage_report",
]
