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

from .episode_evidence import EpisodeEvidenceError, load_episode_evidence
from .episodes import EpisodeManifest, load_episodes


class BenchmarkError(ValueError):
    pass


@dataclass(frozen=True)
class BenchmarkGates:
    min_episodes: int
    min_episodes_per_value: int
    min_success_rate: float
    max_intervention_rate: float
    max_failure_stage_rates: dict[str, float]


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    task_id: str
    task_fingerprint: str
    binding_equals: dict[str, Any]
    coverage: dict[str, tuple[str, ...]]
    required_evidence_kinds: tuple[str, ...]
    gates: BenchmarkGates
    fingerprint: str

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkSpec:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        common = {
            "version",
            "benchmark_id",
            "task_id",
            "task_fingerprint",
            "binding_equals",
            "coverage",
            "gates",
        }
        version = raw.get("version") if isinstance(raw, dict) else None
        expected = common if version == 1 else common | {"required_evidence_kinds"}
        if not isinstance(raw, dict) or version not in {1, 2} or set(raw) != expected:
            raise BenchmarkError("benchmark spec has missing or unexpected fields")
        benchmark_id = _identifier(raw["benchmark_id"], "benchmark_id")
        task_id = _identifier(raw["task_id"], "task_id")
        task_fingerprint = _sha(raw["task_fingerprint"], "task_fingerprint")
        binding = _binding(raw["binding_equals"])
        coverage = _coverage(raw["coverage"])
        required_evidence = (
            _identifier_list(raw["required_evidence_kinds"], "required_evidence_kinds")
            if version == 2
            else ()
        )
        gates = _gates(raw["gates"])
        payload = {
            "version": version,
            "benchmark_id": benchmark_id,
            "task_id": task_id,
            "task_fingerprint": task_fingerprint,
            "binding_equals": binding,
            "coverage": {key: list(value) for key, value in coverage.items()},
            "gates": {
                "min_episodes": gates.min_episodes,
                "min_episodes_per_value": gates.min_episodes_per_value,
                "min_success_rate": gates.min_success_rate,
                "max_intervention_rate": gates.max_intervention_rate,
                "max_failure_stage_rates": gates.max_failure_stage_rates,
            },
        }
        if version == 2:
            payload["required_evidence_kinds"] = list(required_evidence)
        return cls(
            benchmark_id=benchmark_id,
            task_id=task_id,
            task_fingerprint=task_fingerprint,
            binding_equals=binding,
            coverage=coverage,
            required_evidence_kinds=required_evidence,
            gates=gates,
            fingerprint=_digest(payload),
        )


@dataclass(frozen=True)
class BenchmarkReport:
    schema_version: int
    benchmark_id: str
    benchmark_fingerprint: str
    task_id: str
    task_fingerprint: str
    binding_equals: dict[str, Any]
    required_evidence_kinds: tuple[str, ...]
    generated_at: float
    episode_ids: tuple[str, ...]
    episode_set_digest: str
    success_rate: float
    intervention_rate: float
    failure_stage_rates: dict[str, float]
    coverage_counts: dict[str, dict[str, int]]
    passed: bool
    failures: tuple[str, ...]
    report_sha256: str

    @classmethod
    def create(
        cls,
        spec: BenchmarkSpec,
        episodes: list[EpisodeManifest],
        *,
        evidence_sha256_by_episode: dict[str, str] | None = None,
        generated_at: float | None = None,
    ) -> BenchmarkReport:
        selected = sorted(
            (episode for episode in episodes if _matches(spec, episode)),
            key=lambda episode: episode.episode_id,
        )
        total = len(selected)
        success_rate = sum(episode.success for episode in selected) / total if total else 0.0
        intervention_rate = (
            sum(episode.human_intervention for episode in selected) / total if total else 0.0
        )
        stages = sorted({episode.failure_stage for episode in selected if episode.failure_stage})
        stage_rates = {
            stage: sum(episode.failure_stage == stage for episode in selected) / total
            for stage in stages
        } if total else {}
        coverage_counts = {
            axis: {
                value: sum(episode.tags.get(axis) == value for episode in selected)
                for value in values
            }
            for axis, values in spec.coverage.items()
        }
        failures: list[str] = []
        if total < spec.gates.min_episodes:
            failures.append(f"episode count {total} is below {spec.gates.min_episodes}")
        if success_rate < spec.gates.min_success_rate:
            failures.append(
                f"success rate {success_rate:.4f} is below {spec.gates.min_success_rate:.4f}"
            )
        if intervention_rate > spec.gates.max_intervention_rate:
            failures.append(
                f"intervention rate {intervention_rate:.4f} exceeds "
                f"{spec.gates.max_intervention_rate:.4f}"
            )
        for stage, maximum in spec.gates.max_failure_stage_rates.items():
            actual = stage_rates.get(stage, 0.0)
            if actual > maximum:
                failures.append(
                    f"failure stage {stage} rate {actual:.4f} exceeds {maximum:.4f}"
                )
        for axis, counts in coverage_counts.items():
            for value, count in counts.items():
                if count < spec.gates.min_episodes_per_value:
                    failures.append(
                        f"coverage {axis}={value} count {count} is below "
                        f"{spec.gates.min_episodes_per_value}"
                    )
        identifiers = tuple(episode.episode_id for episode in selected)
        evidence_hashes = evidence_sha256_by_episode or {}
        if spec.required_evidence_kinds:
            missing = sorted(set(identifiers) - set(evidence_hashes))
            if missing:
                raise BenchmarkError(
                    "benchmark episodes are missing required evidence: " + ", ".join(missing)
                )
            for episode_id in identifiers:
                _sha(evidence_hashes[episode_id], "episode evidence sha256")
        report = cls(
            schema_version=2,
            benchmark_id=spec.benchmark_id,
            benchmark_fingerprint=spec.fingerprint,
            task_id=spec.task_id,
            task_fingerprint=spec.task_fingerprint,
            binding_equals=dict(spec.binding_equals),
            required_evidence_kinds=spec.required_evidence_kinds,
            generated_at=time.time() if generated_at is None else generated_at,
            episode_ids=identifiers,
            episode_set_digest=_digest(
                [
                    {
                        "episode_id": episode.episode_id,
                        "manifest_sha256": episode.manifest_sha256,
                        **(
                            {"evidence_sha256": evidence_hashes[episode.episode_id]}
                            if spec.required_evidence_kinds
                            else {}
                        ),
                    }
                    for episode in selected
                ]
            ),
            success_rate=success_rate,
            intervention_rate=intervention_rate,
            failure_stage_rates=stage_rates,
            coverage_counts=coverage_counts,
            passed=not failures,
            failures=tuple(failures),
            report_sha256="",
        )
        report.validate(include_hash=False)
        return cls(**{**report.__dict__, "report_sha256": _digest(report._hash_payload())})

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> BenchmarkReport:
        common = {
            "schema_version",
            "benchmark_id",
            "benchmark_fingerprint",
            "task_id",
            "task_fingerprint",
            "binding_equals",
            "generated_at",
            "episode_ids",
            "episode_set_digest",
            "success_rate",
            "intervention_rate",
            "failure_stage_rates",
            "coverage_counts",
            "passed",
            "failures",
            "report_sha256",
        }
        version = raw.get("schema_version") if isinstance(raw, dict) else None
        expected = common if version == 1 else common | {"required_evidence_kinds"}
        if not isinstance(raw, dict) or version not in {1, 2} or set(raw) != expected:
            raise BenchmarkError("benchmark report has missing or unexpected fields")
        report = cls(
            schema_version=version,
            benchmark_id=raw["benchmark_id"],
            benchmark_fingerprint=raw["benchmark_fingerprint"],
            task_id=raw["task_id"],
            task_fingerprint=raw["task_fingerprint"],
            binding_equals=raw["binding_equals"],
            required_evidence_kinds=(
                tuple(raw["required_evidence_kinds"]) if version == 2 else ()
            ),
            generated_at=raw["generated_at"],
            episode_ids=tuple(raw["episode_ids"]),
            episode_set_digest=raw["episode_set_digest"],
            success_rate=raw["success_rate"],
            intervention_rate=raw["intervention_rate"],
            failure_stage_rates=raw["failure_stage_rates"],
            coverage_counts=raw["coverage_counts"],
            passed=raw["passed"],
            failures=tuple(raw["failures"]),
            report_sha256=raw["report_sha256"],
        )
        report.validate(include_hash=True)
        return report

    def validate(self, *, include_hash: bool) -> None:
        if self.schema_version not in {1, 2}:
            raise BenchmarkError("benchmark report schema version is invalid")
        _identifier(self.benchmark_id, "benchmark_id")
        _identifier(self.task_id, "task_id")
        _sha(self.benchmark_fingerprint, "benchmark_fingerprint")
        _sha(self.task_fingerprint, "task_fingerprint")
        _binding(self.binding_equals)
        if self.schema_version == 1 and self.required_evidence_kinds:
            raise BenchmarkError("legacy benchmark report cannot require evidence")
        if self.schema_version == 2:
            _validated_identifier_tuple(
                self.required_evidence_kinds,
                "required_evidence_kinds",
                allow_empty=True,
            )
        if (
            isinstance(self.generated_at, bool)
            or not isinstance(self.generated_at, (int, float))
            or not math.isfinite(self.generated_at)
            or self.generated_at <= 0
        ):
            raise BenchmarkError("benchmark report timestamp is invalid")
        if len(self.episode_ids) != len(set(self.episode_ids)) or not all(
            _is_identifier(value) for value in self.episode_ids
        ):
            raise BenchmarkError("benchmark report episode IDs are invalid")
        _sha(self.episode_set_digest, "episode_set_digest")
        for name, value in {
            "success_rate": self.success_rate,
            "intervention_rate": self.intervention_rate,
            **self.failure_stage_rates,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise BenchmarkError(f"benchmark rate {name!r} is invalid")
        if not isinstance(self.coverage_counts, dict):
            raise BenchmarkError("benchmark coverage counts must be an object")
        for axis, counts in self.coverage_counts.items():
            _identifier(axis, "coverage axis")
            if not isinstance(counts, dict):
                raise BenchmarkError("benchmark coverage axis counts must be an object")
            for value, count in counts.items():
                _identifier(value, f"coverage {axis}")
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise BenchmarkError("benchmark coverage count is invalid")
        if not isinstance(self.passed, bool) or not all(
            isinstance(value, str) and value for value in self.failures
        ):
            raise BenchmarkError("benchmark verdict is invalid")
        if self.passed == bool(self.failures):
            raise BenchmarkError("benchmark passed flag contradicts failures")
        if include_hash:
            _sha(self.report_sha256, "report_sha256")
            if self.report_sha256 != _digest(self._hash_payload()):
                raise BenchmarkError("benchmark report hash does not match content")

    def require_candidate(
        self,
        *,
        task_id: str,
        task_fingerprint: str,
        execution_binding: dict[str, Any],
        now: float | None = None,
        max_age_seconds: float | None = None,
    ) -> None:
        if not self.passed:
            raise BenchmarkError("candidate benchmark did not pass")
        if self.task_id != task_id or self.task_fingerprint != task_fingerprint:
            raise BenchmarkError("benchmark report belongs to another task contract")
        for key, expected in self.binding_equals.items():
            if execution_binding.get(key) != expected:
                raise BenchmarkError(f"benchmark report does not cover binding key {key!r}")
        if max_age_seconds is not None:
            current = time.time() if now is None else now
            age = current - self.generated_at
            if not math.isfinite(age) or age < 0 or age > max_age_seconds:
                raise BenchmarkError("benchmark report is stale or from the future")

    def _hash_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "benchmark_id": self.benchmark_id,
            "benchmark_fingerprint": self.benchmark_fingerprint,
            "task_id": self.task_id,
            "task_fingerprint": self.task_fingerprint,
            "binding_equals": self.binding_equals,
            "generated_at": self.generated_at,
            "episode_ids": list(self.episode_ids),
            "episode_set_digest": self.episode_set_digest,
            "success_rate": self.success_rate,
            "intervention_rate": self.intervention_rate,
            "failure_stage_rates": self.failure_stage_rates,
            "coverage_counts": self.coverage_counts,
            "passed": self.passed,
            "failures": list(self.failures),
        }
        if self.schema_version == 2:
            payload["required_evidence_kinds"] = list(self.required_evidence_kinds)
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "report_sha256": self.report_sha256}


def evaluate_benchmark(
    spec: BenchmarkSpec,
    episodes: list[EpisodeManifest],
    *,
    evidence_sha256_by_episode: dict[str, str] | None = None,
) -> BenchmarkReport:
    return BenchmarkReport.create(
        spec,
        episodes,
        evidence_sha256_by_episode=evidence_sha256_by_episode,
    )


def evaluate_benchmark_store(spec: BenchmarkSpec, root: str | Path) -> BenchmarkReport:
    episodes = load_episodes(root)
    evidence_hashes: dict[str, str] = {}
    if spec.required_evidence_kinds:
        for episode in episodes:
            if not _matches(spec, episode):
                continue
            try:
                evidence = load_episode_evidence(
                    Path(root) / episode.episode_id,
                    required_kinds=spec.required_evidence_kinds,
                )
            except EpisodeEvidenceError as exc:
                raise BenchmarkError(
                    f"benchmark episode {episode.episode_id} evidence is invalid: {exc}"
                ) from exc
            evidence_hashes[episode.episode_id] = evidence.evidence_sha256
    return evaluate_benchmark(
        spec,
        episodes,
        evidence_sha256_by_episode=evidence_hashes,
    )


def load_report(path: str | Path) -> BenchmarkReport:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"cannot load benchmark report: {exc}") from exc
    return BenchmarkReport.from_dict(raw)


def write_report(path: str | Path, report: BenchmarkReport) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate immutable robot episode manifests")
    parser.add_argument("spec")
    parser.add_argument("episodes")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        report = evaluate_benchmark_store(BenchmarkSpec.load(args.spec), args.episodes)
        write_report(args.output, report)
    except (BenchmarkError, OSError, ValueError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(2) from exc
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    raise SystemExit(0 if report.passed else 1)


def _matches(spec: BenchmarkSpec, episode: EpisodeManifest) -> bool:
    if episode.task_id != spec.task_id or episode.task_fingerprint != spec.task_fingerprint:
        return False
    return all(
        episode.execution_binding.get(key) == value
        for key, value in spec.binding_equals.items()
    )


def _gates(raw: Any) -> BenchmarkGates:
    expected = {
        "min_episodes",
        "min_episodes_per_value",
        "min_success_rate",
        "max_intervention_rate",
        "max_failure_stage_rates",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise BenchmarkError("benchmark gates have missing or unexpected fields")
    min_episodes = _positive_int(raw["min_episodes"], "min_episodes", maximum=100000)
    per_value = _positive_int(
        raw["min_episodes_per_value"],
        "min_episodes_per_value",
        maximum=100000,
    )
    stage_rates = raw["max_failure_stage_rates"]
    if not isinstance(stage_rates, dict) or len(stage_rates) > 64:
        raise BenchmarkError("max_failure_stage_rates must be a bounded object")
    parsed_rates = {
        _identifier(key, "failure stage"): _rate(value, f"failure stage {key}")
        for key, value in stage_rates.items()
    }
    return BenchmarkGates(
        min_episodes=min_episodes,
        min_episodes_per_value=per_value,
        min_success_rate=_rate(raw["min_success_rate"], "min_success_rate"),
        max_intervention_rate=_rate(
            raw["max_intervention_rate"],
            "max_intervention_rate",
        ),
        max_failure_stage_rates=dict(sorted(parsed_rates.items())),
    )


def _coverage(raw: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw, dict) or not raw or len(raw) > 16:
        raise BenchmarkError("benchmark coverage must be a non-empty bounded object")
    coverage: dict[str, tuple[str, ...]] = {}
    for axis, values in raw.items():
        key = _identifier(axis, "coverage axis")
        if not isinstance(values, list) or not values or len(values) > 64:
            raise BenchmarkError(f"coverage axis {key!r} values are invalid")
        parsed = tuple(_identifier(value, f"coverage {key}") for value in values)
        if len(parsed) != len(set(parsed)):
            raise BenchmarkError(f"coverage axis {key!r} has duplicate values")
        coverage[key] = parsed
    return dict(sorted(coverage.items()))


def _binding(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or not raw or len(raw) > 16:
        raise BenchmarkError("binding_equals must be a non-empty bounded object")
    binding: dict[str, Any] = {}
    for key, value in raw.items():
        name = _identifier(key, "binding key")
        if value is None or isinstance(value, (dict, list, float)):
            raise BenchmarkError(f"binding value {name!r} must be a scalar")
        if not isinstance(value, (str, int, bool)):
            raise BenchmarkError(f"binding value {name!r} is invalid")
        if isinstance(value, str) and (not value or len(value) > 256):
            raise BenchmarkError(f"binding value {name!r} is invalid")
        binding[name] = value
    return dict(sorted(binding.items()))


def _identifier_list(raw: Any, label: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw or len(raw) > 128:
        raise BenchmarkError(f"{label} must be a non-empty bounded list")
    values = tuple(_identifier(value, label) for value in raw)
    if len(values) != len(set(values)):
        raise BenchmarkError(f"{label} contains duplicates")
    return values


def _validated_identifier_tuple(
    raw: Any,
    label: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    if (
        not isinstance(raw, tuple)
        or len(raw) > 128
        or (not allow_empty and not raw)
    ):
        raise BenchmarkError(f"{label} must be a bounded tuple")
    values = tuple(_identifier(value, label) for value in raw)
    if len(values) != len(set(values)):
        raise BenchmarkError(f"{label} contains duplicates")
    return values


def _positive_int(value: Any, label: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise BenchmarkError(f"{label} must be in [1, {maximum}]")
    return value


def _rate(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkError(f"{label} must be a rate")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise BenchmarkError(f"{label} must be in [0, 1]")
    return parsed


def _identifier(value: Any, label: str) -> str:
    if not _is_identifier(value):
        raise BenchmarkError(f"{label} is not a safe identifier")
    return value


def _is_identifier(value: Any) -> bool:
    return isinstance(value, str) and bool(
        re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}", value)
    )


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise BenchmarkError(f"{label} must be a SHA-256 digest")
    return value


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    main()


__all__ = [
    "BenchmarkError",
    "BenchmarkGates",
    "BenchmarkReport",
    "BenchmarkSpec",
    "evaluate_benchmark",
    "evaluate_benchmark_store",
    "load_report",
    "main",
    "write_report",
]
