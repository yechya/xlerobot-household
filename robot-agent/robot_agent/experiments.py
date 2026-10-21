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

from .episodes import EpisodeManifest, load_episodes


class ExperimentError(ValueError):
    pass


EXECUTION_MODES = (
    "direct_vla",
    "rgb_precondition_vla",
    "rgb_pre_post_vla",
)
_MODE_FIELDS = {
    "mode",
    "episode_ids",
    "episode_count",
    "success_rate",
    "intervention_rate",
    "failure_stage_rates",
    "coverage_counts",
    "sufficient",
}


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    task_id: str
    task_fingerprint: str
    modes: tuple[str, ...]
    binding_equals: dict[str, Any]
    coverage: dict[str, tuple[str, ...]]
    min_episodes_per_mode: int
    report_fingerprint: str

    @classmethod
    def load(cls, path: str | Path) -> ExperimentSpec:
        import yaml

        config_path = Path(path).expanduser().resolve()
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ExperimentError(f"cannot load experiment spec: {exc}") from exc
        expected = {
            "version",
            "experiment_id",
            "task_id",
            "task_fingerprint",
            "modes",
            "binding_equals",
            "coverage",
            "min_episodes_per_mode",
        }
        if not isinstance(raw, dict) or set(raw) != expected or raw.get("version") != 1:
            raise ExperimentError("experiment spec has missing or unexpected fields")
        experiment_id = _identifier(raw["experiment_id"], "experiment_id")
        task_id = _identifier(raw["task_id"], "task_id")
        task_fingerprint = _sha(raw["task_fingerprint"], "task_fingerprint")
        modes = _modes(raw["modes"])
        binding_equals = _binding(raw["binding_equals"])
        coverage = _coverage(raw["coverage"])
        min_episodes = _positive_int(
            raw["min_episodes_per_mode"],
            "min_episodes_per_mode",
            maximum=100000,
        )
        payload = {
            "version": 1,
            "experiment_id": experiment_id,
            "task_id": task_id,
            "task_fingerprint": task_fingerprint,
            "modes": list(modes),
            "binding_equals": binding_equals,
            "coverage": {key: list(value) for key, value in coverage.items()},
            "min_episodes_per_mode": min_episodes,
        }
        return cls(
            experiment_id=experiment_id,
            task_id=task_id,
            task_fingerprint=task_fingerprint,
            modes=modes,
            binding_equals=binding_equals,
            coverage=coverage,
            min_episodes_per_mode=min_episodes,
            report_fingerprint=_digest(payload),
        )


@dataclass(frozen=True)
class ExperimentModeReport:
    mode: str
    episode_ids: tuple[str, ...]
    success_rate: float
    intervention_rate: float
    failure_stage_rates: dict[str, float]
    coverage_counts: dict[str, dict[str, int]]
    sufficient: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "episode_ids": list(self.episode_ids),
            "episode_count": len(self.episode_ids),
            "success_rate": self.success_rate,
            "intervention_rate": self.intervention_rate,
            "failure_stage_rates": dict(self.failure_stage_rates),
            "coverage_counts": self.coverage_counts,
            "sufficient": self.sufficient,
        }


@dataclass(frozen=True)
class ExperimentReport:
    schema_version: int
    experiment_id: str
    spec_fingerprint: str
    task_id: str
    task_fingerprint: str
    generated_at: float
    modes: tuple[ExperimentModeReport, ...]
    comparison: dict[str, Any]
    passed: bool
    failures: tuple[str, ...]
    report_sha256: str

    @classmethod
    def create(
        cls,
        spec: ExperimentSpec,
        episodes: list[EpisodeManifest],
        *,
        generated_at: float | None = None,
    ) -> ExperimentReport:
        episode_ids = [episode.episode_id for episode in episodes]
        if len(episode_ids) != len(set(episode_ids)):
            raise ExperimentError("experiment input contains duplicate episode IDs")
        selected = [
            episode
            for episode in episodes
            if episode.task_id == spec.task_id
            and episode.task_fingerprint == spec.task_fingerprint
            and all(
                episode.execution_binding.get(key) == value
                for key, value in spec.binding_equals.items()
            )
            and episode.execution_binding.get("experiment_mode") in spec.modes
        ]
        mode_reports: list[ExperimentModeReport] = []
        failures: list[str] = []
        for mode in spec.modes:
            values = sorted(
                (
                    episode
                    for episode in selected
                    if episode.execution_binding.get("experiment_mode") == mode
                ),
                key=lambda episode: episode.episode_id,
            )
            total = len(values)
            success_rate = sum(episode.success for episode in values) / total if total else 0.0
            intervention_rate = (
                sum(episode.human_intervention for episode in values) / total if total else 0.0
            )
            stages = sorted({episode.failure_stage for episode in values if episode.failure_stage})
            stage_rates = {
                stage: sum(episode.failure_stage == stage for episode in values) / total
                for stage in stages
            } if total else {}
            coverage_counts = {
                axis: {
                    value: sum(episode.tags.get(axis) == value for episode in values)
                    for value in values_for_axis
                }
                for axis, values_for_axis in spec.coverage.items()
            }
            sufficient = total >= spec.min_episodes_per_mode
            if not sufficient:
                failures.append(
                    f"mode {mode} episode count {total} is below "
                    f"{spec.min_episodes_per_mode}"
                )
            mode_reports.append(
                ExperimentModeReport(
                    mode=mode,
                    episode_ids=tuple(episode.episode_id for episode in values),
                    success_rate=success_rate,
                    intervention_rate=intervention_rate,
                    failure_stage_rates=stage_rates,
                    coverage_counts=coverage_counts,
                    sufficient=sufficient,
                )
            )
        comparison = _comparison(mode_reports)
        if not comparison["direct_vla_episode_count"]:
            failures.append("direct_vla has no usable episodes for comparison")
        if not comparison["best_mode"]:
            failures.append("no execution mode has usable episodes")
        report = cls(
            schema_version=1,
            experiment_id=spec.experiment_id,
            spec_fingerprint=spec.report_fingerprint,
            task_id=spec.task_id,
            task_fingerprint=spec.task_fingerprint,
            generated_at=time.time() if generated_at is None else generated_at,
            modes=tuple(mode_reports),
            comparison=comparison,
            passed=not failures,
            failures=tuple(failures),
            report_sha256="",
        )
        report.validate(include_hash=False)
        return cls(**{**report.__dict__, "report_sha256": _digest(report._hash_payload())})

    def validate(self, *, include_hash: bool) -> None:
        if self.schema_version != 1:
            raise ExperimentError("experiment report schema version is invalid")
        _identifier(self.experiment_id, "experiment_id")
        _sha(self.spec_fingerprint, "spec_fingerprint")
        _identifier(self.task_id, "task_id")
        _sha(self.task_fingerprint, "task_fingerprint")
        if (
            not isinstance(self.generated_at, (int, float))
            or isinstance(self.generated_at, bool)
            or not math.isfinite(self.generated_at)
            or self.generated_at <= 0
        ):
            raise ExperimentError("experiment report timestamp is invalid")
        if not self.modes:
            raise ExperimentError("experiment report must contain modes")
        modes = [self._validate_mode_report(mode) for mode in self.modes]
        if len(modes) != len(set(modes)):
            raise ExperimentError("experiment report contains duplicate modes")
        if "direct_vla" not in modes:
            raise ExperimentError("experiment report must contain direct_vla")
        _validate_comparison(self.comparison)
        if not isinstance(self.passed, bool) or not isinstance(self.failures, tuple) or not all(
            isinstance(value, str) and value for value in self.failures
        ):
            raise ExperimentError("experiment verdict is invalid")
        if self.passed != (not self.failures):
            raise ExperimentError("experiment passed flag contradicts failures")
        if include_hash:
            _sha(self.report_sha256, "report_sha256")
            if self.report_sha256 != _digest(self._hash_payload()):
                raise ExperimentError("experiment report hash does not match content")

    @staticmethod
    def _validate_mode_report(mode: ExperimentModeReport) -> str:
        _identifier(mode.mode, "mode")
        if mode.mode not in EXECUTION_MODES:
            raise ExperimentError(f"mode must be one of {EXECUTION_MODES}")
        if not isinstance(mode.episode_ids, tuple) or not all(
            isinstance(value, str) and value for value in mode.episode_ids
        ):
            raise ExperimentError(f"experiment mode {mode.mode} episode IDs are invalid")
        if len(mode.episode_ids) != len(set(mode.episode_ids)):
            raise ExperimentError(f"experiment mode {mode.mode} has duplicate episode IDs")
        _rate(mode.success_rate, f"mode {mode.mode} success_rate")
        _rate(mode.intervention_rate, f"mode {mode.mode} intervention_rate")
        if not isinstance(mode.failure_stage_rates, dict):
            raise ExperimentError(f"mode {mode.mode} failure_stage_rates must be an object")
        for stage, rate in mode.failure_stage_rates.items():
            _identifier(stage, "failure stage")
            _rate(rate, f"failure stage {stage}")
        if not isinstance(mode.coverage_counts, dict):
            raise ExperimentError(f"mode {mode.mode} coverage_counts must be an object")
        for axis, counts in mode.coverage_counts.items():
            _identifier(axis, "coverage axis")
            if not isinstance(counts, dict):
                raise ExperimentError(f"coverage counts for {axis} must be an object")
            for value, count in counts.items():
                _identifier(value, f"coverage {axis} value")
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ExperimentError(f"coverage count {axis}={value} is invalid")
        if not isinstance(mode.sufficient, bool):
            raise ExperimentError(f"mode {mode.mode} sufficient must be boolean")
        return mode.mode

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "spec_fingerprint": self.spec_fingerprint,
            "task_id": self.task_id,
            "task_fingerprint": self.task_fingerprint,
            "generated_at": self.generated_at,
            "modes": [mode.to_dict() for mode in self.modes],
            "comparison": self.comparison,
            "passed": self.passed,
            "failures": list(self.failures),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "report_sha256": self.report_sha256}


def evaluate_experiment(spec: ExperimentSpec, episodes: list[EpisodeManifest]) -> ExperimentReport:
    return ExperimentReport.create(spec, episodes)


def evaluate_experiment_store(spec: ExperimentSpec, root: str | Path) -> ExperimentReport:
    return evaluate_experiment(spec, load_episodes(root))


def load_report(path: str | Path) -> ExperimentReport:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"cannot load experiment report: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version", "experiment_id", "spec_fingerprint", "task_id",
        "task_fingerprint", "generated_at", "modes", "comparison", "passed",
        "failures", "report_sha256",
    }:
        raise ExperimentError("experiment report has invalid fields")
    try:
        if not isinstance(raw["modes"], list):
            raise ExperimentError("experiment report modes must be a list")
        modes = tuple(_mode_from_dict(value) for value in raw["modes"])
        failures = raw["failures"]
        if not isinstance(failures, list):
            raise ExperimentError("experiment report failures must be a list")
        report = ExperimentReport(
            schema_version=raw["schema_version"],
            experiment_id=raw["experiment_id"],
            spec_fingerprint=raw["spec_fingerprint"],
            task_id=raw["task_id"],
            task_fingerprint=raw["task_fingerprint"],
            generated_at=raw["generated_at"],
            modes=modes,
            comparison=raw["comparison"],
            passed=raw["passed"],
            failures=tuple(failures),
            report_sha256=raw["report_sha256"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentError(f"experiment report values are invalid: {exc}") from exc
    report.validate(include_hash=True)
    return report


def write_report(path: str | Path, report: ExperimentReport) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RGB/VLA execution modes")
    parser.add_argument("spec")
    parser.add_argument("episodes")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        report = evaluate_experiment_store(ExperimentSpec.load(args.spec), args.episodes)
        write_report(args.output, report)
    except (ExperimentError, OSError, ValueError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(2) from exc
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    raise SystemExit(0 if report.passed else 1)


def _comparison(reports: list[ExperimentModeReport]) -> dict[str, Any]:
    usable = [report for report in reports if report.episode_ids]
    if not usable:
        return {
            "best_mode": "",
            "best_success_rate": 0.0,
            "direct_vla_episode_count": 0,
            "success_rate_delta_vs_direct": {},
            "intervention_rate_delta_vs_direct": {},
        }
    direct = next((report for report in usable if report.mode == "direct_vla"), None)
    best = max(usable, key=lambda report: (report.success_rate, -report.intervention_rate))
    return {
        "best_mode": best.mode,
        "best_success_rate": best.success_rate,
        "direct_vla_episode_count": len(direct.episode_ids) if direct else 0,
        "success_rate_delta_vs_direct": {
            report.mode: report.success_rate - direct.success_rate
            for report in usable
        } if direct else {},
        "intervention_rate_delta_vs_direct": {
            report.mode: report.intervention_rate - direct.intervention_rate
            for report in usable
        } if direct else {},
    }


def _modes(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise ExperimentError("modes must be a non-empty list")
    values = tuple(_identifier(value, "mode") for value in raw)
    if len(values) != len(set(values)) or any(value not in EXECUTION_MODES for value in values):
        raise ExperimentError(f"modes must be unique values from {EXECUTION_MODES}")
    if "direct_vla" not in values:
        raise ExperimentError("modes must include direct_vla as the comparison baseline")
    return values


def _coverage(raw: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw, dict) or not raw or len(raw) > 16:
        raise ExperimentError("coverage must be a non-empty object")
    output: dict[str, tuple[str, ...]] = {}
    for axis, values in raw.items():
        key = _identifier(axis, "coverage axis")
        if not isinstance(values, list) or not values or len(values) > 64:
            raise ExperimentError(f"coverage {key} must be a non-empty list")
        parsed = tuple(_identifier(value, f"coverage {key}") for value in values)
        if len(parsed) != len(set(parsed)):
            raise ExperimentError(f"coverage {key} contains duplicates")
        output[key] = parsed
    return output


def _binding(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ExperimentError("binding_equals must be an object")
    output: dict[str, Any] = {}
    for key, value in raw.items():
        name = _identifier(key, "binding key")
        if isinstance(value, (dict, list, float)) or value is None:
            raise ExperimentError(f"binding value {name} must be a scalar")
        if not isinstance(value, (str, int, bool)):
            raise ExperimentError(f"binding value {name} is invalid")
        output[name] = value
    return output


def _positive_int(value: Any, label: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ExperimentError(f"{label} must be in [1, {maximum}]")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}", value):
        raise ExperimentError(f"{label} is not a safe identifier")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ExperimentError(f"{label} must be a SHA-256 digest")
    return value


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _mode_from_dict(raw: Any) -> ExperimentModeReport:
    if not isinstance(raw, dict) or set(raw) != _MODE_FIELDS:
        raise ExperimentError("experiment mode report has invalid fields")
    episode_ids = raw["episode_ids"]
    coverage_counts = raw["coverage_counts"]
    if (
        not isinstance(episode_ids, list)
        or isinstance(raw["episode_count"], bool)
        or not isinstance(raw["episode_count"], int)
        or raw["episode_count"] != len(episode_ids)
    ):
        raise ExperimentError("experiment mode report episode count does not match IDs")
    if not isinstance(coverage_counts, dict):
        raise ExperimentError("experiment mode report coverage counts must be an object")
    return ExperimentModeReport(
        mode=raw["mode"],
        episode_ids=tuple(episode_ids),
        success_rate=raw["success_rate"],
        intervention_rate=raw["intervention_rate"],
        failure_stage_rates=raw["failure_stage_rates"],
        coverage_counts=coverage_counts,
        sufficient=raw["sufficient"],
    )


def _validate_comparison(value: Any) -> None:
    expected = {
        "best_mode",
        "best_success_rate",
        "direct_vla_episode_count",
        "success_rate_delta_vs_direct",
        "intervention_rate_delta_vs_direct",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ExperimentError("experiment comparison has invalid fields")
    best_mode = value["best_mode"]
    if not isinstance(best_mode, str):
        raise ExperimentError("experiment comparison best_mode is invalid")
    if best_mode and best_mode not in EXECUTION_MODES:
        raise ExperimentError("experiment comparison best_mode is unknown")
    _rate(value["best_success_rate"], "best_success_rate")
    count = value["direct_vla_episode_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ExperimentError("direct_vla_episode_count is invalid")
    for key in ("success_rate_delta_vs_direct", "intervention_rate_delta_vs_direct"):
        deltas = value[key]
        if not isinstance(deltas, dict):
            raise ExperimentError(f"{key} must be an object")
        for mode, delta in deltas.items():
            if mode not in EXECUTION_MODES:
                raise ExperimentError(f"{key} contains unknown mode {mode!r}")
            if (
                not isinstance(delta, (int, float))
                or isinstance(delta, bool)
                or not math.isfinite(delta)
                or not -1 <= delta <= 1
            ):
                raise ExperimentError(f"{key} contains an invalid rate")


def _rate(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ExperimentError(f"{label} must be a finite rate in [0, 1]")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise ExperimentError(f"{label} must be a finite rate in [0, 1]")
    return parsed


__all__ = [
    "EXECUTION_MODES",
    "ExperimentError",
    "ExperimentModeReport",
    "ExperimentReport",
    "ExperimentSpec",
    "evaluate_experiment",
    "evaluate_experiment_store",
    "load_report",
    "main",
    "write_report",
]
