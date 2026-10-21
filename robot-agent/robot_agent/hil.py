from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass(frozen=True)
class HilPolicy:
    max_stop_latency_seconds: float = 0.5
    stop_hold_seconds: float = 0.2
    max_sample_gap_seconds: float = 0.15
    linear_stop_threshold: float = 0.02
    angular_stop_threshold: float = 0.05
    max_action_terminal_seconds: float = 2.0
    allowed_action_statuses: tuple[str, ...] = ("canceled", "aborted", "rejected")
    required_scenarios: tuple[str, ...] = ()

    @classmethod
    def load(cls, path: str | Path | None) -> "HilPolicy":
        if path is None:
            return cls()
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("HIL policy must be a YAML object")
        allowed = raw.get("allowed_action_statuses", cls.allowed_action_statuses)
        required = raw.get("required_scenarios", ())
        if not isinstance(allowed, list) or not all(isinstance(item, str) for item in allowed):
            raise ValueError("allowed_action_statuses must be a string list")
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise ValueError("required_scenarios must be a string list")
        return cls(
            max_stop_latency_seconds=_positive(raw, "max_stop_latency_seconds", 0.5),
            stop_hold_seconds=_positive(raw, "stop_hold_seconds", 0.2),
            max_sample_gap_seconds=_positive(raw, "max_sample_gap_seconds", 0.15),
            linear_stop_threshold=_positive(raw, "linear_stop_threshold", 0.02),
            angular_stop_threshold=_positive(raw, "angular_stop_threshold", 0.05),
            max_action_terminal_seconds=_positive(
                raw,
                "max_action_terminal_seconds",
                2.0,
            ),
            allowed_action_statuses=tuple(allowed),
            required_scenarios=tuple(required),
        )


@dataclass(frozen=True)
class HilEvent:
    scenario_id: str
    scenario: str
    kind: str
    timestamp: float
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Any, line_number: int = 0) -> "HilEvent":
        location = f" on line {line_number}" if line_number else ""
        if not isinstance(raw, dict):
            raise ValueError(f"HIL event must be an object{location}")
        required = ("scenario_id", "scenario", "kind", "timestamp")
        if any(key not in raw for key in required):
            raise ValueError(f"HIL event is missing required fields{location}")
        scenario_id = raw["scenario_id"]
        scenario = raw["scenario"]
        kind = raw["kind"]
        timestamp = raw["timestamp"]
        if not all(isinstance(value, str) and value for value in (scenario_id, scenario, kind)):
            raise ValueError(f"HIL event identifiers must be non-empty strings{location}")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise ValueError(f"HIL event timestamp must be numeric{location}")
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError(f"HIL event timestamp must be finite{location}")
        data = raw.get("data", {})
        if not isinstance(data, dict):
            raise ValueError(f"HIL event data must be an object{location}")
        return cls(scenario_id, scenario, kind, timestamp, data)


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    scenario: str
    passed: bool
    stop_latency_seconds: float | None
    action_terminal_seconds: float | None
    failures: tuple[str, ...]


@dataclass(frozen=True)
class HilReport:
    passed: bool
    policy: HilPolicy
    scenarios: tuple[ScenarioResult, ...]
    missing_scenarios: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "policy": asdict(self.policy),
            "scenarios": [asdict(result) for result in self.scenarios],
            "missing_scenarios": list(self.missing_scenarios),
        }


def load_events(path: str | Path) -> list[HilEvent]:
    events: list[HilEvent] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                events.append(HilEvent.from_dict(json.loads(line), line_number))
    return events


def evaluate_events(events: Iterable[HilEvent], policy: HilPolicy | None = None) -> HilReport:
    active_policy = policy or HilPolicy()
    grouped: dict[str, list[HilEvent]] = {}
    for event in events:
        grouped.setdefault(event.scenario_id, []).append(event)
    results = tuple(
        _evaluate_scenario(sorted(group, key=lambda event: event.timestamp), active_policy)
        for _, group in sorted(grouped.items())
    )
    observed = {result.scenario for result in results}
    missing = tuple(sorted(set(active_policy.required_scenarios) - observed))
    return HilReport(
        passed=bool(results) and all(result.passed for result in results) and not missing,
        policy=active_policy,
        scenarios=results,
        missing_scenarios=missing,
    )


def _evaluate_scenario(events: list[HilEvent], policy: HilPolicy) -> ScenarioResult:
    scenario_id = events[0].scenario_id
    scenario = events[0].scenario
    failures: list[str] = []
    if any(event.scenario != scenario for event in events):
        failures.append("scenario_id contains inconsistent scenario names")
    faults = [event for event in events if event.kind == "fault"]
    if len(faults) != 1:
        failures.append("scenario must contain exactly one fault event")
        fault_at = None
    else:
        fault_at = faults[0].timestamp

    if any(event.kind == "collision" and bool(event.data.get("detected")) for event in events):
        failures.append("collision was detected")

    stop_latency = None
    action_terminal_seconds = None
    if fault_at is not None:
        samples = [
            event
            for event in events
            if event.kind == "velocity" and event.timestamp >= fault_at
        ]
        stopped_at = _stable_stop_time(samples, policy)
        if stopped_at is None:
            failures.append("stable zero velocity was not observed after fault")
        else:
            stop_latency = stopped_at - fault_at
            if stop_latency > policy.max_stop_latency_seconds:
                failures.append(
                    f"stop latency {stop_latency:.3f}s exceeds "
                    f"{policy.max_stop_latency_seconds:.3f}s"
                )

        terminals = [
            event
            for event in events
            if event.kind == "action_terminal" and event.timestamp >= fault_at
        ]
        if not terminals:
            failures.append("Nav2 action terminal state was not observed after fault")
        else:
            terminal = terminals[0]
            action_terminal_seconds = terminal.timestamp - fault_at
            status = str(terminal.data.get("status", ""))
            if status not in policy.allowed_action_statuses:
                failures.append(f"unexpected Nav2 terminal status: {status or 'missing'}")
            if action_terminal_seconds > policy.max_action_terminal_seconds:
                failures.append(
                    f"action terminal latency {action_terminal_seconds:.3f}s exceeds "
                    f"{policy.max_action_terminal_seconds:.3f}s"
                )
    return ScenarioResult(
        scenario_id=scenario_id,
        scenario=scenario,
        passed=not failures,
        stop_latency_seconds=stop_latency,
        action_terminal_seconds=action_terminal_seconds,
        failures=tuple(failures),
    )


def _stable_stop_time(samples: list[HilEvent], policy: HilPolicy) -> float | None:
    epsilon = 1e-9
    for index, sample in enumerate(samples):
        if not _is_stopped(sample, policy):
            continue
        deadline = sample.timestamp + policy.stop_hold_seconds
        window = [
            event for event in samples[index:] if event.timestamp <= deadline + epsilon
        ]
        if not window or window[-1].timestamp + epsilon < deadline:
            continue
        gaps = [right.timestamp - left.timestamp for left, right in zip(window, window[1:])]
        if gaps and max(gaps) > policy.max_sample_gap_seconds:
            continue
        if all(_is_stopped(event, policy) for event in window):
            return sample.timestamp
    return None


def _is_stopped(event: HilEvent, policy: HilPolicy) -> bool:
    linear = event.data.get("linear_speed")
    angular = event.data.get("angular_speed")
    if isinstance(linear, bool) or not isinstance(linear, (int, float)):
        return False
    if isinstance(angular, bool) or not isinstance(angular, (int, float)):
        return False
    return abs(float(linear)) <= policy.linear_stop_threshold and abs(
        float(angular)
    ) <= policy.angular_stop_threshold


def _positive(raw: dict[str, Any], key: str, default: float) -> float:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{key} must be a positive number")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recorded ROS 2 HIL safety events")
    parser.add_argument("events", help="JSONL file produced by robot-hil-record")
    parser.add_argument("--policy", help="HIL acceptance policy YAML")
    parser.add_argument("--output", help="write JSON report to this path")
    args = parser.parse_args()
    try:
        report = evaluate_events(load_events(args.events), HilPolicy.load(args.policy))
        payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).write_text(payload + "\n", encoding="utf-8")
        print(payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(2) from exc
    raise SystemExit(0 if report.passed else 1)


if __name__ == "__main__":
    main()


__all__ = [
    "HilEvent",
    "HilPolicy",
    "HilReport",
    "ScenarioResult",
    "evaluate_events",
    "load_events",
    "main",
]
