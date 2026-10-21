from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .client import RobotClient, RobotClientError
from .config import RuntimeConfig


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        path.chmod(0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _load_cached_token(path: Path, task_id: str) -> str | None:
    try:
        if path.stat().st_mode & 0o077:
            raise ValueError(f"token file must not be accessible by group or others: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        raise ValueError(f"token file is not valid JSON: {path}") from exc
    if payload.get("task_id") != task_id:
        return None
    token = payload.get("token")
    return token if isinstance(token, str) and token else None


def _load_observation(path: str) -> dict[str, Any]:
    source = Path(path).expanduser()
    try:
        info = source.lstat()
    except OSError as exc:
        raise ValueError(f"observation file cannot be read: {source}") from exc
    if source.is_symlink() or not source.is_file() or info.st_size > 512 * 1024:
        raise ValueError("observation file must be a bounded regular file")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("observation file is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("observation file must contain a JSON object")
    return raw


def _cache_arm_token(config: RuntimeConfig, response: dict[str, Any]) -> None:
    data = response.get("data")
    if not response.get("success") or not isinstance(data, dict):
        return
    token = data.pop("token_secret", None)
    task_id = data.get("task_id")
    if not isinstance(token, str) or not isinstance(task_id, str):
        return
    _write_private_json(config.token_cache_path, {"task_id": task_id, "token": token})
    data["token_cached"] = True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone robotd control client")
    parser.add_argument("--socket", help="robotd Unix socket path")
    parser.add_argument("--timeout", type=float, default=30.0, help="request timeout in seconds")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in (
        "status",
        "health",
        "capabilities",
        "motion-catalog",
        "mission-catalog",
        "world-state",
        "pause",
        "stop",
        "estop",
    ):
        subparsers.add_parser(command)

    world_events = subparsers.add_parser("world-events")
    world_events.add_argument("--since", type=int, default=0, dest="since_sequence")
    world_events.add_argument("--limit", type=int, default=100)

    world_fuse = subparsers.add_parser("world-fuse")
    world_fuse.add_argument("observation")

    reset = subparsers.add_parser("reset")
    reset.add_argument("--confirm", default="", dest="operator_confirmation")

    plan = subparsers.add_parser("plan")
    plan.add_argument("goal")

    visual_plan = subparsers.add_parser("inspect-plan")
    visual_plan.add_argument("goal")

    arm_plan = subparsers.add_parser("arm-plan")
    arm_plan.add_argument("plan_id")
    arm_plan.add_argument("--confirm", required=True, dest="operator_confirmation")

    execute_plan = subparsers.add_parser("execute-plan")
    execute_plan.add_argument("plan_id")
    execute_plan.add_argument("--real", action="store_true")
    execute_plan.add_argument("--token-file", help="protected JSON token cache for automation")

    dry_run = subparsers.add_parser("dry-run")
    dry_run.add_argument("goal")

    observe = subparsers.add_parser("observe")
    observe.add_argument("instruction", nargs="?", default="描述当前环境")

    mission_dry_run = subparsers.add_parser("mission-dry-run")
    mission_dry_run.add_argument("mission_id")

    arm_mission = subparsers.add_parser("arm-mission")
    arm_mission.add_argument("mission_id")
    arm_mission.add_argument("--confirm", required=True, dest="operator_confirmation")

    run_mission = subparsers.add_parser("run-mission")
    run_mission.add_argument("mission_id")
    run_mission.add_argument("--real", action="store_true")
    run_mission.add_argument("--token-file", help="protected JSON token cache for automation")

    arm = subparsers.add_parser("arm")
    arm.add_argument("task_id")
    arm.add_argument("--confirm", required=True, dest="operator_confirmation")

    arm_motion = subparsers.add_parser("arm-motion")
    arm_motion.add_argument("operation", choices=("navigate", "dock"))
    arm_motion.add_argument("target")
    arm_motion.add_argument("--confirm", required=True, dest="operator_confirmation")

    execute = subparsers.add_parser("execute")
    execute.add_argument("task_id")
    execute.add_argument("--real", action="store_true")
    execute.add_argument("--token-file", help="protected JSON token cache for automation")

    for command in ("navigate", "dock"):
        motion = subparsers.add_parser(command)
        motion.add_argument("target")
        motion.add_argument("--real", action="store_true")
        motion.add_argument("--token-file", help="protected JSON token cache for automation")
    return parser


def _request_for_args(
    args: argparse.Namespace,
    config: RuntimeConfig,
) -> tuple[str, dict[str, Any]]:
    if args.command in {"status", "health", "capabilities", "pause", "stop", "estop"}:
        return args.command, {}
    if args.command == "motion-catalog":
        return "motion_catalog", {}
    if args.command == "mission-catalog":
        return "mission_catalog", {}
    if args.command == "world-state":
        return "world_state", {}
    if args.command == "world-events":
        return "world_events", {
            "since_sequence": args.since_sequence,
            "limit": args.limit,
        }
    if args.command == "world-fuse":
        return "world_fuse", {"observation": _load_observation(args.observation)}
    if args.command == "reset":
        return "reset", {"operator_confirmation": args.operator_confirmation}
    if args.command == "plan":
        return "plan", {"goal": args.goal}
    if args.command == "inspect-plan":
        return "visual_plan_inspect", {"goal": args.goal}
    if args.command == "arm-plan":
        return "visual_plan_arm", {
            "plan_id": args.plan_id,
            "operator_confirmation": args.operator_confirmation,
        }
    if args.command == "execute-plan" and not args.real:
        raise ValueError("execute-plan requires --real after arm-plan")
    if args.command == "execute-plan":
        task_id = f"plan:{args.plan_id}"
        token_path = (
            Path(args.token_file).expanduser() if args.token_file else config.token_cache_path
        )
        token = _load_cached_token(token_path, task_id)
        return "visual_plan_execute", {"plan_id": args.plan_id, "token": token}
    if args.command == "dry-run":
        return "dry_run", {"goal": args.goal}
    if args.command == "observe":
        return "observe", {"instruction": args.instruction}
    if args.command == "mission-dry-run":
        return "mission_dry_run", {"mission_id": args.mission_id}
    if args.command == "arm-mission":
        return "mission_arm", {
            "mission_id": args.mission_id,
            "operator_confirmation": args.operator_confirmation,
        }
    if args.command == "run-mission" and not args.real:
        return "mission_dry_run", {"mission_id": args.mission_id}
    if args.command == "run-mission":
        task_id = f"mission:{args.mission_id}"
        token_path = (
            Path(args.token_file).expanduser() if args.token_file else config.token_cache_path
        )
        token = _load_cached_token(token_path, task_id)
        return "mission_execute", {"mission_id": args.mission_id, "token": token}
    if args.command == "arm":
        return "arm", {
            "task_id": args.task_id,
            "operator_confirmation": args.operator_confirmation,
        }
    if args.command == "arm-motion":
        return "motion_arm", {
            "operation": args.operation,
            "target": args.target,
            "operator_confirmation": args.operator_confirmation,
        }
    if args.command == "execute" and not args.real:
        return "dry_run", {"goal": args.task_id}
    if args.command == "execute":
        token_path = (
            Path(args.token_file).expanduser() if args.token_file else config.token_cache_path
        )
        token = _load_cached_token(token_path, args.task_id)
        return "execute", {"task_id": args.task_id, "token": token}
    if args.command in {"navigate", "dock"} and not args.real:
        return "motion_dry_run", {"operation": args.command, "target": args.target}
    if args.command in {"navigate", "dock"}:
        task_id = f"motion:{args.command}:{args.target}"
        token_path = (
            Path(args.token_file).expanduser() if args.token_file else config.token_cache_path
        )
        token = _load_cached_token(token_path, task_id)
        return "motion_execute", {
            "operation": args.command,
            "target": args.target,
            "token": token,
        }
    raise ValueError(f"unsupported command: {args.command}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = RuntimeConfig.from_env()
    socket_path = Path(args.socket).expanduser() if args.socket else config.socket_path
    try:
        command, params = _request_for_args(args, config)
    except (OSError, ValueError) as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(2) from exc
    timeout = args.timeout
    if command in {"execute", "motion_execute", "mission_execute", "visual_plan_execute"}:
        timeout = max(timeout, 600.0)

    try:
        response = RobotClient(socket_path, timeout=timeout).call(command, **params)
    except RobotClientError as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(3) from exc

    if args.command in {"arm", "arm-motion", "arm-mission", "arm-plan"}:
        _cache_arm_token(config, response)
    elif command in {"execute", "motion_execute", "mission_execute", "visual_plan_execute"}:
        token_path = (
            Path(args.token_file).expanduser() if args.token_file else config.token_cache_path
        )
        token_path.unlink(missing_ok=True)

    print(json.dumps(response, ensure_ascii=False, indent=2))
    raise SystemExit(0 if response.get("success") else 1)


if __name__ == "__main__":
    main()


__all__ = ["main"]
