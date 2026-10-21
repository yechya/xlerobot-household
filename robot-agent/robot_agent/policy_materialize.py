from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .policies import PolicyRegistry, PolicyRegistryError, sha256_file


def materialize_registry(
    template_path: str | Path,
    output_path: str | Path,
    *,
    enabled_tasks: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    template = Path(template_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    raw = yaml.safe_load(template.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise PolicyRegistryError("policy template must be a YAML object")
    policies = raw.get("policies")
    if not isinstance(policies, dict) or not policies:
        raise PolicyRegistryError("policy template policies must be a non-empty object")

    known_tasks: set[str] = set()
    digest_cache: dict[Path, str] = {}
    materialized: dict[str, dict[str, str]] = {}
    for policy_id, value in policies.items():
        if not isinstance(policy_id, str) or not isinstance(value, dict):
            raise PolicyRegistryError("policy template entries require an ID and object")
        task_id = value.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise PolicyRegistryError(f"policy {policy_id!r} requires task_id")
        known_tasks.add(task_id)
        executable = _resolve_existing_file(value, "executable", template.parent, policy_id)
        runtime = _resolve_existing_file(value, "runtime_manifest", template.parent, policy_id)
        model_path = _resolve_existing_directory(value, "model_path", template.parent, policy_id)
        model_file = value.get("model_file")
        if not isinstance(model_file, str) or not model_file.strip():
            raise PolicyRegistryError(f"policy {policy_id!r} requires model_file")
        model = (model_path / model_file).resolve()
        if not model.is_file():
            raise PolicyRegistryError(f"policy model artifact does not exist: {model}")

        value["executable"] = str(executable)
        value["runtime_manifest"] = str(runtime)
        value["model_path"] = str(model_path)
        value["executable_sha256"] = _digest(executable, digest_cache)
        value["runtime_sha256"] = _digest(runtime, digest_cache)
        value["model_sha256"] = _digest(model, digest_cache)
        value["enabled"] = task_id in enabled_tasks
        materialized[policy_id] = {
            "task_id": task_id,
            "model_path": str(model_path),
            "model_sha256": value["model_sha256"],
        }

    unknown = sorted(enabled_tasks - known_tasks)
    if unknown:
        raise PolicyRegistryError("cannot enable unknown tasks: " + ", ".join(unknown))

    output.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(output, yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
    try:
        registry = PolicyRegistry.load(output)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return {
        "output": str(output),
        "registry_version": registry.version,
        "enabled_tasks": sorted(enabled_tasks),
        "policies": materialized,
    }


def _resolve_existing_file(
    raw: dict[str, Any],
    key: str,
    base: Path,
    policy_id: str,
) -> Path:
    path = _resolve_path(raw, key, base, policy_id)
    if not path.is_file():
        raise PolicyRegistryError(f"policy {key} does not exist: {path}")
    return path


def _resolve_existing_directory(
    raw: dict[str, Any],
    key: str,
    base: Path,
    policy_id: str,
) -> Path:
    path = _resolve_path(raw, key, base, policy_id)
    if not path.is_dir():
        raise PolicyRegistryError(f"policy {key} does not exist: {path}")
    return path


def _resolve_path(raw: dict[str, Any], key: str, base: Path, policy_id: str) -> Path:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PolicyRegistryError(f"policy {policy_id!r} requires {key}")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _digest(path: Path, cache: dict[Path, str]) -> str:
    digest = cache.get(path)
    if digest is None:
        digest = sha256_file(path)
        cache[path] = digest
    return digest


def _write_atomic(path: Path, content: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seal a learned-policy template with hashes from explicit local paths",
    )
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--enable",
        action="append",
        default=[],
        metavar="TASK_ID",
        help="enable one reviewed task; may be repeated",
    )
    args = parser.parse_args(argv)
    try:
        result = materialize_registry(
            args.template,
            args.output,
            enabled_tasks=frozenset(args.enable),
        )
    except (OSError, PolicyRegistryError, yaml.YAMLError) as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"success": True, **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "materialize_registry"]
