# Contributing

## Scope

This repository contains the standalone runtime (`robot-agent`), framework
documentation, and hardware guidance. The LeRobot training/inference
workspaces are external dependencies; changes that depend on local model
weights, datasets, or machine-specific paths are not accepted into the runtime.

## Development setup

```bash
cd robot-agent
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

Run the documentation link check from the repository root:

```bash
python scripts/check_markdown_links.py
```

## Safety invariants

Any change touching actuation must preserve these invariants:

1. Unknown tasks and motion targets are rejected before execution.
2. Real actuation requires a valid one-time token bound to the task fingerprint,
   executor, and deployment fingerprint.
3. `stop` and `estop` cancel the active process group and motion gateway.
4. A missing or stale observation fails closed when the task requires one.
5. New executors validate their config, bound their runtime, and fail closed on
   observation or actuation errors.
6. Mission `retry` cannot repeat a learned policy without a reviewed recovery
   action.

Add tests for every new refusal path, boundary, and state transition.

## Style

- Python 3.10+, standard library first; avoid new runtime dependencies.
- Keep `ruff` and `mypy` configurations in `robot-agent/pyproject.toml`.
- No machine-specific absolute paths (`/home/<user>`), private endpoints, or
  credentials in tracked files. Use environment variables and `.env.example`.
- Documentation must state what is implemented, what is simulated, and what is
  not supported.

## Pull requests

- One logical change per pull request.
- Include the verification command and its output.
- Do not commit datasets, checkpoints, TensorRT engines, videos, or generated
  build artifacts.
- Update the relevant document and `docs/known-limitations.md` when behavior or
  scope changes.

## Security and privacy

Follow [SECURITY.md](SECURITY.md). Report vulnerabilities privately. Do not
attach camera images containing people, tokens, or internal network details.
