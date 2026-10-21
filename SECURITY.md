# Security Policy

## Reporting a vulnerability

Use GitHub's **private vulnerability reporting** (Security tab → Report a
vulnerability). Do not open a public issue for:

- authentication or one-time-token bypass;
- path traversal, arbitrary command execution, or audit-log tampering;
- unsafe actuation, emergency-stop bypass, or watchdog defeat;
- credential, camera-image, or dataset leakage;
- dependency vulnerabilities with a reachable exploit path.

Include the affected version/commit, a minimal reproduction, and the impact.
Please do not include real tokens, WiFi credentials, private IPs, or images of
people in the report.

## Supported versions

Security fixes target the `main` branch. Released tags receive fixes on a
best-effort basis; this project does not provide an SLA.

## Security model and non-goals

The runtime is not a certified functional-safety system. It provides
engineering safeguards: one-time authorization tokens, allowlisted tasks,
bounded missions, a software watchdog, an optional emergency-stop command, and
an append-only audit log. Physical emergency stop, motor limits, collision
detection, and safe torque-off must be implemented and validated independently
on the hardware controller.

Known boundaries:

- `robotd` is a task authorization and supervision boundary, not a hard
  real-time controller.
- The upstream XLeRobot host exposes unauthenticated ZMQ sockets; the bridge
  only permits loopback addresses until authenticated transport is available.
- VLM endpoints receive camera snapshots. Operators must review the data-flow,
  retention, and privacy policy of the configured provider. Prefer a local
  endpoint for images of people.
- Demo videos, datasets, and model weights are not covered by the software
  security policy.

## Dependency and secret hygiene

- Never commit `.env`, token caches, runtime state, private keys, WiFi
  credentials, or internal proxy addresses.
- Before publishing a fork, rewrite history that contains such values; deleting
  a file in a later commit does not remove it from earlier history.
- Report committed secrets immediately and rotate the credential.
