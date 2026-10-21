## Summary

Describe the change and the user-visible behavior.

## Verification

- [ ] `cd robot-agent && pytest -q`
- [ ] `python scripts/check_markdown_links.py`
- [ ] Relevant integration patch still applies (`integrations/*/verify.sh`) when touched
- [ ] Smoke test for the changed path (command and observed result)

```text
paste test/smoke output here
```

## Safety

- [ ] Unknown tasks, motion targets, and unsafe arguments still fail closed.
- [ ] `stop` / `estop` still cancel the active process group and motion gateway.
- [ ] New actuation paths require a one-time token and bound deployment fingerprint.
- [ ] No new unbounded retry, loop, or runtime.

## Hygiene

- [ ] No secrets, tokens, private endpoints, absolute home paths, or internal IPs.
- [ ] No datasets, checkpoints, engines, videos, or generated build artifacts.
- [ ] Documentation and `docs/known-limitations.md` updated when scope changes.
