# Learned Policy Registry

Manipulation tasks and learned-policy deployments are separate concerns. `skill_flow.py` defines household task semantics and safety metadata; `/etc/robot-agent/policies.yaml` selects the exact executable and checkpoint allowed to implement each task.

## Promotion Contract

Every policy entry contains:

- stable `policy_id`, family and version;
- exactly one canonical `task_id`;
- direct executable path and SHA-256 from a dedicated environment, without `conda activate` or shell evaluation;
- runtime lockfile path and SHA-256 to pin the Python dependency set;
- workspace, checkpoint directory and primary model filename;
- SHA-256 of the model weights;
- fixed English task description, timeout and extra argv;
- explicit `enabled` flag.

The registry also owns robot type, serial ports and camera mapping. These values do not appear in language-model prompts or task aliases. The current hardware contract requires exactly three `opencv` RGB cameras named `head_cam`, `left_cam`, and `right_cam`. Device indices remain deployment values; depth topics, point-cloud fields, and additional camera roles are rejected.

The current operator-confirmed capture mapping is `left_cam=/dev/video4`,
`right_cam=/dev/video0`, and `head_cam=/dev/video2`; see [hardware-profile.md](hardware-profile.md).
Revalidate the mapping after USB topology or device-rule changes.

## Prepare A Policy

Install each policy environment as a self-contained venv or container image. The configured executable must be directly runnable:

```bash
/opt/lerobot-pzy/bin/lerobot-record --help
sha256sum /opt/lerobot-pzy/bin/lerobot-record \
  /opt/lerobot-pzy/requirements.lock \
  /opt/models/smolvla_fold_simple/v1/model.safetensors
```

Start from [policies.example.yaml](../config/policies.example.yaml), insert the measured digest, keep `enabled: false`, and run a schema check:

```bash
robot-policy-check /etc/robot-agent/policies.yaml --schema-only
```

The hashes do not need to be copied by hand. After replacing the executable, runtime manifest,
model, serial-port, and camera-device paths with reviewed local values, materialize a sealed registry:

```bash
robot-policy-materialize policies.template.yaml /etc/robot-agent/policies.yaml
```

The output defaults every policy to `enabled: false`. Use repeated `--enable TASK_ID` flags only for
tasks whose exact checkpoint and hardware configuration have been reviewed.

After offline replay and RGB observation checks pass, set only the approved policy to `enabled: true` and run the full check:

```bash
robot-policy-check /etc/robot-agent/policies.yaml
```

Artifact readiness is necessary but no longer sufficient for production arming. Generate a passing report with `robot-benchmark-evaluate`; when `ROBOT_AGENT_REQUIRE_POLICY_BENCHMARK=true`, `robotd` binds that report to the same one-time token. See [data-flywheel.md](data-flywheel.md).

The full check reads the model weights and fails on a digest mismatch, missing executable, missing workspace, disabled default or incorrect task mapping.

## Authorization Binding

When an operator arms a task, `robotd` resolves and verifies the default policy. The one-time token is bound to `policy_id` and a fingerprint covering the complete policy configuration and model digest. Execution resolves it again; changing the default, argv, model path, timeout or digest invalidates the token.

The worker receives the registry path directly from `robotd`, enters an internal authorization context, and launches a structured argv list with `shell=False` semantics. Direct `LeRobotClient` calls without that context are default-deny. The policy ID, policy fingerprint and required benchmark report identity are written to daemon status, results, episode and audit evidence.

## Remaining Hardware Gate

A correct hash proves artifact identity, not task quality or physical safety. Policies remain disabled until the exact checkpoint and three-camera RGB configuration pass task precondition, stop, postcondition, and failure-recovery validation.
