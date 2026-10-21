# RGB Table-Edge Docking

`dock_table` closes the semantic gap between "go to the table" and a mapped
Nav2 dock. It needs no fiducial, dock marker, depth camera, or point cloud: the
head camera and the two wrist cameras estimate table-edge geometry, and the
holonomic base takes bounded steps until a settled confirmation observation is
inside tolerance.

## Control Loop

```text
observe (3-camera RGB)
      |
      +-- confidence < floor --> no motion; abort after two consecutive misses
      |
      +-- inside tolerance ----> settle -> observe again -> success only if still inside
      |
      +-- outside tolerance ---> bounded step (forward / lateral / yaw)
                                      |
                                      +-- repeat, max_steps + deadline + stall guard
```

Fail-closed rules:

- No actuator command is issued from an observation below `min_confidence`.
- Any observer or actuator exception ends the run without further motion.
- Three consecutive observations without measurable progress abort the run.
- The final success requires a second observation after `settle_seconds`.
- Cancellation and the mission deadline are checked before every observation and step.

## Sign Conventions

| Field | Meaning |
|---|---|
| `distance_m` | Forward distance from the base to the nearest table/cabinet edge |
| `lateral_offset_m` | Edge midpoint offset to the robot's right (positive right) |
| `yaw_error_deg` | Heading error; positive means rotate CCW (left) to square up |
| `forward_m` step | Positive moves toward the edge |
| `right_m` step | Positive moves right |
| `yaw_deg` step | Positive rotates CCW |

## Profiles

Profiles live in a versioned catalog (`config/tables.example.yaml`) and are
loaded from `ROBOT_AGENT_TABLE_CATALOG`. Every value is part of the deployment
fingerprint and the mission token binding; editing a profile after arming
invalidates the token.

```yaml
version: 1
tables:
  drawer_cabinet_edge:
    standoff_m: 0.42
    distance_tolerance_m: 0.04
    lateral_tolerance_m: 0.03
    yaw_tolerance_deg: 2.0
    min_confidence: 0.65
    max_step_m: 0.10
    max_step_yaw_deg: 5.0
    max_steps: 14
    deadline_seconds: 75.0
    settle_seconds: 0.6
```

## Usage

Direct task (catalog task `dock_table`, executor `table_docking`). A fresh
observation is required before arming when `ROBOT_AGENT_REQUIRE_OBSERVATION` is on:

```bash
robotctl observe "描述当前环境"
robotctl arm dock_table --confirm "CONFIRM dock_table"
robotctl execute dock_table --real
```

Mission action:

```yaml
- id: approach_cabinet
  type: action
  action: dock_table
  target: drawer_cabinet_edge
  timeout_seconds: 90
```

The executor requires the `robotd` one-time token. Without an authorization
context, or when the binding marks a dry-run, no chassis command is sent.

## Observability

Every run returns `observations`, `commands`, `trace`, and
`final_observation`. Mission nodes record a child episode with
`task_executor: table_docking` and the profile fingerprint.
