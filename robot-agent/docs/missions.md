# Bounded Mobile-Manipulation Missions

A mission composes reviewed named motion and learned-task contracts into a longer workflow. Unsupported manipulation executors are rejected by the task catalog.

## Catalog Contract

The built-in mission catalog is intentionally empty. Deployments set:

```bash
ROBOT_AGENT_MISSION_CATALOG=/etc/robot-agent/missions.yaml
```

Each mission declares a stable ID, aliases, observation requirement, total runtime, node-visit limit and one bounded graph. Supported nodes are:

| Node | Behavior |
|------|----------|
| `sequence` | Runs children in order and stops on failure. |
| `selector` | Runs reviewed fallback branches until one succeeds. |
| `retry` | Repeats navigation/docking descendants, at most three attempts. With a declared `recovery`, also repeats task descendants, running the recovery action before every retry. |
| `action` | Calls `navigate`, `dock`, `dock_table` or one registered `task` by ID. |

Mission YAML cannot contain coordinates, joint targets, shell commands, Python modules or plugin class names. `navigate` and `dock` targets must exist in the versioned motion catalog; `dock_table` targets must exist in the versioned table catalog. `task` targets must exist in the task catalog and still execute their own preconditions, backend, postconditions, watchdog and policy/benchmark gates.

`robotd` now cross-checks every mission action against the loaded task, motion and table catalogs at
startup. An unknown target rejects the deployment before the service listens, instead of
remaining latent until an operator attempts to arm the mission.

Task actions are forbidden below `retry` unless that retry declares a reviewed `recovery`
action. Each attempt still consumes exactly one policy attempt; the recovery action runs
between attempts and must succeed, otherwise the retry stops with a recovery failure. Recovery
actions are `observe` (fresh VLM evidence, no actuation) or a registered `task`.

```yaml
- id: place_object
  type: retry
  max_attempts: 2
  recovery:
    action: observe
    target: drawer_workspace_check
    timeout_seconds: 45
  child:
    id: operate_drawer
    type: action
    action: task
    target: operate_drawer
    timeout_seconds: 180
```

Semantic table docking is available as a mission action; see
[table-docking.md](table-docking.md).

```yaml
- id: approach_cabinet
  type: action
  action: dock_table
  target: drawer_cabinet_edge
  timeout_seconds: 90
```

## Authorization

At mission arm time, `robotd` resolves:

- the mission fingerprint;
- the complete deployment fingerprint;
- every named motion target and a fingerprint of its catalog version and calibrated target;
- every task fingerprint, executor, policy/model fingerprint and required benchmark report;
- every node timeout and task maximum runtime.

The complete binding enters one one-time token. Execution recalculates it before consuming the token. Changing a map target, task YAML, checkpoint, benchmark report or mission graph after arm makes the token invalid.

Coordinates are hashed into motion target fingerprints but are not returned by mission catalog, plan, dry-run or arm responses.

## Runtime Flow

```bash
robotctl mission-catalog
robotctl plan "整理客厅遥控器"
robotctl mission-dry-run tidy_remote_at_living_room
robotctl arm-mission tidy_remote_at_living_room \
  --confirm "CONFIRM mission:tidy_remote_at_living_room"
robotctl run-mission tidy_remote_at_living_room --real
```

Only one mission or hardware task runs at a time. `stop` and `estop` call both the manipulation supervisor and motion gateway. The mission executor checks daemon state before every node, so cancellation prevents later actions even if the current adapter returns after being canceled.

Every action produces a structured trace. Named motion and task nodes each write their normal episode; the mission writes an outer `mission_graph` episode referencing child episode IDs in its trace. This supports end-to-end and per-layer benchmarks without flattening all failure causes into one label.

## Why Not Free-Form LLM Plans

Natural language alias matching may select a registered mission, but it cannot create a real mission graph. Unknown goals remain dry-run planning output. This preserves the useful role of language models for intent and explanation while keeping the actuation graph versioned, reviewable and token-bound.

BehaviorTree.CPP or PlanSys2 can later replace or supervise this interpreter when lifecycle introspection or PDDL scale justifies the dependency. They must retain the same named-action, binding, cancellation and episode contracts.
