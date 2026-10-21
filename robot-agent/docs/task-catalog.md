# Task Catalog

The task catalog is the single source of truth for user-facing task IDs, aliases, evidence checks,
timeouts, and execution type. The current hardware profile accepts only `learned_policy` tasks.

The default catalog contains the four original tasks. To add another task:

1. Add a task definition with `executor: learned_policy`. Keep `executor_config` empty
   unless the installed executor explicitly declares and validates a configuration schema.
2. Add RGB-verifiable preconditions and postconditions.
3. Add a policy registry entry with the exact executable, runtime, model hashes, and three-camera rig.
4. Validate the task in dry-run before enabling the policy.

The catalog stores executor IDs without importing implementations. An executor that is not
registered by the runtime is rejected when a task is executed; the current production service
still binds real manipulation only to `learned_policy`, so adding a catalog entry does not enable
unverified hardware by itself.

`executor_config` is part of the task fingerprint and is validated by the registered executor at
runtime. An executor that does not expose `validate_config` is default-deny for non-empty config;
the current LeRobot adapter therefore continues to require `{}`. This allows a future, reviewed
RGB-only executor to add bounded configuration without changing the task execution pipeline.
