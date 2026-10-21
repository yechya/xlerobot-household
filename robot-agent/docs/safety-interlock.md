# Household Safety Interlock

The safety interlock is a structured, fail-closed boundary check for people, pets, privacy state and safety-sensor health. It supplements but never replaces collision monitoring, torque/current limits, hardware watchdogs or the physical emergency stop.

## Snapshot Contract

A reviewed perception/safety adapter atomically updates `ROBOT_AGENT_SAFETY_SNAPSHOT` with:

- a strictly increasing sequence and wall-clock observation time;
- a versioned source identity;
- sensor health (`ok`, `degraded` or `lost`);
- whether a physical emergency stop is available and pressed;
- person/pet counts and nearest distances;
- person/pet occupancy of the manipulation workspace;
- privacy-zone activity;
- a canonical SHA-256 of all fields.

The file must be mode `0600`. `robotd` rejects missing, malformed, stale, future, hash-mismatched, source-unapproved or group-readable snapshots. Within one daemon lifetime it also rejects sequence rollback and different content under the same sequence.

`write_safety_snapshot()` is the atomic writer library for the trusted adapter. There is intentionally no operator CLI that can publish a synthetic “safe” state.

## Policy

`ROBOT_AGENT_SAFETY_POLICY` defines maximum age, allowed source prefixes, physical-estop requirement, privacy behavior and separate person/pet distances for motion and manipulation. Manipulation can additionally require an empty workspace.

Production `RuntimeConfig.from_env()` defaults `ROBOT_AGENT_REQUIRE_SAFETY_INTERLOCK=true`. Missing policy or snapshot leaves `robotd` healthy enough for status/stop/estop and dry-run, but all real arm requests refuse with `safety_interlock_blocked`.

## Check Points

The interlock runs:

- before arming and before consuming a token for a single manipulation task;
- before arming and executing named navigation/docking;
- for every operation class before mission arm and execute;
- immediately before every mission child action.

An unsafe execute check does not consume the one-time token. If the environment becomes safe under a newer snapshot, the operator can retry before token expiry. Successful decisions contribute snapshot sequence/hash evidence to command results and episodes.

`stop`, `estop`, status and reset are never gated.

## Continuous Safety Boundary

File checks are discrete. A person or pet can enter while Nav2 or a manipulator action is already running. Continuous protection must remain below this layer:

- Nav2 collision monitor and safety-rated proximity inputs for the mobile base;
- controller collision/effort limits and a manipulation heartbeat lease;
- MCU/host watchdogs that stop on command loss;
- a directly wired physical emergency stop;
- a supervised safety process that can assert the hardware stop independently of `robotd`.

Before an unsupervised home pilot, validate detector false negatives across lighting/occlusion, pet motion, privacy zones and sensor loss, then run the same 100-cycle fault-injection gates used for base stop behavior.
