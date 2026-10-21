# Embodied Home Robot Roadmap

## Verified Baseline

The current robot has a three-wheel omnidirectional base, two LeRobot arms, and three plain RGB
cameras (`head_cam`, `left_cam`, and `right_cam`). The repository must not assume depth images,
point clouds, LiDAR, force/torque sensing, tactile sensing, or a commissioned MoveIt model.
The confirmed capture mapping is documented in [hardware-profile.md](hardware-profile.md).

The first delivery target is a reliable standalone runtime for the four existing learned tasks:

- `fold_simple`
- `stack_contrast`
- `pick_up_bottle_towel`
- `pick_up_bottle`

Each task uses the same path: RGB observation, task matching, catalog validation, one-time
authorization, LeRobot policy execution, and a newer RGB postcondition observation.

## Near-Term Work

1. Finish the standalone runtime and remove all Claude Code/MCP runtime coupling.
2. Bind policy authorization to the exact three-camera RGB rig and model artifacts.
3. Validate the existing four tasks across lighting, placement, and object variations.
4. Improve RGB-only object detection, tracking, and task precondition evidence.
5. Add new learned tasks through catalog and policy entries without changing the daemon.
6. Improve data collection throughput with automatic episode slicing and failure triage.

## Reducing VLA Dependence

The project should reduce terminal VLA use only through capabilities supported by owned hardware.
Useful RGB-only candidates include open-vocabulary 2D detection, segmentation, multi-view tracking,
VLM task-state checks, scripted base motion, and policy selection. Geometry or contact primitives
must stay in research until their required sensing and calibration are explicitly available.

Simulation may be used for task definitions and software regression, but it is not part of the
current robot execution path and cannot justify new hardware assumptions.

## Admission Rule

A proposed capability must document its required sensors, actuators, calibration, runtime assets,
and fallback behavior before implementation. If any required hardware is unverified, the capability
remains a research item and must not appear in the product task catalog or runtime entry points.
