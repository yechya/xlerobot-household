# Verified Hardware Profile

This file records only hardware facts confirmed by the operator. New runtime features must not
assume additional sensors or interfaces without updating this profile from real hardware evidence.

## RGB Cameras

Operator-confirmed mapping on 2026-07-15:

| Camera role | Capture device | Sensor contract |
|---|---|---|
| `left_cam` | `/dev/video4` | Plain RGB |
| `right_cam` | `/dev/video0` | Plain RGB |
| `head_cam` | `/dev/video2` | Plain RGB |

The three nodes successfully produced `640x480` JPEG test frames. This confirms the capture mapping,
not the production frame rate, exposure, white balance, latency, or stable device enumeration after
reboot. Those values must be measured before enabling a policy.

## Unsupported Assumptions

The current product runtime must not require RGB-D, point clouds, force/torque sensing, tactile
sensing, or a commissioned MoveIt model. Simulation assets are research inputs, not robot hardware
evidence.
