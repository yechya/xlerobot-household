# XLeRobot Base Bridge

`robot-base-bridge` converts the final safety-filtered ROS 2 velocity into the existing XLeRobot host JSON protocol and publishes wheel-derived odometry back to ROS.

```text
Nav2 controller -> cmd_vel_nav
                       |
              collision_monitor
                       |
        /xlerobot/base/cmd_vel_safe
                       |
              robot-base-bridge
                       |
              ZMQ loopback only
                       |
             XLeRobot hardware host
                       |
          500ms host watchdog -> motors
```

## Safety Behavior

- ROS linear `x/y` is limited as one holonomic velocity vector, not independently per axis.
- ROS yaw uses rad/s; the host boundary explicitly converts to and from deg/s.
- A command older than 250ms becomes zero while the bridge continues refreshing the host at 30Hz.
- Non-finite commands fail closed. Shutdown attempts three zero-velocity writes.
- Host observation gaps do not integrate an unbounded odometry jump and are reported as diagnostic errors.
- The existing 500ms host watchdog remains independent and calls `stop_base()` if bridge traffic disappears.

## Transport Boundary

The current upstream XLeRobot host binds unauthenticated ZMQ sockets. Until both sides support ZMQ CURVE or are replaced by authenticated local IPC, preflight permits only `127.0.0.1`, `::1` or `localhost`. Do not expose ports 5555/5556 to a LAN, container bridge or Wi-Fi interface.

Only one bridge/client may connect to the conflated host sockets. Stop keyboard teleoperation and legacy `XLerobotClient` processes before bringup. Operator interfaces still call `robotd`; they must never publish directly to `/xlerobot/base/cmd_vel_safe`.

## Odometry Limitation

The upstream host returns present wheel velocity, not encoder position. The bridge therefore integrates velocity into `odom -> base_link`; it is suitable for initial AMCL/HIL work but will drift. Before M1 acceptance, replace or validate it against encoder-position odometry and IMU fusion with `robot_localization`. Covariance values in the example are placeholders and require measured calibration.
