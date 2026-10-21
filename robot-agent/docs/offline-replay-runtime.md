# Offline Replay Runtime

Offline replay is isolated from `robotd` and cannot actuate hardware. A replay plan selects immutable
episodes, validates evidence hashes, and invokes reviewed handlers through the
`xlerobot.replay_handlers` entry-point group.

The core package provides evidence export. The optional FFmpeg plugin extracts bounded RGB video
clips for review and data triage. Handler outputs remain in a dedicated output directory and are
hashed before another handler can consume them.

RGB-D evaluation, raw depth/TF orchestration, and rosbag geometry capture handlers have been removed
because the current robot owns only plain RGB cameras.
