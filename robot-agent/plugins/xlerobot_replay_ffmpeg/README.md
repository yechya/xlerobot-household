# XLeRobot Terminal Failure Video Clip

This independent replay handler exports a bounded video window immediately before the
episode terminal timestamp. The window is derived from the immutable episode and evidence
time ranges, not from an inferred stage timestamp. A reviewed FFmpeg executable is bound by
SHA in the replay runtime config and invoked without a shell.

The handler is intended to reduce review and learned-fallback dataset volume. It does not
label a failure automatically and does not make the resulting clip training-ready without
human review and task-specific annotation.
