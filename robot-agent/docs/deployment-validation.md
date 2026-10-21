# Deployment Catalog Validation

`robotd` treats the task, mission, navigation, semantic-map and world-fusion files as one
deployment. Each file remains independently versioned, but startup validates their references
before the semantic map is written, the daemon is created or the Unix socket begins listening.

The preflight rejects:

- mission `task`, `navigate` or `dock` actions whose targets are absent from their catalogs;
- semantic-map, fusion and persistent navigation locations that disagree on planning frame;
- semantic navigation bindings on entities other than zones;
- zone bindings that name an unknown target or use unexpected fields;
- a navigation zone farther from its named pose than its reviewed `max_distance_m`.

A semantic navigation binding is an optional reserved property:

```yaml
properties:
  navigation_binding:
    action: navigate
    target: living_room_service
    max_distance_m: 0.25
```

Dock bindings use only `action: dock` and `target`. Other semantic properties remain
application-defined and JSON-safe. Persistent `navigate` targets must use the semantic map's
planning frame. The distance gate compares the reviewed zone XY position with the navigation
catalog pose; it does not replace Nav2 collision checking or docking HIL.

`robotctl capabilities` returns a `deployment` object with full, motion-domain and
manipulation-domain SHA-256 fingerprints, planning frame, semantic map ID and catalog counts.
The motion fingerprint binds navigation coordinates and the semantic map. The manipulation
fingerprint binds task, fusion and semantic assets. The full fingerprint also binds missions.

The same fingerprint enters task, mission and named-motion execution bindings. Named-motion
tokens also bind the fully resolved pose or dock target, not only its name. A calibration or
catalog change after arming therefore invalidates the old token, and completed episodes carry
the exact deployment fingerprint consumed by data triage.

Domain fingerprints prevent unnecessary HIL repetition. Adding or editing a mission changes
the full deployment fingerprint but leaves Nav2 and manipulation acceptance valid. Changing a
navigation pose invalidates the motion acceptance; changing a task, fusion threshold or
semantic manipulation asset invalidates manipulation acceptance.

This contract keeps growth data-driven. Adding a household workflow means adding or changing
catalog entries whose references are checked generically; it does not require a new command
branch in `robotd` or a new end-to-end VLA policy.
