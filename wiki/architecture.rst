Architecture
============

Manipulation primitives
-----------------------

The primitive layer has two halves:

- config classes define task frames, processors, validation, and entry-time
  target resolution
- env classes own runtime state such as current target pose, primitive
  completion, and scripted trajectory progress

This split is deliberate. Runtime state stays in the env so config objects stay
serializable and easier to reason about.

Primitive entry
---------------

When a primitive transition fires, MP-Net stores a small boundary context:

- the processed observation from the primitive that just ended
- the previous task-frame origin
- source and target primitive names

The next primitive receives that context during ``reset()`` via its
``on_entry(...)`` hook.

If relative EE observations are enabled, the processed observation rewrites
``.ee_pos`` into the relative policy-facing view. Primitive-entry math and
pose-based transitions currently read those processed channels directly.

Transitions
-----------

Transitions are meant to stay small and declarative:

- threshold checks read directly from observation or info
- target-pose transitions read the target from ``info`` and the current pose
  from processed observation through the shared pose utility
- omitted ``OnTargetPoseReached.axes`` defaults to the source primitive's fixed
  task-space ``POS`` axes whose ``policy_mode`` is ``None``
- scripted completion transitions use ``OnSuccess(success_key="primitive_complete")``

Dynamic primitives
------------------

The current dynamic primitives are:

- ``ManipulationPrimitiveConfig``
  Static targets from config.
- ``MoveDeltaPrimitiveConfig``
  Resolve a target once on entry from a delta in either world or current-EE
  coordinates.
- ``OpenLoopTrajectoryPrimitiveConfig``
  Own a ``trajectory`` spec, resolve start/goal poses on entry, and sample the
  current scripted target through ``target_pose_at(...)`` while the env handles
  runtime stepping and progress bookkeeping.

Task-frame overrides
--------------------

``TaskFrame`` now carries optional ``controller_overrides``. Shared config and
serialization code round-trips the field generically, and robot wrappers
translate the supported keys into concrete controller commands. UR currently
consumes wrench/compliance overrides this way.

What to preserve when editing
-----------------------------

- keep entry context small
- do not move runtime target state back into configs
- prefer reusing the shared observation-pose utility instead of adding new
  redundant info keys
- keep package imports light to avoid circular-import churn
