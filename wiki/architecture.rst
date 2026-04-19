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

Transitions
-----------

Transitions are meant to stay small and declarative:

- threshold checks read directly from observation or info
- target-pose transitions read the target from ``info`` and the current pose
  from processed observation through the shared pose utility
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
  Resolve an entry target, then hand off execution to a scripted env subclass.

What to preserve when editing
-----------------------------

- keep entry context small
- do not move runtime target state back into configs
- prefer reusing the shared observation-pose utility instead of adding new
  redundant info keys
- keep package imports light to avoid circular-import churn

Task-frame rotation semantics
-----------------------------

The UR task-frame controller intentionally mixes two rotation semantics:

- absolute rotational ``POS`` targets are exposed as wrapped XYZ Euler angles at
  the API, because that is the only representation here where "lock roll, set
  yaw, leave pitch learnable" is directly meaningful per axis
- relative rotational ``POS`` targets are treated as a masked angular velocity
  in the task-frame basis and integrated on ``SO(3)``

The update rule is therefore:

.. math::

   R_{k+\!1/2} = \exp(\widehat{\omega}_{\mathrm{mask}} \, \Delta t)\, R_k

followed, when absolute rotational axes are present, by converting
``R_{k+1/2}`` to wrapped XYZ Euler angles, overwriting the absolute slots, and
converting back to the controller's internal rotation-vector state.

This is deliberate. Euler angles are a chart, not a linear space. A mixed
policy that "integrates the relative slots in Euler" becomes unintuitive away
from zero because the order of XYZ Euler rotations means an absolute setting on
one axis can change the meaning of the remaining slots.

The practical release-ready pattern is:

- put the large fixed orientation bias into the task-frame ``origin``
- keep mixed absolute rotational targets near zero whenever possible

Near zero, the Euler chart is locally well behaved and the masked ``SO(3)``
delta update is much easier to reason about. This keeps mixed absolute/relative
orientation commands predictable without pretending that Euler slots and
rotation-vector integration are the same space.
