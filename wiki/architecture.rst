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


Data-driven connector configs
-----------------------------

The Hoermann connector envs are split along one line: what belongs to the *rig*
stays in Python, what belongs to a *connector* is data in that connector's
``object_dir``.

``ur3e_hoermann_connector`` (``experiments/envs/hoermann/connector/``) is the
only env registered for insertion work. Robot IP, camera serials, control rates
and the controller gain profile are dataclass fields on it. Everything that
varies per connector -- the taught poses, the LED success detector, the camera
crops, the gripper opening, the insertion limits, the checkpoints -- loads from
files next to the mesh. Adding a connector therefore adds no Python and no
registry entry; see :doc:`connector_setup`.

Connector-level values are deliberately *not* dataclass fields. Because the
config assigns ``robot``/``teleop``/``cameras`` unconditionally in
``__post_init__``, draccus would silently discard a ``--env.robot.ip=...``
override; keeping LED and crop settings out of the field set means draccus
rejects ``--env.led_pixel_x=...`` outright instead. The value a run used is then
always recorded in ``connector.json`` beside the connector, not in a shell
history.

The teach env (``ur3e_hoermann_connector_teach``) exists because of an ordering
constraint: the insertion env reads ``poses.json`` while building its graph, so
it cannot be the env used to produce ``poses.json``. The teach env reads
``object_spec.json`` plus the shared rig poses and mirrors the production pickup graph:
teach the hanging grasp, pull and re-scan, teach the insertion grasp, then stretch and
approach before teaching the seated pose. Unknown targets are replaced with compliant
SpaceMouse states and store primitives; known rig motions remain absolute scripted moves.
This avoids seeding dangerous placeholder targets merely to construct the normal env.

``experiments/envs/hoermann/e2e/*`` are the frozen per-connector configs this
replaced. They are kept as-is so existing runs stay reproducible;
``tests/experiments/envs/test_connector_config.py`` asserts the generic config
builds a graph identical to ``RightTTL_090726`` given the same data.
