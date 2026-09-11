Experiment config conventions
=============================

Experiment configs should make the user's mental model visible. A config says
what the run should do; the primitive environment and shared processors contain
the machinery required to do it.

User-facing values
------------------

Values that a user may change between runs belong on the environment config:

.. code-block:: python

   @dataclass
   class ExampleConfig(ManipulationPrimitiveNetConfig):
       skip_grasp: bool = False
       pushdown: bool = False
       calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

Module-level constants are appropriate for fixed implementation vocabulary,
policy shapes, runtime keys, and values that are not meaningful run settings.
Keep them grouped near the top of the module so the public config surface is
easy to distinguish.

Primitive and config boundaries
-------------------------------

Start with a plain ``ManipulationPrimitiveConfig``. Set ``env_class`` and pass
primitive-specific constructor values through ``env_kwargs``. Add a config
subclass only when it needs custom construction or entry-time target
resolution. Do not create a wrapper config that only validates a task-frame
shape or copies one field into ``env_kwargs``.

The primitive environment owns runtime state: measured entry poses, captured
relative geometry, live targets, calibration values, and coordinated hardware
commands. The config owns the serializable description of the primitive.

Task frames and cooperative behavior
-------------------------------------

If per-robot targets are always used together, keep them together in one
explicit mapping:

.. code-block:: python

   task_frame={
       "left": TaskFrame(target=left_target, policy_mode=left_mode),
       "right": TaskFrame(target=right_target, policy_mode=right_mode),
   }

An explicit ``driver`` may be passed to the runtime environment when one robot
provides the cooperative action. The user should not need to know which arm is
the implementation-side driver; names and options should describe the task.
Use one cooperative runtime primitive for the coordination and projection
logic. Do not create a second subclass merely to change the observation schema.

Observation features
--------------------

Robot modality and axis selection belong in ``ObservationConfig``. Derived
signals belong in shared, opt-in observation processors. Such a processor must
handle all of the following together:

* reset-time values, including the first observation of an episode;
* static feature-shape inference before the first runtime step;
* serialization if it has configurable state;
* ordering and normalization assumptions documented by focused tests.

For example, entry-relative EE position should use ``relative_ee_pos`` before
``StateObservationProcessor`` assembles ``observation.state``. A generic
previous-action processor can then append the action to the state without
creating task-specific keys such as ``dy`` or ``prev_y``.

Refactoring checklist
---------------------

When adding a similar environment, check:

* Is every top-level constant truly fixed, or should it be a config field?
* Can a direct task-frame mapping replace a helper or wrapper?
* Does a custom class own real runtime behavior, or only parameterization?
* Can an observation difference be expressed by an existing or generic shared
  processor?
* Does reset produce the same observation shape as a normal step?
* Do tests protect the intended config shape and state ordering?
