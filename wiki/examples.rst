Examples
========

Real example
------------

- ``examples/demo_mp_net_real.py``
  A UR-focused example that combines:

  - a static home primitive
  - a world-frame move-delta approach
  - an adaptive inspection primitive
  - a scripted open-loop retract authored through ``trajectory=OpenLoopTrajectorySpec(...)``

Mock examples
-------------

- ``examples/demo_mp_net_mock_bookshelf.py``
  Slide across a bookshelf, lean in, then retreat.
- ``examples/demo_mp_net_mock_tea_pour.py``
  Tilt and trace a pouring arc with orientation-heavy task-space motion.
- ``examples/demo_mp_net_mock_dual_arm_handover.py``
  Demonstrates multi-robot MP-Net configs with a synchronized handover.
- ``examples/demo_mp_net.py``
  A smaller baseline MP-Net demo kept around as a compact starting point.

What these examples are meant to teach
--------------------------------------

- how to wire transitions between static, move-delta, and open-loop primitives
- how dynamic primitives resolve entry-time targets from processed observation
- how open-loop primitives keep final target info while sampling per-step setpoints from config
- how multi-robot primitive dictionaries look in practice
- how to build mock demos that are illustrative without depending on hardware
